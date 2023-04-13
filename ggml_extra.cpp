#include "ggml_extra.h"
#include "ggml.h"

#include <limits>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include <thread>
#include <atomic>
#include <cstring>

namespace {

constexpr int kChunkSize = 32*32*8;
constexpr int QK = 32;
constexpr int kBucketSize0 = QK/2 + sizeof(float);
constexpr int kBucketSize1 = QK/2 + 2*sizeof(float);

inline int toNearestInt(float fval) {
    assert(fval <= 4194303.f);
    constexpr float kSnapper=3<<22;
    auto val = fval + kSnapper;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Adapted from PR #835, function quantize_row_q4_0_rmse()
//
// I absolutely cannot reproduce the rmse = 0.00185915 reported in #835.
// Instead, I get rmse = 0.00197 with the original and rmse = 0.00192 // with the modification that determines the scale actually minimizing
// the rmse.
//
// Do I have a bug? iI don't see it.
// The only difference is that I'm using toNearestInt()
// instead of round(), but what are the odds for getting scaled weights at
// exactly 2.5, 4.5, and 6.5, where toNearestInt() and round() differ.
// (with toNearestInt() behaving as expected and rounding towards the even integer,
// while round() always rounding up.
float quanizeRmse(int n, const float* X, int8_t* L) {
#define Q4_0_SCALE_CANDIDATE_COUNT 8
    static const float candidates[Q4_0_SCALE_CANDIDATE_COUNT] = { -8.7f, -8.5f, -8.3f, -8.1f, -7.9f, -7.7f, -7.2f, +7.0f };
    float max = 0, amax = 0;
    for (int i=0; i<n; ++i) {
        float ax = std::abs(X[i]);
        if (ax > amax) { amax = ax; max = X[i]; }
    }
    if (!amax) { // all zero
        for (int i=0; i<n; ++i) L[i] = 0;
        return 1.f;
    }
    float best = std::numeric_limits<float>::max(), bestScale = 0;
    for (int si=0; si<Q4_0_SCALE_CANDIDATE_COUNT; ++si) {
        float iscale = candidates[si]/max;
        float err = 0;
        for (int i=0; i<n; ++i) {
            float sx = iscale*X[i];
            int l = std::max(-8, std::min(7, toNearestInt(sx)));
            sx -= l;
            err += sx*sx;
        }
        if (err < best) {
            best = err; bestScale = iscale;
        }
    }
    // The follwoing is a departure from #835. Given the quants produces by bestScale,
    // it determines the scale the actually minimizes the MSE (or RMSE).
    // With this, I get rmse = 0.00192 for the 7B model.
    float sumlx = 0; int suml2 = 0;
    for (int i=0; i<n; ++i) {
        int l = std::max(-8, std::min(7, toNearestInt(bestScale*X[i])));
        sumlx += X[i]*l; suml2 += l*l;
        L[i] = l;
    }
    return sumlx/suml2;
    // The following is what is in quantize_row_q4_0_rmse() in PR #835
    // With this version, I get rmse = 0.00197 for the 7B model.
    //for (int i=0; i<n; ++i) L[i] = std::max(-8, std::min(7, toNearestInt(bestScale*X[i])));
    //return 1/bestScale;
}

float quanizeRmseK(int n, const float* X, int8_t* L,
        int nCandidates, const float* candidates, int nmin, int nmax) {
    float max = 0;
    for (int i=0; i<n; ++i) max = std::max(max, std::abs(X[i]));
    if (!max) { // all zero
        for (int i=0; i<n; ++i) L[i] = 0;
        return 1.f;
    }
    float best = 0, bestScale = 0;
    for (int si=0; si<nCandidates; ++si) {
        float iscale = candidates[si]/max;
        float sumlx = 0; int suml2 = 0;
        for (int i=0; i<n; ++i) {
            int l = std::max(nmin, std::min(nmax, toNearestInt(iscale*X[i])));
            sumlx += X[i]*l; suml2 += l*l;
        }
        if (sumlx*sumlx > best*suml2) {
            best = sumlx*sumlx/suml2; bestScale = iscale;
        }
    }
    float sumlx = 0; int suml2 = 0;
    for (int i=0; i<n; ++i) {
        int l = std::max(nmin, std::min(nmax, toNearestInt(bestScale*X[i])));
        sumlx += X[i]*l; suml2 += l*l;
        L[i] = l;
    }
    float scale = sumlx/suml2;
    best = scale*sumlx;
    for (int itry=0; itry<3; ++itry) {
        bool haveChanges = false;
        for (int i=0; i<n; ++i) {
            auto g = X[i] - scale*L[i];
            if (g > 0 && L[i] < nmax) {
                auto s1 = sumlx + X[i];
                auto s2 = suml2 + 2*L[i] + 1;
                if (s2 > 0 && s1*s1 > best*s2) {
                    scale = s1/s2; best = scale*s1; ++L[i]; sumlx = s1; suml2 = s2; haveChanges = true;
                }
            }
            else if (g < 0 && L[i] > nmin) {
                auto s1 = sumlx - X[i];
                auto s2 = suml2 - 2*L[i] + 1;
                if (s2 > 0 && s1*s1 > best*s2) {
                    scale = s1/s2; best = scale*s1; --L[i]; sumlx = s1; suml2 = s2; haveChanges = true;
                }
            }
        }
        if (!haveChanges) break;
    }
    return scale;
}
// The following improves the above.
// It gives RMSE = 0.00185228 for the 7B model.
float quanizeRmseK7(int n, const float* X, int8_t* L) {
    constexpr int kCandiateCount = 20;
    static const float candidates[kCandiateCount] = { -8.7f, -8.5f, -8.3f, -8.1f, -7.9f, -7.7f, -7.2f, -7.0f, -6.3f, -5.7f,
                                                      +8.7f, +8.5f, +8.3f, +8.1f, +7.9f, +7.7f, +7.2f, +7.0f, +6.3f, +5.7f};
    return quanizeRmseK(n, X, L, kCandiateCount, candidates, -8, 7);
}

float quanizeRmseK15(int n, const float* X, int8_t* L) {
    constexpr int kCandiateCount = 16;
    static const float candidates[kCandiateCount] = {
        +17.75f, +17.25f, +16.75f, +16.25f, +15.75f, +15.25f, +14.75f, +14.25f, +13.75f, +13.25f, +12.75f, +12.25, +11.75f,
        +11.25f, +10.75f, +10.25f
    };
    return quanizeRmseK(n, X, L, kCandiateCount, candidates, 0, 15);
}

float quanizeRmseK31(int n, const float* X, int8_t* L) {
    constexpr int kCandiateCount = 24;
    static const float candidates[kCandiateCount] = {
        +35.25, +34.25f, +33.25f, +32.75f, +32.25f, +31.75f, +31.25f, +30.75f, +30.25f, +29.75f, +29.25f, +28.25f, +27.25f, +26.25f,
        +25.25f, +24.25f, +23.25, +22.25f, +21.25f, +20.25f, +19.25f, +18.25f, +17.25f, +16.25f
    };
    //static const float candidates[kCandiateCount] = {
    //    +33.25f, +32.25f, +31.75f, +31.25f, +30.75f, +30.25f, +30.25f, +29.25f, +28.75f, +27.25f, +26.25f, +25.25f, +24.25f, +23.25, +22.25f,
    //    +21.25f
    //};
    return quanizeRmseK(n, X, L, kCandiateCount, candidates, 0, 31);
}

// Fast (as much faster than doing the optimization), but not very good.
float quanizeRmseFast(int n, const float* X, int8_t* L) {
    //constexpr int kCandiateCount = 3;
    //static const float candidates[kCandiateCount] = { +8.3f, +7.2f, +5.7f};
    constexpr int kCandiateCount = 4;
    static const float candidates[kCandiateCount] = { +8.7f, +7.9f, +7.2f, +5.7f};
    float max = 0;
    for (int i=0; i<n; ++i) max = std::max(max, std::abs(X[i]));
    if (!max) { // all zero
        for (int i=0; i<n; ++i) L[i] = 0;
        return 1.f;
    }
    float best = 0, bestScale = 0;
    for (int si=0; si<kCandiateCount; ++si) {
        float iscale = candidates[si]/max;
        float sumxlp = 0, sumxlm = 0;
        int   suml2p = 0, suml2m = 0;
        for (int i=0; i<n; ++i) {
            float x = X[i];
            float sx = iscale*x;
            int lx = toNearestInt(sx);
            int lp = std::max(-8, std::min(7, +lx));
            int lm = std::max(-8, std::min(7, -lx));
            sumxlp += x*lp;  sumxlm += x*lm;
            suml2p += lp*lp; suml2m += lm*lm;
        }
        if (sumxlp*sumxlp*suml2m >= sumxlm*sumxlm*suml2p) {
            if (sumxlp*sumxlp > best*suml2p) {
                best = sumxlp*sumxlp/suml2p; bestScale = iscale;
            }
        } else {
            if (sumxlm*sumxlm > best*suml2m) {
                best = sumxlm*sumxlm/suml2m; bestScale = -iscale;
            }
        }
    }
    float sumlx = 0; int suml2 = 0;
    for (int i=0; i<n; ++i) {
        int l = std::max(-8, std::min(7, toNearestInt(bestScale*X[i])));
        sumlx += X[i]*l; suml2 += l*l;
        L[i] = l;
    }
    return sumlx/suml2;
}

float quanizeRmseOpt(int n, const float* X, int8_t* L, std::vector<std::pair<float,int>>& work) {
    work.clear();
    work.reserve(n*17);
    for (int l=-8; l<=8; ++l) {
        float scale = l - 0.4999f;
        for (int i=0; i<n; ++i) {
            if (X[i]) work.push_back({scale/std::abs(X[i]), i});
        }
    }
    for (int i=0; i<n; ++i) L[i] = 0;
    if (work.empty()) return 1.f; // all values are zero
    std::sort(work.begin(), work.end());
    float best = 0, bestScale = 0, lasts = work.front().first - 1;
    double sumlx = 0; int suml2 = 0;
    for (int k=0; k<int(work.size()); ++k) {
        float s = work[k].first; int i = work[k].second;
        int l = std::max(-8, std::min(7, toNearestInt(s*X[i])));
        if (l != L[i]) {
            sumlx += X[i]*(l-L[i]); suml2 += l*l - L[i]*L[i];
            L[i] = l;
            if ((s != lasts || k == int(work.size())-1) && suml2 > 0 && sumlx*sumlx > best*suml2) {
                best = sumlx*sumlx/suml2; bestScale = s;
            }
        }
    }
    sumlx = 0; suml2 = 0;
    for (int i=0; i<n; ++i) {
        int l = std::max(-8, std::min(7, toNearestInt(bestScale*X[i])));
        sumlx += X[i]*l; suml2 += l*l;
        L[i] = l;
    }
    return sumlx/suml2;
}

std::pair<float, float> kQuantize0(int n, const float* X, int8_t* L, std::vector<std::pair<float,int>>& work, int nmin, int nmax) {
    work.clear();
    work.reserve(n*(nmax+2));
    float max = 0; int imax = -1;
    for (int i=0; i<n; ++i) {
        float x = std::abs(X[i]);
        if (x > max) { max = x; imax = i; }
    }
    if (imax < 0) {  // all X are zero
        for (int i=0; i<n; ++i) L[i] = 0;
        return {1.f, 0.f};
    }
    float maxi = 1/max;
    int kmin, kmax;
    {
        float scale0 = nmax*maxi;
        double sumlx0 = 0; int suml20 = 0;
        for (int i=0; i<n; ++i) {
            int l = std::max(nmin, std::min(nmax, toNearestInt(scale0*X[i])));
            sumlx0 += X[i]*l; suml20 += l*l;
        }
        auto df0 = suml20/scale0 - sumlx0;
        if (df0 > 0) {
            kmin = nmax-2; kmax = nmax+1;
        } else {
            kmin = nmax/2; kmax = nmax+1;
        }
    }
    for (int k=kmin; k<=kmax; ++k) work.push_back({(k + 0.501f)*maxi, imax});
    float minScale = work.front().first;
    float maxScale = work.back().first;
    for (int i=0; i<n; ++i) {
        L[i] = 0;
        auto x = std::abs(X[i]);
        if (i == imax || !x) continue;
        int kkmin = std::max(0, int(minScale*x-0.501f));
        int kkmax = std::min(kmax, int(maxScale*x));
        auto xi = 1/x;
        for (int k=kkmin; k<=kkmax; ++k) {
            auto s = (k + 0.501f)*xi;
            if (s > maxScale) break;
            if (s > minScale) work.push_back({s,i});
        }
    }
    std::sort(work.begin(), work.end());
    float sumlx = 0; int suml2 = 0;
    float s = work.front().first;
    for (int i=0; i<n; ++i) {
        int l = std::max(nmin, std::min(nmax, toNearestInt(s*X[i])));
        sumlx += X[i]*l; suml2 += l*l;
        L[i] = l;
    }
    float bestSumlx = sumlx, bestSumlx2 = sumlx*sumlx; int bestSuml2 = suml2; float bests = s;
    float lasts = s;
    for (int k=1; k<int(work.size()); ++k) {
        s = work[k].first; int i = work[k].second;
        int l = std::max(nmin, std::min(nmax, toNearestInt(s*X[i])));
        if (l == L[i]) { lasts = s; continue; }
        if (l > L[i]) {
            sumlx += X[i];
            suml2 += 1 + 2*L[i];
        }
        else {
            sumlx -= X[i];
            suml2 += 1 - 2*L[i];
        }
        L[i] = l;
        float sumlx2 = sumlx*sumlx;
        if ((s != lasts || k == int(work.size())-1) && suml2 > 0 && sumlx2*bestSuml2 > bestSumlx2*suml2) {
            bestSumlx = sumlx; bestSumlx2 = sumlx2; bestSuml2 = suml2; bests = s;
        }
        lasts = s;
    }
    for (int i=0; i<n; ++i) L[i] = std::max(nmin, std::min(nmax, toNearestInt(bests*X[i])));
    return {bestSumlx/bestSuml2, bestSumlx*bestSumlx/bestSuml2};
}

std::pair<float, float> kQuantize1(int n, const float* X, int8_t* L, std::vector<float>& tmpX,
        std::vector<std::pair<float,int>>& work, int nmax) {
    float min = X[0], max = X[1];
    for (int i=1; i<n; ++i) {
        min = std::min(min, X[i]); max = std::max(max, X[i]);
    }
    if (max == min) {
        for (int i=0; i<n; ++i) L[i] = 0;
        return {min, 1.f};
    }
    if (int(tmpX.size()) < n) tmpX.resize(n);
    double a = min, b = 0;
    for (int itry=0; itry<5; ++itry) {
        for (int i=0; i<n; ++i) tmpX[i] = X[i] - a;
        if (nmax == 7) quanizeRmseK15(n, tmpX.data(), L);
        else if (nmax == 15) quanizeRmseK31(n, tmpX.data(), L);
        else kQuantize0(n, tmpX.data(), L, work, 0, 2*nmax+1);
        double sumlx = 0, sumx = 0;
        int suml2 = 0, suml = 0;
        for (int i=0; i<n; ++i) {
            auto l = L[i];
            sumlx += X[i]*l;
            suml2 += l*l;
            suml  += l;
            sumx  += X[i];
        }
        int64_t D = suml2*n - suml*suml;
        auto aold = a, bold = b;
        a = (sumx*suml2 - sumlx*suml)/D;
        b = (sumlx*n - sumx*suml)/D;
        if (itry > 0 && std::abs(a - aold) < 1e-6*std::abs(aold) && std::abs(b - bold) < 1e-6*std::abs(bold)) break;
    }
    return {a, b};
}

std::pair<float, float> kQuantize1Fast(int n, const float* X, int8_t* L, int nmax) {
    float min = X[0], max = X[1];
    for (int i=1; i<n; ++i) {
        min = std::min(min, X[i]); max = std::max(max, X[i]);
    }
    if (max == min) {
        for (int i=0; i<n; ++i) L[i] = 0;
        return {min, 1.f};
    }
    float scale = (nmax - 0.499f)/(max - min);
    double sumlx = 0, sumx = 0;
    int suml2 = 0, suml = 0;
    for (int i=0; i<n; ++i) {
        int l = toNearestInt(scale*(X[i] - min));
        L[i] = l;
        sumlx += X[i]*l;
        suml2 += l*l;
        suml  += l;
        sumx  += X[i];
    }
    int64_t D = suml2*n - suml*suml;
    double a = (sumx*suml2 - sumlx*suml)/D;
    double b = (sumlx*n - sumx*suml)/D;
    return {a, b};
}

void kQuantizeQ4(const float* X, void* buffer, int k, int type) {
    assert(k % QK == 0);

    auto processOne = [type] (const float* X, int8_t* L, char* y, std::vector<std::pair<float, int>>& work, std::vector<float>& tmpX) {
        auto q = (uint8_t*)y;
        if (type == 0) {
            auto scale = quanizeRmseK7(QK, X, L);
            //auto scale = quanizeRmseFast(QK, X, L);
            //auto scale = quanizeRmseOpt(QK, X, L, work);
            // The following is not quite as good as quanizeRmseK() and it is slower too.
            //if (int(tmpX.size()) < QK) tmpX.resize(QK);
            //auto r1 = kQuantize0(QK, X, L, work, -8, 7);
            //for (int i=0; i<QK; ++i) tmpX[i] = -X[i];
            //int8_t L2[QK];
            //auto r2 = kQuantize0(QK, tmpX.data(), L2, work, -8, 7);
            //float scale = r1.first;
            //if (r2.second > r1.first) {
            //    scale = -r2.first;
            //    std::memcpy(L, L2, QK);
            //}
            ////float scale = kQuantize0(QK, X, L, work, -7, 7);
            std::memcpy(q, &scale, sizeof(scale)); q += sizeof(scale);
            for (int k=0; k<QK/2; ++k) q[k] = (L[2*k] + 8) | ((L[2*k+1] + 8) << 4);
        } else if (type == 1) {
            auto result = kQuantize1(QK, X, L, tmpX, work, 7);
            std::memcpy(q, &result.second, sizeof(result.second)); q += sizeof(result.second);
            std::memcpy(q, &result.first,  sizeof(result.first));  q += sizeof(result.first);
            for (int k=0; k<QK/2; ++k) q[k] = L[2*k] | (L[2*k+1] << 4);
        } else if (type == 4) {
            auto scale1 = quanizeRmseK7(QK/2, X, L);
            auto scale2 = quanizeRmseK7(QK/2, X+QK/2, L+QK/2);
            //printf("scale1 = %g, scale2 = %g\n",scale1,scale2);
            auto scale1fp16 = ggml_fp32_to_fp16(scale1);
            auto scale2fp16 = ggml_fp32_to_fp16(scale2);
            std::memcpy(q, &scale1fp16, sizeof(scale1fp16)); q += sizeof(scale1fp16);
            std::memcpy(q, &scale2fp16, sizeof(scale2fp16)); q += sizeof(scale2fp16);
            for (int k=0; k<QK/2; ++k) q[k] = (L[2*k] + 8) | ((L[2*k+1] + 8) << 4);
        } else if (type == 5) {
            auto result1 = kQuantize1(QK/2, X, L, tmpX, work, 7);
            auto result2 = kQuantize1(QK/2, X + QK/2, L + QK/2, tmpX, work, 7);
            auto a1fp16 = ggml_fp32_to_fp16(result1.first);
            auto b1fp16 = ggml_fp32_to_fp16(result1.second);
            auto a2fp16 = ggml_fp32_to_fp16(result2.first);
            auto b2fp16 = ggml_fp32_to_fp16(result2.second);
            std::memcpy(q, &a1fp16, sizeof(a1fp16)); q += sizeof(a1fp16);
            std::memcpy(q, &b1fp16, sizeof(b1fp16)); q += sizeof(b1fp16);
            std::memcpy(q, &a2fp16, sizeof(a2fp16)); q += sizeof(a2fp16);
            std::memcpy(q, &b2fp16, sizeof(b2fp16)); q += sizeof(b2fp16);
            for (int k=0; k<QK/2; ++k) q[k] = L[2*k] | (L[2*k+1] << 4);
        } else {
            auto result = type == 2 ? kQuantize1(QK, X, L, tmpX, work, 15) : kQuantize1Fast(QK, X, L, 31);
            auto afp16 = ggml_fp32_to_fp16(result.first);
            auto bfp16 = ggml_fp32_to_fp16(result.second);
            std::memcpy(q, &afp16, sizeof(afp16)); q += sizeof(afp16);
            std::memcpy(q, &bfp16, sizeof(bfp16)); q += sizeof(bfp16);
            auto u = (uint32_t*)q;
            *u = 0;
            q += sizeof(uint32_t);
            uint32_t m = 1u;
            for (int k=0; k<QK/2; ++k) {
                auto l1 = L[2*k], l2 = L[2*k+1];
                if (l1 > 15) { l1 -= 16; *u |= m; }
                m <<= 1;
                if (l2 > 15) { l2 -= 16; *u |= m; }
                m <<= 1;
                q[k] = l1 | (l2 << 4);
            }
        }
    };

    auto bucketSize = type == 0 || type == 4 ? kBucketSize0 : kBucketSize1;
    auto y = (char*)buffer;
    int nchunk = (k + kChunkSize-1)/kChunkSize;
    if (nchunk < 2) {
        std::vector<int8_t> L(QK);
        std::vector<std::pair<float,int>> work;
        std::vector<float> tmpX;
        int nb = k / QK;
        auto x = X;
        for (int i=0; i<nb; ++i) {
            processOne(x, L.data(), y, work, tmpX);
            y += bucketSize; x += QK;
        }
        return;
    }

    std::atomic<int> counter(0);
    auto compute = [&counter, X, y, k, bucketSize, &processOne] () {
        std::vector<int8_t> L(QK);
        std::vector<std::pair<float,int>> work;
        std::vector<float> tmpX;
        while (true) {
            int first = counter.fetch_add(kChunkSize, std::memory_order_relaxed);
            if (first >= k) break;
            int last = first + kChunkSize;
            if (last > k) last = k;
            auto xi = X + first;
            auto yi = y + (first/QK)*bucketSize;
            int n = (last - first)/QK;
            for (int i=0; i<n; ++i) {
                processOne(xi, L.data(), yi, work, tmpX);
                yi += bucketSize; xi += QK;
            }
        }
    };
    int nthread = std::min(nchunk, int(std::thread::hardware_concurrency()));
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();
}

void collectHisto(int k, const void* buffer, int64_t* hist, int type) {
    if (!hist) return;
    auto y = (const uint8_t*)buffer;
    int m = type == 0 ? 4 : 8;
    int n = k / 32;
    for (int i=0; i<n; ++i) {
        y += m;
        for (int l=0; l<16; ++l) {
            ++hist[y[l] & 15];
            ++hist[y[l] >> 4];
        }
        y += 16;
    }
}

}

extern "C" {

void kQuantizeQ4_0(const float* x, void* buffer, int k) {
    kQuantizeQ4(x, buffer, k, 0);
}

void kQuantizeQ4_1(const float* x, void* buffer, int k) {
    kQuantizeQ4(x, buffer, k, 1);
}

void kQuantizeQ5_1(const float* x, void* buffer, int k) {
    kQuantizeQ4(x, buffer, k, 2);
}

void kQuantizeQ5_1_Fast(const float* x, void* buffer, int k) {
    kQuantizeQ4(x, buffer, k, 3);
}

size_t kQuantizeQ4_0H(const float* x, void* buffer, int k, int64_t* hist) {
    kQuantizeQ4(x, buffer, k, 0);
    collectHisto(k, buffer, hist, 0);
    return (k / QK) * kBucketSize0;
}

size_t kQuantizeQ4_1H(const float* x, void* buffer, int k, int64_t* hist) {
    kQuantizeQ4(x, buffer, k, 1);
    collectHisto(k, buffer, hist, 1);
    return (k / QK) * kBucketSize1;
}

size_t kQuantizeQ5_1H(const float* x, void* buffer, int k, int64_t* hist) {
    kQuantizeQ4(x, buffer, k, 2);
    collectHisto(k, buffer, hist, 1);
    return (k / QK) * kBucketSize1;
}

size_t kQuantizeQ5_1H_Fast(const float* x, void* buffer, int k, int64_t* hist) {
    kQuantizeQ4(x, buffer, k, 3);
    collectHisto(k, buffer, hist, 1);
    return (k / QK) * kBucketSize1;
}

void kDequantizeQ5_1(const void* x, float* y, int k) {
    assert(k % QK == 0);
    int n = k / QK;
    auto data = (const uint8_t*)x;
    for (int i=0; i<n; ++i) {
        ggml_fp16_t afp16, bfp16;
        std::memcpy(&afp16, data, sizeof(afp16)); data += sizeof(afp16);
        std::memcpy(&bfp16, data, sizeof(bfp16)); data += sizeof(bfp16);
        auto a = ggml_fp16_to_fp32(afp16);
        auto b = ggml_fp16_to_fp32(bfp16);
        uint32_t u;
        std::memcpy(&u, data, sizeof(u)); data += sizeof(u);
        uint32_t m = 1u;
        for (int k=0; k<16; ++k) {
            auto l1 = data[k] & 15, l2 = data[k] >> 4;
            if (u & m) l1 += 16;
            m <<= 1;
            if (u & m) l2 += 16;
            m <<= 1;
            *y++ = a + b*l1;
            *y++ = a + b*l2;
        }
        data += 16;
    }
}

void kQuantizeQ4_0K(const float* x, void* buffer, int k) {
    kQuantizeQ4(x, buffer, k, 4);
}

void kDequantizeQ4_0K(const void* x, float* y, int k) {
    assert(k % QK == 0);
    int n = k / QK;
    auto data = (const uint8_t*)x;
    for (int i=0; i<n; ++i) {
        ggml_fp16_t afp16, bfp16;
        std::memcpy(&afp16, data, sizeof(afp16)); data += sizeof(afp16);
        std::memcpy(&bfp16, data, sizeof(bfp16)); data += sizeof(bfp16);
        auto a = ggml_fp16_to_fp32(afp16);
        auto b = ggml_fp16_to_fp32(bfp16);
        for (int k=0; k<8; ++k) {
            int8_t l1 = data[k] & 15, l2 = data[k] >> 4;
            l1 -= 8; l2 -= 8;
            *y++ = a*l1; *y++ = a*l2;
        }
        data += 8;
        for (int k=0; k<8; ++k) {
            int8_t l1 = data[k] & 15, l2 = data[k] >> 4;
            l1 -= 8; l2 -= 8;
            *y++ = b*l1; *y++ = b*l2;
        }
        data += 8;
    }
}

void kQuantizeQ4_1K(const float* x, void* buffer, int k) {
    kQuantizeQ4(x, buffer, k, 5);
}

void kDequantizeQ4_1K(const void* x, float* y, int k) {
    assert(k % QK == 0);
    int n = k / QK;
    auto data = (const uint8_t*)x;
    for (int i=0; i<n; ++i) {
        ggml_fp16_t a1fp16, b1fp16, a2fp16, b2fp16;
        std::memcpy(&a1fp16, data, sizeof(a1fp16)); data += sizeof(a1fp16);
        std::memcpy(&b1fp16, data, sizeof(b1fp16)); data += sizeof(b1fp16);
        std::memcpy(&a2fp16, data, sizeof(a2fp16)); data += sizeof(a2fp16);
        std::memcpy(&b2fp16, data, sizeof(b2fp16)); data += sizeof(b2fp16);
        auto a1 = ggml_fp16_to_fp32(a1fp16);
        auto b1 = ggml_fp16_to_fp32(b1fp16);
        auto a2 = ggml_fp16_to_fp32(a2fp16);
        auto b2 = ggml_fp16_to_fp32(b2fp16);
        for (int k=0; k<8; ++k) {
            int8_t l1 = data[k] & 15, l2 = data[k] >> 4;
            *y++ = a1 + b1*l1; *y++ = a1 + b1*l2;
        }
        data += 8;
        for (int k=0; k<8; ++k) {
            int8_t l1 = data[k] & 15, l2 = data[k] >> 4;
            *y++ = a2 + b2*l1; *y++ = a2 + b2*l2;
        }
        data += 8;
    }
}

void kQuantizeQ8Simple(const float* x, void* y, int k) {
    assert(k % QK == 0);
    auto data = (int8_t*)y;
    int n = k / (QK/2);
    for (int i=0; i<n; ++i) {
        float max = 0;
        for (int k=0; k<16; ++k) max = std::max(max, std::abs(x[k]));
        if (max > 0) {
            float iscale = 127.f/max;
            float scale = max/127.f;
            std::memcpy(data, &scale, sizeof(scale)); data += sizeof(scale);
            for (int k=0; k<16; ++k) data[k] = toNearestInt(iscale * *x++);
            data += 16;
        } else {
            float scale = 1;
            std::memcpy(data, &scale, sizeof(scale)); data += sizeof(scale);
            auto aux = (uint32_t*)data;
            aux[0] = aux[1] = aux[2] = aux[3] = 0;
            data += 16;
        }
    }
}

void kDequantizeQ8(const void* x, float* y, int k) {
    assert(k % QK == 0);
    auto data = (const int8_t*)x;
    int n = k / (QK/2);
    for (int i=0; i<n; ++i) {
        float scale;
        std::memcpy(&scale, data, sizeof(scale)); data += sizeof(scale);
        for (int k=0; k<16; ++k) *y++ = scale*data[k];
        data += 16;
    }
}

}
