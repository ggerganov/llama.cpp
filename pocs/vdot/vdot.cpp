#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <array>

#include <ggml.h>

constexpr int kVecSize = 1 << 18;

float drawFromGaussianPdf(std::mt19937& rndm) {
    constexpr double kScale = 1./(1. + std::mt19937::max());
    constexpr double kTwoPiTimesScale = 6.28318530717958647692*kScale;
    static float lastX;
    static bool haveX = false;
    if (haveX) { haveX = false; return lastX; }
    auto r = sqrt(-2*log(1 - kScale*rndm()));
    auto phi = kTwoPiTimesScale * rndm();
    lastX = r*sin(phi);
    haveX = true;
    return r*cos(phi);
}
void fillRandomGaussianFloats(std::vector<float>& values, std::mt19937& rndm, float mean = 0) {
    for (auto& v : values) v = mean + drawFromGaussianPdf(rndm);
}

// Copy-pasted from ggml.c
#define QK4_0 32
typedef struct {
    float   d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

// Copy-pasted from ggml.c
#define QK8_0 32
typedef struct {
    float   d;          // delta
    int8_t  qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

// "Scalar" dot product between the quantized vector x and float vector y
inline double dot(int n, const block_q4_0* x, const float* y) {
    const static float kValues[16] = {-8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    constexpr uint32_t kMask1 = 0x0f0f0f0f;
    uint32_t u1, u2;
    auto q1 = (const uint8_t*)&u1;
    auto q2 = (const uint8_t*)&u2;
    double sum = 0;
    for (int i=0; i<n; ++i) {
        float d = x->d;
        auto u = (const uint32_t*)x->qs;
        float s = 0;
        for (int k=0; k<4; ++k) {
            u1 = u[k] & kMask1;
            u2 = (u[k] >> 4) & kMask1;
            s += y[0]*kValues[q1[0]] + y[1]*kValues[q2[0]] +
                 y[2]*kValues[q1[1]] + y[3]*kValues[q2[1]] +
                 y[4]*kValues[q1[2]] + y[5]*kValues[q2[2]] +
                 y[6]*kValues[q1[3]] + y[7]*kValues[q2[3]];
            y += 8;
        }
        sum += s*d;
        ++x;
    }
    return sum;
}
// Alternative version of the above. Faster on my Mac (~45 us vs ~55 us per dot product),
// but about the same on X86_64 (Ryzen 7950X CPU).
inline double dot3(int n, const block_q4_0* x, const float* y) {
    const static std::pair<float,float> kValues[256] = {
        {-8.f, -8.f}, {-7.f, -8.f}, {-6.f, -8.f}, {-5.f, -8.f}, {-4.f, -8.f}, {-3.f, -8.f}, {-2.f, -8.f}, {-1.f, -8.f},
        { 0.f, -8.f}, { 1.f, -8.f}, { 2.f, -8.f}, { 3.f, -8.f}, { 4.f, -8.f}, { 5.f, -8.f}, { 6.f, -8.f}, { 7.f, -8.f},
        {-8.f, -7.f}, {-7.f, -7.f}, {-6.f, -7.f}, {-5.f, -7.f}, {-4.f, -7.f}, {-3.f, -7.f}, {-2.f, -7.f}, {-1.f, -7.f},
        { 0.f, -7.f}, { 1.f, -7.f}, { 2.f, -7.f}, { 3.f, -7.f}, { 4.f, -7.f}, { 5.f, -7.f}, { 6.f, -7.f}, { 7.f, -7.f},
        {-8.f, -6.f}, {-7.f, -6.f}, {-6.f, -6.f}, {-5.f, -6.f}, {-4.f, -6.f}, {-3.f, -6.f}, {-2.f, -6.f}, {-1.f, -6.f},
        { 0.f, -6.f}, { 1.f, -6.f}, { 2.f, -6.f}, { 3.f, -6.f}, { 4.f, -6.f}, { 5.f, -6.f}, { 6.f, -6.f}, { 7.f, -6.f},
        {-8.f, -5.f}, {-7.f, -5.f}, {-6.f, -5.f}, {-5.f, -5.f}, {-4.f, -5.f}, {-3.f, -5.f}, {-2.f, -5.f}, {-1.f, -5.f},
        { 0.f, -5.f}, { 1.f, -5.f}, { 2.f, -5.f}, { 3.f, -5.f}, { 4.f, -5.f}, { 5.f, -5.f}, { 6.f, -5.f}, { 7.f, -5.f},
        {-8.f, -4.f}, {-7.f, -4.f}, {-6.f, -4.f}, {-5.f, -4.f}, {-4.f, -4.f}, {-3.f, -4.f}, {-2.f, -4.f}, {-1.f, -4.f},
        { 0.f, -4.f}, { 1.f, -4.f}, { 2.f, -4.f}, { 3.f, -4.f}, { 4.f, -4.f}, { 5.f, -4.f}, { 6.f, -4.f}, { 7.f, -4.f},
        {-8.f, -3.f}, {-7.f, -3.f}, {-6.f, -3.f}, {-5.f, -3.f}, {-4.f, -3.f}, {-3.f, -3.f}, {-2.f, -3.f}, {-1.f, -3.f},
        { 0.f, -3.f}, { 1.f, -3.f}, { 2.f, -3.f}, { 3.f, -3.f}, { 4.f, -3.f}, { 5.f, -3.f}, { 6.f, -3.f}, { 7.f, -3.f},
        {-8.f, -2.f}, {-7.f, -2.f}, {-6.f, -2.f}, {-5.f, -2.f}, {-4.f, -2.f}, {-3.f, -2.f}, {-2.f, -2.f}, {-1.f, -2.f},
        { 0.f, -2.f}, { 1.f, -2.f}, { 2.f, -2.f}, { 3.f, -2.f}, { 4.f, -2.f}, { 5.f, -2.f}, { 6.f, -2.f}, { 7.f, -2.f},
        {-8.f, -1.f}, {-7.f, -1.f}, {-6.f, -1.f}, {-5.f, -1.f}, {-4.f, -1.f}, {-3.f, -1.f}, {-2.f, -1.f}, {-1.f, -1.f},
        { 0.f, -1.f}, { 1.f, -1.f}, { 2.f, -1.f}, { 3.f, -1.f}, { 4.f, -1.f}, { 5.f, -1.f}, { 6.f, -1.f}, { 7.f, -1.f},
        {-8.f,  0.f}, {-7.f,  0.f}, {-6.f,  0.f}, {-5.f,  0.f}, {-4.f,  0.f}, {-3.f,  0.f}, {-2.f,  0.f}, {-1.f,  0.f},
        { 0.f,  0.f}, { 1.f,  0.f}, { 2.f,  0.f}, { 3.f,  0.f}, { 4.f,  0.f}, { 5.f,  0.f}, { 6.f,  0.f}, { 7.f,  0.f},
        {-8.f,  1.f}, {-7.f,  1.f}, {-6.f,  1.f}, {-5.f,  1.f}, {-4.f,  1.f}, {-3.f,  1.f}, {-2.f,  1.f}, {-1.f,  1.f},
        { 0.f,  1.f}, { 1.f,  1.f}, { 2.f,  1.f}, { 3.f,  1.f}, { 4.f,  1.f}, { 5.f,  1.f}, { 6.f,  1.f}, { 7.f,  1.f},
        {-8.f,  2.f}, {-7.f,  2.f}, {-6.f,  2.f}, {-5.f,  2.f}, {-4.f,  2.f}, {-3.f,  2.f}, {-2.f,  2.f}, {-1.f,  2.f},
        { 0.f,  2.f}, { 1.f,  2.f}, { 2.f,  2.f}, { 3.f,  2.f}, { 4.f,  2.f}, { 5.f,  2.f}, { 6.f,  2.f}, { 7.f,  2.f},
        {-8.f,  3.f}, {-7.f,  3.f}, {-6.f,  3.f}, {-5.f,  3.f}, {-4.f,  3.f}, {-3.f,  3.f}, {-2.f,  3.f}, {-1.f,  3.f},
        { 0.f,  3.f}, { 1.f,  3.f}, { 2.f,  3.f}, { 3.f,  3.f}, { 4.f,  3.f}, { 5.f,  3.f}, { 6.f,  3.f}, { 7.f,  3.f},
        {-8.f,  4.f}, {-7.f,  4.f}, {-6.f,  4.f}, {-5.f,  4.f}, {-4.f,  4.f}, {-3.f,  4.f}, {-2.f,  4.f}, {-1.f,  4.f},
        { 0.f,  4.f}, { 1.f,  4.f}, { 2.f,  4.f}, { 3.f,  4.f}, { 4.f,  4.f}, { 5.f,  4.f}, { 6.f,  4.f}, { 7.f,  4.f},
        {-8.f,  5.f}, {-7.f,  5.f}, {-6.f,  5.f}, {-5.f,  5.f}, {-4.f,  5.f}, {-3.f,  5.f}, {-2.f,  5.f}, {-1.f,  5.f},
        { 0.f,  5.f}, { 1.f,  5.f}, { 2.f,  5.f}, { 3.f,  5.f}, { 4.f,  5.f}, { 5.f,  5.f}, { 6.f,  5.f}, { 7.f,  5.f},
        {-8.f,  6.f}, {-7.f,  6.f}, {-6.f,  6.f}, {-5.f,  6.f}, {-4.f,  6.f}, {-3.f,  6.f}, {-2.f,  6.f}, {-1.f,  6.f},
        { 0.f,  6.f}, { 1.f,  6.f}, { 2.f,  6.f}, { 3.f,  6.f}, { 4.f,  6.f}, { 5.f,  6.f}, { 6.f,  6.f}, { 7.f,  6.f},
        {-8.f,  7.f}, {-7.f,  7.f}, {-6.f,  7.f}, {-5.f,  7.f}, {-4.f,  7.f}, {-3.f,  7.f}, {-2.f,  7.f}, {-1.f,  7.f},
        { 0.f,  7.f}, { 1.f,  7.f}, { 2.f,  7.f}, { 3.f,  7.f}, { 4.f,  7.f}, { 5.f,  7.f}, { 6.f,  7.f}, { 7.f,  7.f}
    };
    double sum = 0;
    for (int i=0; i<n; ++i) {
        float d = x->d;
        auto q = x->qs;
        float s = 0;
        for (int k=0; k<4; ++k) {
            s += y[0]*kValues[q[0]].first + y[1]*kValues[q[0]].second +
                 y[2]*kValues[q[1]].first + y[3]*kValues[q[1]].second +
                 y[4]*kValues[q[2]].first + y[5]*kValues[q[2]].second +
                 y[6]*kValues[q[3]].first + y[7]*kValues[q[3]].second;
            y += 8; q += 4;
        }
        sum += s*d;
        ++x;
    }
    return sum;
}

// Copy-pasted from ggml.c
static void quantize_row_q8_0_reference(const float *x, block_q8_0 *y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK8_0; l++) {
            const float v = x[i*QK8_0 + l];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK8_0; ++l) {
            const float   v  = x[i*QK8_0 + l]*id;
            y[i].qs[l] = roundf(v);
        }
    }
}

// Copy-pasted from ggml.c
static void dot_q4_q8(const int n, float* s, const void* vx, const void* vy) {
    const int nb = n / QK8_0;
    const block_q4_0* x = (const block_q4_0*)vx;
    const block_q8_0* y = (const block_q8_0*)vy;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        const uint8_t * p0 = x[i].qs;
        const  int8_t * p1 = y[i].qs;

        int sumi = 0;
        for (int j = 0; j < QK8_0/2; j++) {
            const uint8_t v0 = p0[j];

            const int i0 = (int8_t) (v0 & 0xf) - 8;
            const int i1 = (int8_t) (v0 >> 4)  - 8;

            const int i2 = p1[2*j + 0];
            const int i3 = p1[2*j + 1];

            sumi += i0*i2 + i1*i3;
        }
        sumf += d0*d1*sumi;
    }
    *s = sumf;
}

int main(int argc, char** argv) {

    int nloop = argc > 1 ? atoi(argv[1]) : 10;
    bool scalar = argc > 2 ? atoi(argv[2]) : false;

    std::mt19937 rndm(1234);

    auto funcs = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_0);

    int n4 = kVecSize / QK4_0; n4 = 64*((n4 + 63)/64);
    int n8 = kVecSize / QK8_0; n8 = 64*((n8 + 63)/64);
    std::vector<float> x1(kVecSize), y1(kVecSize);
    std::vector<block_q4_0> q4(n4);
    std::vector<block_q8_0> q8(n8);
    std::vector<int64_t> H(16, 0);
    double sumt = 0, sumt2 = 0, maxt = 0;
    double sumqt = 0, sumqt2 = 0, maxqt = 0;
    double sum = 0, sumq = 0;
    for (int iloop=0; iloop<nloop; ++iloop) {

        // Fill vector x with random numbers
        fillRandomGaussianFloats(x1, rndm);

        // Fill vector y with random numbers
        fillRandomGaussianFloats(y1, rndm);

        // quantize x.
        // Note, we do not include this in the timing as in practical application
        // we already have the quantized model weights.
        ggml_quantize_q4_0(x1.data(), q4.data(), kVecSize, QK4_0, H.data());

        // Now measure time the dot product needs using the "scalar" version above
        auto t1 = std::chrono::high_resolution_clock::now();
        sum += dot(kVecSize / QK4_0, q4.data(), y1.data());
        auto t2 = std::chrono::high_resolution_clock::now();
        auto t = 1e-3*std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        sumt += t; sumt2 += t*t; maxt = std::max(maxt, t);

        // And now measure the time needed to quantize y and perform the dot product with the quantized y
        t1 = std::chrono::high_resolution_clock::now();
        float result;
        if (scalar) {
            quantize_row_q8_0_reference(y1.data(), q8.data(), kVecSize);
            dot_q4_q8(kVecSize, &result, q4.data(), q8.data());
        }
        else {
            funcs.quantize_row_q_dot(y1.data(), q8.data(), kVecSize);
            funcs.vec_dot_q(kVecSize, &result, q4.data(), q8.data());
        }
        sumq += result;
        t2 = std::chrono::high_resolution_clock::now();
        t = 1e-3*std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        sumqt += t; sumqt2 += t*t; maxqt = std::max(maxqt, t);

    }

    // Report the time (and the average of the dot products so the compiler does not come up with the idea
    // of optimizing away the function calls after figuring that the result is not used).
    sum /= nloop; sumq /= nloop;
    printf("<dot> = %g, %g\n",sum,sumq);
    sumt /= nloop; sumt2 /= nloop; sumt2 -= sumt*sumt;
    if (sumt2 > 0) sumt2 = sqrt(sumt2);
    printf("time = %g +/- %g us. maxt = %g us\n",sumt,sumt2,maxt);
    sumqt /= nloop; sumqt2 /= nloop; sumqt2 -= sumqt*sumqt;
    if (sumqt2 > 0) sumqt2 = sqrt(sumqt2);
    printf("timeq = %g +/- %g us. maxt = %g us\n",sumqt,sumqt2,maxqt);
    return 0;
}
