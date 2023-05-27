#include "k_quants.h"

#include <math.h>
#include <string.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    int weight_type = rmse_type%2;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = weight_type == 1 ? x[i] * x[i] : 1;
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = sumlx/suml2;
    float best = scale * sumlx;
    for (int itry = 0; itry < 3; ++itry) {
        iscale = 1/scale;
        float slx = 0;
        float sl2 = 0;
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            if (l + nmax != L[i]) { changed = true; }
            float w = weight_type == 1 ? x[i] * x[i] : 1.f;
            slx += w*x[i]*l;
            sl2 += w*l*l;
        }
        if (!changed || sl2 == 0 || slx*slx <= best*sl2) { break; }
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        sumlx = slx; suml2 = sl2;
        scale = sumlx/suml2;
        best = scale * sumlx;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = weight_type == 1 ? x[i]*x[i] : 1;
            int l = L[i] - nmax;
            float slx = sumlx - w*x[i]*l;
            if (slx > 0) {
                float sl2 = suml2 - w*l*l;
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MAX(-nmax, MIN(nmax-1, new_l));
                if (new_l != l) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = nmax + new_l; sumlx = slx; suml2 = sl2;
                        scale = sumlx / suml2; best = scale * sumlx;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) { break; }
    }
    if (rmse_type < 3) {
        return scale;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = weight_type == 1 ? x[i] * x[i] : 1;
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}


static void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        float iscale = -128.f/max_scale;
        y[i].d = ggml_fp32_to_fp16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = ggml_fp16_to_fp32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * restrict ql = y[i].ql;
        uint8_t * restrict qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;

    }
}

