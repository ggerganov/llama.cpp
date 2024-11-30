#include <cassert>
#include <algorithm>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "ggml-fp8.h"

union fp32_int32 {
    float f;
    uint32_t bits;
};

template<int E>
inline FP8<E> float_to_fp8(float value) {
    FP8<E> out;
    fp32_int32 in = {value};
    // the sign
    out.bits = (in.bits >> 24) & 0x80;
    // value without sign
    in.bits &= 0x7fffffff;
    //GGML_ASSERT(in.bits < 0x7f800000); // +/- infinity or NAN
    if (in.f >= FP8<E>::MAX) {
        out.bits |= 0x7E;
    } else if (in.f < FP8<E>::MIN) { // => 0.
        // OK: S.0000000
    } else {
        in.f *= exp_f2<FP8<E>::E_BIAS-127>();
        // - trunc
        //uint32_t eps = 0;
        // - rounding half away from zero
        //uint32_t eps = 0x400000>>FP8<E>::M;
        // - rounding half toward zero
        //uint32_t eps = 0x3fffff>>FP8<E>::M;
        // - rounding to nearest even
        uint32_t eps = (0x3fffff>>FP8<E>::M) + ((in.bits >> (23-FP8<E>::M)) & 0x1);
        // shift mantissa.
        in.bits += eps;
        out.bits |= (in.bits >> (23-FP8<E>::M)) & 0x7F;
    }
    return out;
}

template<int E>
inline float fp8_to_float(const FP8<E>& in) {
    fp32_int32 out = {0};
    out.bits = in.bits & 0x80;
    out.bits <<= 24;
    uint32_t _bits = in.bits & 0x7F;
    _bits <<= (23-FP8<E>::M);
    out.bits |= _bits;
    out.f *= exp_f2<127-FP8<E>::E_BIAS>();
    return out.f;
}

template<int E>
static inline void conv(const FP8<E>* x, float* y, int64_t size) {
    for (int64_t i=0; i<size; i++) {
        y[i] = fp8_to_float(x[i]);
    }
}

template<int E>
static inline void conv(const float* x, FP8<E>* y, int64_t size) {
    for (int64_t i=0; i<size; i++) {
        y[i] = float_to_fp8<E>(x[i]);
    }
}

template <int E, int QK>
struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template <int E, int QK>
static inline void conv(const bloc_fp8<E, QK>* x, float* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        for (int64_t i=0; i<QK; i++) {
            y[q*QK+i] = fp8_to_float(x[q].qs[i])*(x[q].d);
        }
    }
}

template <int E, int QK>
static inline void conv(const float* x, bloc_fp8<E, QK>* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float m = 0;
        for (int64_t i=0; i<QK; i++) {
            m = std::max(std::abs(x[q*QK+i]),m);
        }
        const float D = FP8<E>::MAX/m;
        y[q].d = m/FP8<E>::MAX;
        for (int64_t i=0; i<QK; i++) {
            y[q].qs[i] = float_to_fp8<E>(x[q*QK+i]*D);
        }
    }
}

// the C API.
void ggml_e5m2_to_fp32_row(const ggml_e5m2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<5>*>(x), y, k);
}
void ggml_fp32_to_e5m2_row_ref(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<5>*>(y), k);
}

void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<4>*>(x), y, k);
}
void ggml_fp32_to_e4m3_row_ref(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<4>*>(y), k);
}

void dequantize_row_e4m3_q(const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<4, QK_K>*>(x), y, k);
}
void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}

void dequantize_row_e3m4_q(const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<3, QK_K>*>(x), y, k);
}
void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}
