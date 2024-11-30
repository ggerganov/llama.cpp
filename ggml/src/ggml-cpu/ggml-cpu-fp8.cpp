#include <cassert>
#include <algorithm>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "ggml-cpu-fp8.h"

namespace fp8 {
union fp32_int32 {
    float f;
    uint32_t bits;
};

#ifdef GGML_USE_OPENMP_SIMD
#pragma omp declare simd
#endif
template<int E>
inline uint8_t from_float(float value) {
    FP8<E> out;
    fp32_int32 in = {value};
    out.bits = (in.bits >> 24) & 0x80;
    in.bits &= 0x7fffffff;
    if (in.f >= FP8<E>::MAX) {
        out.bits |= 0x7E;
    } else if (in.f < FP8<E>::MIN) { // => 0.
    } else {
        in.f *= exp_f2<FP8<E>::E_BIAS-127>();
        uint32_t eps = (0x3fffff>>FP8<E>::M) + ((in.bits >> (23-FP8<E>::M)) & 0x1);
        in.bits += eps;
        out.bits |= (in.bits >> (23-FP8<E>::M)) & 0x7F;
    }
    return out.bits;
}

#ifdef GGML_USE_OPENMP_SIMD
#pragma omp declare simd
#endif
template<int E>
inline float to_float(const FP8<E>& in) {
    fp32_int32 out = {0};
    out.bits = in.bits & 0x80;
    out.bits <<= 24;
    uint32_t _bits = in.bits & 0x7F;
    _bits <<= (23-FP8<E>::M);
    out.bits |= _bits;
    out.f *= exp_f2<127-FP8<E>::E_BIAS>();
    return out.f;
}
} // namespace fp8

template<int E>
static inline void conv(const float* x, FP8<E>* y, int64_t size) {
#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp simd
#endif
    for (int64_t i=0; i<size; i++) {
        y[i].bits = fp8::from_float<E>(x[i]);
    }
}

template<int E>
static inline float dot(const FP8<E>* x, const float* y, int64_t size) {
    float z = 0;
#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp simd reduction(+:z)
#endif
    for (int64_t i=0; i<size; i++) {
        z += fp8::to_float(x[i])*y[i];
    }
    return z;
}

template <int E, int QK>
struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template <int E, int QK>
static inline void conv(const float* x, bloc_fp8<E, QK>* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float m = 0;
#ifdef GGML_USE_OPENMP_SIMD
        // did not work on macOS and warn.
        // #pragma omp simd reduction(max:m)
#endif
        for (int64_t i=0; i<QK; i++) {
            m = std::max(std::abs(x[q*QK+i]),m);
        }
        const float D = FP8<E>::MAX/m;
        y[q].d = m/FP8<E>::MAX;
#ifdef GGML_USE_OPENMP_SIMD
        #pragma omp simd
#endif
        for (int64_t i=0; i<QK; i++) {
            y[q].qs[i].bits = fp8::from_float<E>(x[q*QK+i]*D);
        }
    }
}

template <int E, int QK>
static inline float dot(const bloc_fp8<E, QK>* x, const float* y, int64_t size) {
    float z = 0;
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float z0 = 0;
#ifdef GGML_USE_OPENMP_SIMD
        #pragma omp simd reduction(+:z0)
#endif
        for (int64_t i=0; i<QK; i++) {
            z0 += fp8::to_float(x[q].qs[i])*y[q*QK+i];
        }
        z += (x[q]).d * z0;
    }
    return z;
}

template <int VECT_SIZE, int NB_REG, int E, int QK, typename _Y>
float dot_reg(const bloc_fp8<E, QK>* x, const _Y* y, int64_t size) {
    static_assert(QK%(VECT_SIZE*NB_REG)==0, "size not supported");
    using fp8_t = FP8<E>;

    float z = 0;
    float Z[NB_REG][VECT_SIZE];
    for(int64_t r=0; r<NB_REG; ++r) {
        for(int64_t v=0; v<VECT_SIZE; ++v) Z[r][v] = 0;
    }
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float Z0[NB_REG][VECT_SIZE];
        for(int64_t r=0; r<NB_REG; ++r) {
            for(int64_t v=0; v<VECT_SIZE; ++v) Z0[r][v] = 0;
        }
        for (int64_t i=0; i<QK; i+=VECT_SIZE*NB_REG) {
            for(int64_t r=0; r<NB_REG; ++r) {
                uint8_t x_8bits[VECT_SIZE];
                uint8_t sign_8bits[VECT_SIZE];
                uint8_t mantice_8bits[VECT_SIZE];
                uint16_t sign_16bits[VECT_SIZE];
                uint16_t mantice_16bits[VECT_SIZE];
                uint16_t x_bf16[VECT_SIZE];
                union { uint32_t bits; float f; } ux[VECT_SIZE];
                float X[VECT_SIZE];
                float Y[VECT_SIZE];
                for(int64_t v=0; v<VECT_SIZE; ++v) { x_8bits[v] = x[q].qs[i+r*VECT_SIZE+v].bits; }
                for(int64_t v=0; v<VECT_SIZE; ++v) { sign_8bits[v] = x_8bits[v] & 0x80; }
                for(int64_t v=0; v<VECT_SIZE; ++v) { mantice_8bits[v] = x_8bits[v] & 0x7F; }

                for(int64_t v=0; v<VECT_SIZE; ++v) { sign_16bits[v] = sign_8bits[v]; }
                for(int64_t v=0; v<VECT_SIZE; ++v) { mantice_16bits[v] = mantice_8bits[v]; }

                for(int64_t v=0; v<VECT_SIZE; ++v) { sign_16bits[v] <<= 8; }
                for(int64_t v=0; v<VECT_SIZE; ++v) { mantice_16bits[v] <<= (7-fp8_t::M); }

                for(int64_t v=0; v<VECT_SIZE; ++v) { x_bf16[v] = sign_16bits[v] | mantice_16bits[v]; }

                for(int64_t v=0; v<VECT_SIZE; ++v) { ux[v].bits = x_bf16[v]; }
                for(int64_t v=0; v<VECT_SIZE; ++v) { ux[v].bits <<= 16; }

                for(int64_t v=0; v<VECT_SIZE; ++v) { X[v] = ux[v].f; } // * exp_f2<127-fp8_t::E_BIAS>(); }
                for(int64_t v=0; v<VECT_SIZE; ++v) { Y[v] = (float)y[q*QK+i+r*VECT_SIZE+v]; }
                for(int64_t v=0; v<VECT_SIZE; ++v) { Z0[r][v] += X[v]*Y[v]; }
            }
        }
        // apply scale
        for(int64_t r=0; r<NB_REG; ++r) {
            for(int64_t v=0; v<VECT_SIZE; ++v) {
                Z[r][v] += Z0[r][v]*(x[q]).d * exp_f2<127-fp8_t::E_BIAS>();
            }
        }
    }
    // reduction 1
    for(int64_t r=1; r<NB_REG; ++r) {
        for(int64_t v=0; v<VECT_SIZE; ++v) {
            Z[0][v] += Z[r][v];
        }
    }
    // reduction 2
    for(int64_t v=0; v<VECT_SIZE; ++v) {
        z += Z[0][v];
    }
    return z;
}

// the C API.
void ggml_fp32_to_e5m2_row(const float * x, ggml_e5m2_t * y, int64_t k) {
    conv(x, reinterpret_cast<FP8<5>*>(y), k);
}

void ggml_fp32_to_e4m3_row(const float * x, ggml_e4m3_t * y, int64_t k) {
    conv(x, reinterpret_cast<FP8<4>*>(y), k);
}

void quantize_row_e4m3_q(const float * x, block_e4m3_q * y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}

void quantize_row_e3m4_q(const float * x, block_e3m4_q * y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}

// the dot product for FP8 weight
void ggml_vec_dot_e5m2(int n, float * s, size_t bs, const ggml_e5m2_t * vx, size_t bx, const float * vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const FP8<5>*>(vx), vy, n);
}

void ggml_vec_dot_e4m3(int n, float * s, size_t bs, const ggml_e4m3_t * vx, size_t bx, const float * vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const FP8<4>*>(vx), vy, n);
}

void ggml_vec_dot_e4m3_q(int n, float * s, size_t bs, const block_e4m3_q * vx, size_t bx, const float * vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
#if defined(__AVX512F__) // 32xfloat32x16_t
    *s = dot_reg<16,4>(reinterpret_cast<const bloc_fp8<4, QK_K>*>(vx), vy, n);
#elif defined(__AVX__) || defined(__AVX2__)  // 16xfloat32x8_t
    *s = dot_reg<8,4>(reinterpret_cast<const bloc_fp8<4, QK_K>*>(vx), vy, n);
#elif defined(__ARM_NEON) // 32xfloat32x4_t
    *s = dot_reg<4,4>(reinterpret_cast<const bloc_fp8<4, QK_K>*>(vx), vy, n);
// #elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)  // 32xfloat16x8_t
#else
    *s = dot(reinterpret_cast<const bloc_fp8<4, QK_K>*>(vx), vy, n);
#endif
}

void ggml_vec_dot_e3m4_q(int n, float * s, size_t bs, const block_e3m4_q * vx, size_t bx, const float * vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
#if defined(__AVX512F__) // 32xfloat32x16_t
    *s = dot_reg<16,4>(reinterpret_cast<const bloc_fp8<3, QK_K>*>(vx), vy, n);
#elif defined(__AVX__) || defined(__AVX2__)  // 16xfloat32x8_t
    *s = dot_reg<8,4>(reinterpret_cast<const bloc_fp8<3, QK_K>*>(vx), vy, n);
#elif defined(__ARM_NEON) // 32xfloat32x4_t
    *s = dot_reg<4,4>(reinterpret_cast<const bloc_fp8<3, QK_K>*>(vx), vy, n);
// #elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)  // 32xfloat16x8_t
#else
    *s = dot(reinterpret_cast<const bloc_fp8<3, QK_K>*>(vx), vy, n);
#endif
}
