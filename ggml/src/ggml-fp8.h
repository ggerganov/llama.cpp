// this is more a .inc.
#ifdef  __cplusplus
template<int N>
constexpr int exp_i2() {
    return 1 << N;
}

template<int N>
constexpr float exp_f2() {
    if constexpr (N>0) return exp_f2<N-1>()*2;
    if constexpr (N<0) return exp_f2<N+1>()/2;
    if constexpr (N==0) return 1.;
}


template<int _E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<_E>;
    static constexpr int E      = _E;
    static constexpr int M      = (7-_E);
    static constexpr int E_BIAS = exp_i2<E-1>()-1;
    static constexpr float MAX  = (2-exp_f2<-M+1>())*exp_f2<exp_i2<E-1>()>();
    static constexpr float MIN  = exp_f2<-M>()*exp_f2<2-exp_i2<E-1>()>();
};

extern "C" {
#endif

    // Note: types are define in ggml-common.h
    GGML_API void ggml_e5m2_to_fp32_row(const ggml_e5m2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void ggml_fp32_to_e5m2_row_ref(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k);

    GGML_API void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void ggml_fp32_to_e4m3_row_ref(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k);

    GGML_API void dequantize_row_e4m3_q(const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k);

    GGML_API void dequantize_row_e3m4_q(const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k);

#ifdef  __cplusplus
}
#endif
