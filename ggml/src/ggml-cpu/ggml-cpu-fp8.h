#include "ggml-fp8.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_fp32_to_e5m2_row(const float * x, ggml_e5m2_t * y, int64_t k);
void ggml_fp32_to_e4m3_row(const float * x, ggml_e4m3_t * y, int64_t k);
void quantize_row_e4m3_q(const float * x, block_e4m3_q * y, int64_t k);
void quantize_row_e3m4_q(const float * x, block_e3m4_q * y, int64_t k);

// TODO: the best depend on the CPU fp32 / bf16 / fp16
#define GGML_FP8_VECT_DOT_TYPE GGML_TYPE_F32
void ggml_vec_dot_e5m2  (int n, float * s, size_t bs, const ggml_e5m2_t *  vx, size_t bx, const float * vy, size_t by, int nrc);
void ggml_vec_dot_e4m3  (int n, float * s, size_t bs, const ggml_e4m3_t *  vx, size_t bx, const float * vy, size_t by, int nrc);
void ggml_vec_dot_e4m3_q(int n, float * s, size_t bs, const block_e4m3_q * vx, size_t bx, const float * vy, size_t by, int nrc);
void ggml_vec_dot_e3m4_q(int n, float * s, size_t bs, const block_e3m4_q * vx, size_t bx, const float * vy, size_t by, int nrc);

#ifdef  __cplusplus
}
#endif
