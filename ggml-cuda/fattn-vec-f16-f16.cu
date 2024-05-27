#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec-f16.cuh"

DECL_FATTN_VEC_F16_INST(64, 1, 4, (vec_dot_fattn_vec_KQ_f16<half, 64>), false, dequantize_1_f16<half>);
DECL_FATTN_VEC_F16_INST(128, 1, 4, (vec_dot_fattn_vec_KQ_f16<half, 128>), false, dequantize_1_f16<half>);
DECL_FATTN_VEC_F16_INST(256, 1, 4, (vec_dot_fattn_vec_KQ_f16<half, 256>), false, dequantize_1_f16<half>);
