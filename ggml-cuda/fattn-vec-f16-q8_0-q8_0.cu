#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec-f16.cuh"

DECL_FATTN_VEC_INST(f16, 128, 1, 4, q8_0, q8_0);
