#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>

GGML_CALL float DotProduct_F32(const float * restrict vec1, const float * restrict vec2, uint32_t count);

#ifdef  __cplusplus
}
#endif

