#pragma once

#include "ggml.h"

#include <stdint.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
void ggml_vec_dot_q4_sq_fp16(const int n, float * restrict s, void * restrict v, ggml_fp16_t * restrict y);
