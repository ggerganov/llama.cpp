/* SPDX-License-Identifier: MIT
 * Private boundary. Never call these unchecked entry points from GGML.
 * The checked translation unit retains normal GCC aliasing semantics.
 */
#ifndef GGML_KNC_INTERNAL_H
#define GGML_KNC_INTERNAL_H
#include "ggml-knc.h"

#if defined(GGML_KNC_TEST_EMULATION)
# if defined(__MIC__) || defined(__KNC__) || defined(__k1om__)
#  error "Test emulation must never be used for a target build"
# endif
#else
# if !defined(__GNUC__) || !defined(__k1om__) || !defined(__KNC__) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
#  error "Requires the MPSS GCC K1OM cross-compiler with -mknc"
# endif
# if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BW__) || defined(__AVX512VNNI__)
#  error "KNC IMCI and AVX-512 targets must not be mixed"
# endif
#endif

/* Assembly uses scalar/pointer ABI only; lanes points to sixteen values
 * in private 64-byte aligned output storage. No input range is writable.
 * Native code cannot select the test-side implementation of these symbols. */
void ggml_knc_q5_block_native(const uint8_t *q4, const uint8_t *qh,
                               const int8_t *q8, const int32_t *scales,
                               const uint8_t *x, const uint8_t *y,
                               size_t remaining_blocks, int32_t *lanes);
void ggml_knc_f32_vectors_native(size_t n, const float *x, const float *y,
                                  float *lanes);

/* These return values rather than accepting output pointers, so all input
 * arrays are read-only and the sole writable range is exclusive workspace.
 * Input x/y may overlap each other: no store touches either range.
 */
float ggml_knc_dot_packed_unchecked(size_t blocks, const uint8_t *x,
                                  const uint8_t *y, ggml_knc_workspace *workspace);
float ggml_knc_dot_f32_unchecked(size_t n, const float *x, const float *y);
#endif
