#pragma once
#include "ggml.h"
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(int64_t, int64_t, int64_t, const void *, int64_t,
                     const void *, int64_t, void *, int64_t, int, int,
                     int, int, int);

bool llamafile_mixmul(const struct ggml_compute_params *, const struct ggml_tensor *,
                      const struct ggml_tensor *, const struct ggml_tensor *, struct ggml_tensor *);

size_t llamafile_mixmul_needs(const struct ggml_tensor *,
                              const struct ggml_tensor *,
                              const struct ggml_tensor *);

#ifdef __cplusplus
}
#endif
