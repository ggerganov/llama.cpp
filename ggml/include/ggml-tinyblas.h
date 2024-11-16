#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// backend register
GGML_API ggml_backend_reg_t ggml_backend_tinyblas_reg(void);


#ifdef  __cplusplus
}
#endif
