#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// buffer_type API
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_amx_buffer_type();

GGML_API GGML_CALL bool ggml_backend_is_amx(ggml_backend_t backend);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_amx_init(void);

GGML_API GGML_CALL void ggml_backend_amx_set_n_threads(ggml_backend_t backend_amx, int n_threads);

#ifdef  __cplusplus
}
#endif
