// SPDX-FileCopyrightText: Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved
// SPDX-License-Identifier: MIT

#ifndef GGML_OPENCL2_H
#define GGML_OPENCL2_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            GGML_ASSERT(0);                                                \
        }                                                           \
    } while (0)

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl2_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl2(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl2_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl2_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_t ggml_backend_reg_opencl2_init(const char * params, void * user_data);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl2_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL2_H
