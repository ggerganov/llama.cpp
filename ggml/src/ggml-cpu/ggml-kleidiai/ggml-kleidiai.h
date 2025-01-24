// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-cpu-traits.h"
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(int n_threads);

#ifdef  __cplusplus
}
#endif
