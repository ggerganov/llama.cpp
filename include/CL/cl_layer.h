/*
 * Copyright (c) 2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

#ifndef OPENCL_CL_LAYER_H
#define OPENCL_CL_LAYER_H

#include <CL/cl_icd.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef cl_uint cl_layer_info;
typedef cl_uint cl_layer_api_version;
#define CL_LAYER_API_VERSION 0x4240
#define CL_LAYER_NAME        0x4241
#define CL_LAYER_API_VERSION_100 100

extern CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(cl_layer_info  param_name,
               size_t         param_value_size,
               void          *param_value,
               size_t        *param_value_size_ret);

typedef cl_int
(CL_API_CALL *pfn_clGetLayerInfo)(cl_layer_info  param_name,
                                  size_t         param_value_size,
                                  void          *param_value,
                                  size_t        *param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(cl_uint                 num_entries,
            const cl_icd_dispatch  *target_dispatch,
            cl_uint                *num_entries_ret,
            const cl_icd_dispatch **layer_dispatch_ret);

typedef cl_int
(CL_API_CALL *pfn_clInitLayer)(cl_uint                 num_entries,
                               const cl_icd_dispatch  *target_dispatch,
                               cl_uint                *num_entries_ret,
                               const cl_icd_dispatch **layer_dispatch_ret);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_CL_LAYER_H */
