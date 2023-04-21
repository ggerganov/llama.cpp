/*******************************************************************************
 * Copyright (c) 2008-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef __OPENCL_CL_DX9_MEDIA_SHARING_H
#define __OPENCL_CL_DX9_MEDIA_SHARING_H

#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************/
/* cl_khr_dx9_media_sharing                                                   */
#define cl_khr_dx9_media_sharing 1

typedef cl_uint             cl_dx9_media_adapter_type_khr;
typedef cl_uint             cl_dx9_media_adapter_set_khr;
    
#if defined(_WIN32)
#include <d3d9.h>
typedef struct _cl_dx9_surface_info_khr
{
    IDirect3DSurface9 *resource;
    HANDLE shared_handle;
} cl_dx9_surface_info_khr;
#endif


/******************************************************************************/

/* Error Codes */
#define CL_INVALID_DX9_MEDIA_ADAPTER_KHR                -1010
#define CL_INVALID_DX9_MEDIA_SURFACE_KHR                -1011
#define CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR       -1012
#define CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR           -1013

/* cl_media_adapter_type_khr */
#define CL_ADAPTER_D3D9_KHR                              0x2020
#define CL_ADAPTER_D3D9EX_KHR                            0x2021
#define CL_ADAPTER_DXVA_KHR                              0x2022

/* cl_media_adapter_set_khr */
#define CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR   0x2023
#define CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR         0x2024

/* cl_context_info */
#define CL_CONTEXT_ADAPTER_D3D9_KHR                      0x2025
#define CL_CONTEXT_ADAPTER_D3D9EX_KHR                    0x2026
#define CL_CONTEXT_ADAPTER_DXVA_KHR                      0x2027

/* cl_mem_info */
#define CL_MEM_DX9_MEDIA_ADAPTER_TYPE_KHR                0x2028
#define CL_MEM_DX9_MEDIA_SURFACE_INFO_KHR                0x2029

/* cl_image_info */
#define CL_IMAGE_DX9_MEDIA_PLANE_KHR                     0x202A

/* cl_command_type */
#define CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR        0x202B
#define CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR        0x202C

/******************************************************************************/

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetDeviceIDsFromDX9MediaAdapterKHR_fn)(
    cl_platform_id                   platform,
    cl_uint                          num_media_adapters,
    cl_dx9_media_adapter_type_khr *  media_adapter_type,
    void *                           media_adapters,
    cl_dx9_media_adapter_set_khr     media_adapter_set,
    cl_uint                          num_entries,
    cl_device_id *                   devices,
    cl_uint *                        num_devices) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromDX9MediaSurfaceKHR_fn)(
    cl_context                    context,
    cl_mem_flags                  flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void *                        surface_info,
    cl_uint                       plane,                                                                          
    cl_int *                      errcode_ret) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireDX9MediaSurfacesKHR_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseDX9MediaSurfacesKHR_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_2;

#ifdef __cplusplus
}
#endif

#endif  /* __OPENCL_CL_DX9_MEDIA_SHARING_H */

