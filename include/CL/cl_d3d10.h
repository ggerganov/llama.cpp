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

#ifndef __OPENCL_CL_D3D10_H
#define __OPENCL_CL_D3D10_H

#if defined(_MSC_VER)
#if _MSC_VER >=1500
#pragma warning( push )
#pragma warning( disable : 4201 )
#pragma warning( disable : 5105 )
#endif
#endif
#include <d3d10.h>
#if defined(_MSC_VER)
#if _MSC_VER >=1500
#pragma warning( pop )
#endif
#endif
#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * cl_khr_d3d10_sharing                                                       */
#define cl_khr_d3d10_sharing 1

typedef cl_uint cl_d3d10_device_source_khr;
typedef cl_uint cl_d3d10_device_set_khr;

/******************************************************************************/

/* Error Codes */
#define CL_INVALID_D3D10_DEVICE_KHR                  -1002
#define CL_INVALID_D3D10_RESOURCE_KHR                -1003
#define CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR       -1004
#define CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR           -1005

/* cl_d3d10_device_source_nv */
#define CL_D3D10_DEVICE_KHR                          0x4010
#define CL_D3D10_DXGI_ADAPTER_KHR                    0x4011

/* cl_d3d10_device_set_nv */
#define CL_PREFERRED_DEVICES_FOR_D3D10_KHR           0x4012
#define CL_ALL_DEVICES_FOR_D3D10_KHR                 0x4013

/* cl_context_info */
#define CL_CONTEXT_D3D10_DEVICE_KHR                  0x4014
#define CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR 0x402C

/* cl_mem_info */
#define CL_MEM_D3D10_RESOURCE_KHR                    0x4015

/* cl_image_info */
#define CL_IMAGE_D3D10_SUBRESOURCE_KHR               0x4016

/* cl_command_type */
#define CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR         0x4017
#define CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR         0x4018

/******************************************************************************/

typedef cl_int (CL_API_CALL *clGetDeviceIDsFromD3D10KHR_fn)(
    cl_platform_id             platform,
    cl_d3d10_device_source_khr d3d_device_source,
    void *                     d3d_object,
    cl_d3d10_device_set_khr    d3d_device_set,
    cl_uint                    num_entries,
    cl_device_id *             devices,
    cl_uint *                  num_devices) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem (CL_API_CALL *clCreateFromD3D10BufferKHR_fn)(
    cl_context     context,
    cl_mem_flags   flags,
    ID3D10Buffer * resource,
    cl_int *       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem (CL_API_CALL *clCreateFromD3D10Texture2DKHR_fn)(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D10Texture2D * resource,
    UINT              subresource,
    cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_mem (CL_API_CALL *clCreateFromD3D10Texture3DKHR_fn)(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D10Texture3D * resource,
    UINT              subresource,
    cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int (CL_API_CALL *clEnqueueAcquireD3D10ObjectsKHR_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef cl_int (CL_API_CALL *clEnqueueReleaseD3D10ObjectsKHR_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

/***************************************************************
* cl_intel_sharing_format_query_d3d10
***************************************************************/
#define cl_intel_sharing_format_query_d3d10 1

/* when cl_khr_d3d10_sharing is supported */

extern CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedD3D10TextureFormatsINTEL(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    DXGI_FORMAT* d3d10_formats,
    cl_uint* num_texture_formats) ;

typedef cl_int (CL_API_CALL *
clGetSupportedD3D10TextureFormatsINTEL_fn)(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    DXGI_FORMAT* d3d10_formats,
    cl_uint* num_texture_formats) ;

#ifdef __cplusplus
}
#endif

#endif  /* __OPENCL_CL_D3D10_H */

