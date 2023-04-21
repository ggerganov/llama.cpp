/**********************************************************************************
 * Copyright (c) 2008-2009 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 **********************************************************************************/

#ifndef __OPENCL_CL_D3D11_EXT_H
#define __OPENCL_CL_D3D11_EXT_H

#include <d3d11.h>
#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * cl_nv_d3d11_sharing                                                        */

typedef cl_uint cl_d3d11_device_source_nv;
typedef cl_uint cl_d3d11_device_set_nv;

/******************************************************************************/

// Error Codes
#define CL_INVALID_D3D11_DEVICE_NV             -1006
#define CL_INVALID_D3D11_RESOURCE_NV           -1007
#define CL_D3D11_RESOURCE_ALREADY_ACQUIRED_NV  -1008
#define CL_D3D11_RESOURCE_NOT_ACQUIRED_NV      -1009

// cl_d3d11_device_source_nv
#define CL_D3D11_DEVICE_NV                     0x4019
#define CL_D3D11_DXGI_ADAPTER_NV               0x401A

// cl_d3d11_device_set_nv
#define CL_PREFERRED_DEVICES_FOR_D3D11_NV      0x401B
#define CL_ALL_DEVICES_FOR_D3D11_NV            0x401C

// cl_context_info
#define CL_CONTEXT_D3D11_DEVICE_NV             0x401D

// cl_mem_info
#define CL_MEM_D3D11_RESOURCE_NV               0x401E

// cl_image_info
#define CL_IMAGE_D3D11_SUBRESOURCE_NV          0x401F

// cl_command_type
#define CL_COMMAND_ACQUIRE_D3D11_OBJECTS_NV    0x4020
#define CL_COMMAND_RELEASE_D3D11_OBJECTS_NV    0x4021

/******************************************************************************/

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetDeviceIDsFromD3D11NV_fn)(
    cl_platform_id            platform,
    cl_d3d11_device_source_nv d3d_device_source,
    void *                    d3d_object,
    cl_d3d11_device_set_nv    d3d_device_set,
    cl_uint                   num_entries, 
    cl_device_id *            devices, 
    cl_uint *                 num_devices) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D11BufferNV_fn)(
    cl_context     context,
    cl_mem_flags   flags,
    ID3D11Buffer * resource,
    cl_int *       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D11Texture2DNV_fn)(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D11Texture2D * resource,
    UINT              subresource,
    cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D11Texture3DNV_fn)(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D11Texture3D * resource,
    UINT              subresource,
    cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireD3D11ObjectsNV_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseD3D11ObjectsNV_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    cl_mem *         mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

#ifdef __cplusplus
}
#endif

#endif  // __OPENCL_CL_D3D11_H

