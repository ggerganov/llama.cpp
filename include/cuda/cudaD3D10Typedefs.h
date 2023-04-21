/*
 * Copyright 2020-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef CUDAD3D10TYPEDEFS_H
#define CUDAD3D10TYPEDEFS_H

// Dependent includes for cudaD3D10.h
#include <rpcsal.h>
#include <D3D10_1.h>

#include <cudaD3D10.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in cudaD3D10.h
 */
#define PFN_cuD3D10GetDevice  PFN_cuD3D10GetDevice_v2010
#define PFN_cuD3D10GetDevices  PFN_cuD3D10GetDevices_v3020
#define PFN_cuGraphicsD3D10RegisterResource  PFN_cuGraphicsD3D10RegisterResource_v3000
#define PFN_cuD3D10CtxCreate  PFN_cuD3D10CtxCreate_v3020
#define PFN_cuD3D10CtxCreateOnDevice  PFN_cuD3D10CtxCreateOnDevice_v3020
#define PFN_cuD3D10GetDirect3DDevice  PFN_cuD3D10GetDirect3DDevice_v3020
#define PFN_cuD3D10RegisterResource  PFN_cuD3D10RegisterResource_v2010
#define PFN_cuD3D10UnregisterResource  PFN_cuD3D10UnregisterResource_v2010
#define PFN_cuD3D10MapResources  PFN_cuD3D10MapResources_v2010
#define PFN_cuD3D10UnmapResources  PFN_cuD3D10UnmapResources_v2010
#define PFN_cuD3D10ResourceSetMapFlags  PFN_cuD3D10ResourceSetMapFlags_v2010
#define PFN_cuD3D10ResourceGetMappedArray  PFN_cuD3D10ResourceGetMappedArray_v2010
#define PFN_cuD3D10ResourceGetMappedPointer  PFN_cuD3D10ResourceGetMappedPointer_v3020
#define PFN_cuD3D10ResourceGetMappedSize  PFN_cuD3D10ResourceGetMappedSize_v3020
#define PFN_cuD3D10ResourceGetMappedPitch  PFN_cuD3D10ResourceGetMappedPitch_v3020
#define PFN_cuD3D10ResourceGetSurfaceDimensions  PFN_cuD3D10ResourceGetSurfaceDimensions_v3020


/**
 * Type definitions for functions defined in cudaD3D10.h
 */
typedef CUresult (CUDAAPI *PFN_cuD3D10GetDevice_v2010)(CUdevice_v1 *pCudaDevice, IDXGIAdapter *pAdapter);
typedef CUresult (CUDAAPI *PFN_cuD3D10GetDevices_v3020)(unsigned int *pCudaDeviceCount, CUdevice_v1 *pCudaDevices, unsigned int cudaDeviceCount, ID3D10Device *pD3D10Device, CUd3d10DeviceList deviceList);
typedef CUresult (CUDAAPI *PFN_cuGraphicsD3D10RegisterResource_v3000)(CUgraphicsResource *pCudaResource, ID3D10Resource *pD3DResource, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuD3D10CtxCreate_v3020)(CUcontext *pCtx, CUdevice_v1 *pCudaDevice, unsigned int Flags, ID3D10Device *pD3DDevice);
typedef CUresult (CUDAAPI *PFN_cuD3D10CtxCreateOnDevice_v3020)(CUcontext *pCtx, unsigned int flags, ID3D10Device *pD3DDevice, CUdevice_v1 cudaDevice);
typedef CUresult (CUDAAPI *PFN_cuD3D10GetDirect3DDevice_v3020)(ID3D10Device **ppD3DDevice);
typedef CUresult (CUDAAPI *PFN_cuD3D10RegisterResource_v2010)(ID3D10Resource *pResource, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuD3D10UnregisterResource_v2010)(ID3D10Resource *pResource);
typedef CUresult (CUDAAPI *PFN_cuD3D10MapResources_v2010)(unsigned int count, ID3D10Resource **ppResources);
typedef CUresult (CUDAAPI *PFN_cuD3D10UnmapResources_v2010)(unsigned int count, ID3D10Resource **ppResources);
typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceSetMapFlags_v2010)(ID3D10Resource *pResource, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedArray_v2010)(CUarray *pArray, ID3D10Resource *pResource, unsigned int SubResource);
typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedPointer_v3020)(CUdeviceptr_v2 *pDevPtr, ID3D10Resource *pResource, unsigned int SubResource);
typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedSize_v3020)(size_t *pSize, ID3D10Resource *pResource, unsigned int SubResource);
typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedPitch_v3020)(size_t *pPitch, size_t *pPitchSlice, ID3D10Resource *pResource, unsigned int SubResource);
typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetSurfaceDimensions_v3020)(size_t *pWidth, size_t *pHeight, size_t *pDepth, ID3D10Resource *pResource, unsigned int SubResource);

/*
 * Type definitions for older versioned functions in cudaD3D10.h
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
    typedef CUresult (CUDAAPI *PFN_cuD3D10CtxCreate_v2010)(CUcontext *pCtx, CUdevice_v1 *pCudaDevice, unsigned int Flags, ID3D10Device *pD3DDevice);
    typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedPitch_v2010)(unsigned int *pPitch, unsigned int *pPitchSlice, ID3D10Resource *pResource, unsigned int SubResource);
    typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedPointer_v2010)(CUdeviceptr_v1 *pDevPtr, ID3D10Resource *pResource, unsigned int SubResource);
    typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetMappedSize_v2010)(unsigned int *pSize, ID3D10Resource *pResource, unsigned int SubResource);
    typedef CUresult (CUDAAPI *PFN_cuD3D10ResourceGetSurfaceDimensions_v2010)(unsigned int *pWidth, unsigned int *pHeight, unsigned int *pDepth, ID3D10Resource *pResource, unsigned int SubResource);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
