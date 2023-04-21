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

#ifndef CUDAGLTYPEDEFS_H
#define CUDAGLTYPEDEFS_H

// Dependent includes for cudagl.h
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <cudaGL.h>

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptds
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptsz
#else
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## default_version
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## default_version
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in cudaGL.h
 */
#define PFN_cuGraphicsGLRegisterBuffer  PFN_cuGraphicsGLRegisterBuffer_v3000
#define PFN_cuGraphicsGLRegisterImage  PFN_cuGraphicsGLRegisterImage_v3000
#define PFN_cuWGLGetDevice  PFN_cuWGLGetDevice_v2020
#define PFN_cuGLGetDevices  PFN_cuGLGetDevices_v6050
#define PFN_cuGLCtxCreate  PFN_cuGLCtxCreate_v3020
#define PFN_cuGLInit  PFN_cuGLInit_v2000
#define PFN_cuGLRegisterBufferObject  PFN_cuGLRegisterBufferObject_v2000
#define PFN_cuGLMapBufferObject  __API_TYPEDEF_PTDS(PFN_cuGLMapBufferObject, 3020, 7000)
#define PFN_cuGLUnmapBufferObject  PFN_cuGLUnmapBufferObject_v2000
#define PFN_cuGLUnregisterBufferObject  PFN_cuGLUnregisterBufferObject_v2000
#define PFN_cuGLSetBufferObjectMapFlags  PFN_cuGLSetBufferObjectMapFlags_v2030
#define PFN_cuGLMapBufferObjectAsync  __API_TYPEDEF_PTSZ(PFN_cuGLMapBufferObjectAsync, 3020, 7000)
#define PFN_cuGLUnmapBufferObjectAsync  PFN_cuGLUnmapBufferObjectAsync_v2030


/**
 * Type definitions for functions defined in cudaGL.h
 */
typedef CUresult (CUDAAPI *PFN_cuGraphicsGLRegisterBuffer_v3000)(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuGraphicsGLRegisterImage_v3000)(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);
#ifdef _WIN32
typedef CUresult (CUDAAPI *PFN_cuWGLGetDevice_v2020)(CUdevice_v1 *pDevice, HGPUNV hGpu);
#endif
typedef CUresult (CUDAAPI *PFN_cuGLGetDevices_v6050)(unsigned int *pCudaDeviceCount, CUdevice_v1 *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
typedef CUresult (CUDAAPI *PFN_cuGLCtxCreate_v3020)(CUcontext *pCtx, unsigned int Flags, CUdevice_v1 device);
typedef CUresult (CUDAAPI *PFN_cuGLInit_v2000)(void);
typedef CUresult (CUDAAPI *PFN_cuGLRegisterBufferObject_v2000)(GLuint buffer);
typedef CUresult (CUDAAPI *PFN_cuGLMapBufferObject_v7000_ptds)(CUdeviceptr_v2 *dptr, size_t *size, GLuint buffer);
typedef CUresult (CUDAAPI *PFN_cuGLUnmapBufferObject_v2000)(GLuint buffer);
typedef CUresult (CUDAAPI *PFN_cuGLUnregisterBufferObject_v2000)(GLuint buffer);
typedef CUresult (CUDAAPI *PFN_cuGLSetBufferObjectMapFlags_v2030)(GLuint buffer, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuGLMapBufferObjectAsync_v7000_ptsz)(CUdeviceptr_v2 *dptr, size_t *size, GLuint buffer, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGLUnmapBufferObjectAsync_v2030)(GLuint buffer, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGLMapBufferObject_v3020)(CUdeviceptr_v2 *dptr, size_t *size, GLuint buffer);
typedef CUresult (CUDAAPI *PFN_cuGLMapBufferObjectAsync_v3020)(CUdeviceptr_v2 *dptr, size_t *size, GLuint buffer, CUstream hStream);

/*
 * Type definitions for older versioned functions in cuda.h
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
typedef CUresult (CUDAAPI *PFN_cuGLGetDevices_v4010)(unsigned int *pCudaDeviceCount, CUdevice_v1 *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
typedef CUresult (CUDAAPI *PFN_cuGLMapBufferObject_v2000)(CUdeviceptr_v1 *dptr, unsigned int *size, GLuint buffer);
typedef CUresult (CUDAAPI *PFN_cuGLMapBufferObjectAsync_v2030)(CUdeviceptr_v1 *dptr, unsigned int *size, GLuint buffer, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGLCtxCreate_v2000)(CUcontext *pCtx, unsigned int Flags, CUdevice_v1 device);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
