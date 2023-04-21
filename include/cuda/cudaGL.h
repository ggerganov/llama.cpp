/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
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

#ifndef CUDAGL_H
#define CUDAGL_H

#include <cuda.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#if defined(__CUDA_API_VERSION_INTERNAL) || defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif

#ifdef CUDA_FORCE_API_VERSION
#error "CUDA_FORCE_API_VERSION is no longer supported."
#endif

#if defined(__CUDA_API_VERSION_INTERNAL) || defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __CUDA_API_PER_THREAD_DEFAULT_STREAM
    #define __CUDA_API_PTDS(api) api ## _ptds
    #define __CUDA_API_PTSZ(api) api ## _ptsz
#else
    #define __CUDA_API_PTDS(api) api
    #define __CUDA_API_PTSZ(api) api
#endif

#define cuGLCtxCreate            cuGLCtxCreate_v2
#define cuGLMapBufferObject      __CUDA_API_PTDS(cuGLMapBufferObject_v2)
#define cuGLMapBufferObjectAsync __CUDA_API_PTSZ(cuGLMapBufferObjectAsync_v2)
#define cuGLGetDevices           cuGLGetDevices_v2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file cudaGL.h
 * \brief Header file for the OpenGL interoperability functions of the
 * low-level CUDA driver application programming interface.
 */

/**
 * \defgroup CUDA_GL OpenGL Interoperability
 * \ingroup CUDA_DRIVER
 *
 * ___MANBRIEF___ OpenGL interoperability functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the OpenGL interoperability functions of the
 * low-level CUDA driver application programming interface. Note that mapping 
 * of OpenGL resources is performed with the graphics API agnostic, resource 
 * mapping interface described in \ref CUDA_GRAPHICS "Graphics Interoperability".
 *
 * @{
 */

#if defined(_WIN32)
#if !defined(WGL_NV_gpu_affinity)
typedef void* HGPUNV;
#endif
#endif /* _WIN32 */

/**
 * \brief Registers an OpenGL buffer object
 *
 * Registers the buffer object specified by \p buffer for access by
 * CUDA.  A handle to the registered object is returned as \p
 * pCudaResource.  The register flags \p Flags specify the intended usage,
 * as follows:
 *
 * - ::CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that CUDA
 *   will not write to this resource.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * \param pCudaResource - Pointer to the returned object handle
 * \param buffer - name of buffer object to be registered
 * \param Flags - Register flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * \notefnerr
 *
 * \sa 
 * ::cuGraphicsUnregisterResource,
 * ::cuGraphicsMapResources,
 * ::cuGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsGLRegisterBuffer
 */
CUresult CUDAAPI cuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);

/**
 * \brief Register an OpenGL texture or renderbuffer object
 *
 * Registers the texture or renderbuffer object specified by \p image for access by CUDA.  
 * A handle to the registered object is returned as \p pCudaResource.  
 *
 * \p target must match the type of the object, and must be one of ::GL_TEXTURE_2D, 
 * ::GL_TEXTURE_RECTANGLE, ::GL_TEXTURE_CUBE_MAP, ::GL_TEXTURE_3D, ::GL_TEXTURE_2D_ARRAY, 
 * or ::GL_RENDERBUFFER.
 *
 * The register flags \p Flags specify the intended usage, as follows:
 *
 * - ::CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that CUDA
 *   will not write to this resource.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will
 *   bind this resource to a surface reference.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform
 *   texture gather operations on this resource.
 *
 * The following image formats are supported. For brevity's sake, the list is abbreviated.
 * For ex., {GL_R, GL_RG} X {8, 16} would expand to the following 4 formats 
 * {GL_R8, GL_R16, GL_RG8, GL_RG16} :
 * - GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY
 * - {GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}
 * - {GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X
 * {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}
 *
 * The following image classes are currently disallowed:
 * - Textures with borders
 * - Multisampled renderbuffers
 *
 * \param pCudaResource - Pointer to the returned object handle
 * \param image - name of texture or renderbuffer object to be registered
 * \param target - Identifies the type of object specified by \p image
 * \param Flags - Register flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * \notefnerr
 *
 * \sa 
 * ::cuGraphicsUnregisterResource,
 * ::cuGraphicsMapResources,
 * ::cuGraphicsSubResourceGetMappedArray,
 * ::cudaGraphicsGLRegisterImage
 */
CUresult CUDAAPI cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);

#ifdef _WIN32
/**
 * \brief Gets the CUDA device associated with hGpu
 *
 * Returns in \p *pDevice the CUDA device associated with a \p hGpu, if
 * applicable.
 *
 * \param pDevice - Device associated with hGpu
 * \param hGpu    - Handle to a GPU, as queried via ::WGL_NV_gpu_affinity()
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuGLMapBufferObject,
 * ::cuGLRegisterBufferObject, ::cuGLUnmapBufferObject,
 * ::cuGLUnregisterBufferObject, ::cuGLUnmapBufferObjectAsync,
 * ::cuGLSetBufferObjectMapFlags,
 * ::cudaWGLGetDevice
 */
CUresult CUDAAPI cuWGLGetDevice(CUdevice *pDevice, HGPUNV hGpu);
#endif /* _WIN32 */

/**
 * CUDA devices corresponding to an OpenGL device
 */
typedef enum CUGLDeviceList_enum {
    CU_GL_DEVICE_LIST_ALL            = 0x01, /**< The CUDA devices for all GPUs used by the current OpenGL context */
    CU_GL_DEVICE_LIST_CURRENT_FRAME  = 0x02, /**< The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame */
    CU_GL_DEVICE_LIST_NEXT_FRAME     = 0x03, /**< The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame */
} CUGLDeviceList;

/**
 * \brief Gets the CUDA devices associated with the current OpenGL context
 *
 * Returns in \p *pCudaDeviceCount the number of CUDA-compatible devices 
 * corresponding to the current OpenGL context. Also returns in \p *pCudaDevices 
 * at most cudaDeviceCount of the CUDA-compatible devices corresponding to 
 * the current OpenGL context. If any of the GPUs being used by the current OpenGL
 * context are not CUDA capable then the call will return CUDA_ERROR_NO_DEVICE.
 *
 * The \p deviceList argument may be any of the following:
 * - ::CU_GL_DEVICE_LIST_ALL: Query all devices used by the current OpenGL context.
 * - ::CU_GL_DEVICE_LIST_CURRENT_FRAME: Query the devices used by the current OpenGL context to
 *   render the current frame (in SLI).
 * - ::CU_GL_DEVICE_LIST_NEXT_FRAME: Query the devices used by the current OpenGL context to
 *   render the next frame (in SLI). Note that this is a prediction, it can't be guaranteed that
 *   this is correct in all cases.
 *
 * \param pCudaDeviceCount - Returned number of CUDA devices.
 * \param pCudaDevices     - Returned CUDA devices.
 * \param cudaDeviceCount  - The size of the output device array pCudaDevices.
 * \param deviceList       - The set of devices to return.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NO_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT
 *
 * \note This function is not supported on Mac OS X.
 * \notefnerr
 *
 * \sa
 * ::cuWGLGetDevice,
 * ::cudaGLGetDevices
 */
CUresult CUDAAPI cuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);

/**
 * \defgroup CUDA_GL_DEPRECATED OpenGL Interoperability [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated OpenGL interoperability functions of the low-level
 * CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated OpenGL interoperability functionality.
 *
 * @{
 */

/** Flags to map or unmap a resource */
typedef enum CUGLmap_flags_enum {
    CU_GL_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02,    
} CUGLmap_flags;

/**
 * \brief Create a CUDA context for interoperability with OpenGL
 *
 * \deprecated This function is deprecated as of Cuda 5.0. 
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a CUDA context with an OpenGL
 * context in order to achieve maximum interoperability performance.
 *
 * \param pCtx   - Returned CUDA context
 * \param Flags  - Options for CUDA context creation
 * \param device - Device on which to create the context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 * \notefnerr
 *
 * \sa ::cuCtxCreate, ::cuGLInit, ::cuGLMapBufferObject,
 * ::cuGLRegisterBufferObject, ::cuGLUnmapBufferObject,
 * ::cuGLUnregisterBufferObject, ::cuGLMapBufferObjectAsync,
 * ::cuGLUnmapBufferObjectAsync, ::cuGLSetBufferObjectMapFlags,
 * ::cuWGLGetDevice
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, CUdevice device );

/**
 * \brief Initializes OpenGL interoperability
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Initializes OpenGL interoperability. This function is deprecated
 * and calling it is no longer required. It may fail if the needed
 * OpenGL driver facilities are not available.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa ::cuGLMapBufferObject,
 * ::cuGLRegisterBufferObject, ::cuGLUnmapBufferObject,
 * ::cuGLUnregisterBufferObject, ::cuGLMapBufferObjectAsync,
 * ::cuGLUnmapBufferObjectAsync, ::cuGLSetBufferObjectMapFlags,
 * ::cuWGLGetDevice
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLInit(void);

/**
 * \brief Registers an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Registers the buffer object specified by \p buffer for access by
 * CUDA. This function must be called before CUDA can map the buffer
 * object.  There must be a valid OpenGL context bound to the current
 * thread when this function is called, and the buffer name is
 * resolved by that context.
 *
 * \param buffer - The name of the buffer object to register.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_ALREADY_MAPPED
 * \notefnerr
 *
 * \sa ::cuGraphicsGLRegisterBuffer
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLRegisterBufferObject(GLuint buffer);

/**
 * \brief Maps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Maps the buffer object specified by \p buffer into the address space of the
 * current CUDA context and returns in \p *dptr and \p *size the base pointer
 * and size of the resulting mapping.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * All streams in the current CUDA context are synchronized with the
 * current GL context.
 *
 * \param dptr   - Returned mapped base pointer
 * \param size   - Returned size of mapping
 * \param buffer - The name of the buffer object to map
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_MAP_FAILED
 * \notefnerr
 *
 * \sa ::cuGraphicsMapResources
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLMapBufferObject(CUdeviceptr *dptr, size_t *size,  GLuint buffer);  

/**
 * \brief Unmaps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Unmaps the buffer object specified by \p buffer for access by CUDA.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * All streams in the current CUDA context are synchronized with the
 * current GL context.
 *
 * \param buffer - Buffer object to unmap
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuGraphicsUnmapResources
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLUnmapBufferObject(GLuint buffer);

/**
 * \brief Unregister an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Unregisters the buffer object specified by \p buffer.  This
 * releases any resources associated with the registered buffer.
 * After this call, the buffer may no longer be mapped for access by
 * CUDA.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * \param buffer - Name of the buffer object to unregister
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuGraphicsUnregisterResource
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLUnregisterBufferObject(GLuint buffer);

/**
 * \brief Set the map flags for an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Sets the map flags for the buffer object specified by \p buffer.
 *
 * Changes to \p Flags will take effect the next time \p buffer is mapped.
 * The \p Flags argument may be any of the following:
 * - ::CU_GL_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA kernels. This is the default value.
 * - ::CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA kernels which
 *   access this resource will not write to this resource.
 * - ::CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA kernels
 *   which access this resource will not read from this resource and will
 *   write over the entire contents of the resource, so none of the data
 *   previously stored in the resource will be preserved.
 *
 * If \p buffer has not been registered for use with CUDA, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If \p buffer is presently
 * mapped for access by CUDA, then ::CUDA_ERROR_ALREADY_MAPPED is returned.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * \param buffer - Buffer object to unmap
 * \param Flags  - Map flags
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * \notefnerr
 *
 * \sa ::cuGraphicsResourceSetMapFlags
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags);

/**
 * \brief Maps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Maps the buffer object specified by \p buffer into the address space of the
 * current CUDA context and returns in \p *dptr and \p *size the base pointer
 * and size of the resulting mapping.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * Stream \p hStream in the current CUDA context is synchronized with
 * the current GL context.
 *
 * \param dptr    - Returned mapped base pointer
 * \param size    - Returned size of mapping
 * \param buffer  - The name of the buffer object to map
 * \param hStream - Stream to synchronize
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_MAP_FAILED
 * \notefnerr
 *
 * \sa ::cuGraphicsMapResources
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLMapBufferObjectAsync(CUdeviceptr *dptr, size_t *size,  GLuint buffer, CUstream hStream);

/**
 * \brief Unmaps an OpenGL buffer object
 *
 * \deprecated This function is deprecated as of Cuda 3.0. 
 *
 * Unmaps the buffer object specified by \p buffer for access by CUDA.
 *
 * There must be a valid OpenGL context bound to the current thread
 * when this function is called.  This must be the same context, or a
 * member of the same shareGroup, as the context that was bound when
 * the buffer was registered.
 *
 * Stream \p hStream in the current CUDA context is synchronized with
 * the current GL context.
 *
 * \param buffer  - Name of the buffer object to unmap
 * \param hStream - Stream to synchronize
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE
 * \notefnerr
 *
 * \sa ::cuGraphicsUnmapResources
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream);

/** @} */ /* END CUDA_GL_DEPRECATED */
/** @} */ /* END CUDA_GL */


#if defined(__CUDA_API_VERSION_INTERNAL)
    #undef cuGLCtxCreate
    #undef cuGLMapBufferObject
    #undef cuGLMapBufferObjectAsync
    #undef cuGLGetDevices

    CUresult CUDAAPI cuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
    CUresult CUDAAPI cuGLMapBufferObject_v2(CUdeviceptr *dptr, size_t *size,  GLuint buffer);
    CUresult CUDAAPI cuGLMapBufferObjectAsync_v2(CUdeviceptr *dptr, size_t *size,  GLuint buffer, CUstream hStream);
    CUresult CUDAAPI cuGLCtxCreate(CUcontext *pCtx, unsigned int Flags, CUdevice device );
    CUresult CUDAAPI cuGLMapBufferObject(CUdeviceptr_v1 *dptr, unsigned int *size,  GLuint buffer);
    CUresult CUDAAPI cuGLMapBufferObjectAsync(CUdeviceptr_v1 *dptr, unsigned int *size,  GLuint buffer, CUstream hStream);
#endif /* __CUDA_API_VERSION_INTERNAL */

#ifdef __cplusplus
};
#endif

#undef __CUDA_DEPRECATED

#endif
