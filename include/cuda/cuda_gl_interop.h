/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUDA_GL_INTEROP_H__)
#define __CUDA_GL_INTEROP_H__

#include "cuda_runtime_api.h"

#if defined(__APPLE__)

#include <OpenGL/gl.h>

#else /* __APPLE__ */

#if defined(__arm__) || defined(__aarch64__)
#ifndef GL_VERSION
#error Please include the appropriate gl headers before including cuda_gl_interop.h
#endif
#else
#include <GL/gl.h>
#endif

#endif /* __APPLE__ */

/** \cond impl_private */
#if defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif
/** \endcond impl_private */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \addtogroup CUDART_OPENGL OpenGL Interoperability
 * This section describes the OpenGL interoperability functions of the CUDA
 * runtime application programming interface. Note that mapping of OpenGL
 * resources is performed with the graphics API agnostic, resource mapping 
 * interface described in \ref CUDART_INTEROP "Graphics Interopability".
 *
 * @{
 */

/**
 * CUDA devices corresponding to the current OpenGL context
 */
enum cudaGLDeviceList
{
  cudaGLDeviceListAll           = 1, /**< The CUDA devices for all GPUs used by the current OpenGL context */
  cudaGLDeviceListCurrentFrame  = 2, /**< The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame */
  cudaGLDeviceListNextFrame     = 3  /**< The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame  */
};

/**
 * \brief Gets the CUDA devices associated with the current OpenGL context
 *
 * Returns in \p *pCudaDeviceCount the number of CUDA-compatible devices 
 * corresponding to the current OpenGL context. Also returns in \p *pCudaDevices 
 * at most \p cudaDeviceCount of the CUDA-compatible devices corresponding to 
 * the current OpenGL context. If any of the GPUs being used by the current OpenGL
 * context are not CUDA capable then the call will return ::cudaErrorNoDevice.
 *
 * \param pCudaDeviceCount - Returned number of CUDA devices corresponding to the 
 *                           current OpenGL context
 * \param pCudaDevices     - Returned CUDA devices corresponding to the current 
 *                           OpenGL context
 * \param cudaDeviceCount  - The size of the output device array \p pCudaDevices
 * \param deviceList       - The set of devices to return.  This set may be
 *                           ::cudaGLDeviceListAll for all devices, 
 *                           ::cudaGLDeviceListCurrentFrame for the devices used to
 *                           render the current frame (in SLI), or
 *                           ::cudaGLDeviceListNextFrame for the devices used to
 *                           render the next frame (in SLI).
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNoDevice,
 * ::cudaErrorInvalidGraphicsContext,
 * ::cudaErrorUnknown
 *
 * \note This function is not supported on Mac OS X.
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources, 
 * ::cudaGraphicsSubResourceGetMappedArray, 
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGLGetDevices 
 */
extern __host__ cudaError_t CUDARTAPI cudaGLGetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, enum cudaGLDeviceList deviceList);

/**
 * \brief Register an OpenGL texture or renderbuffer object
 *
 * Registers the texture or renderbuffer object specified by \p image for access by CUDA.
 * A handle to the registered object is returned as \p resource.
 *
 * \p target must match the type of the object, and must be one of ::GL_TEXTURE_2D, 
 * ::GL_TEXTURE_RECTANGLE, ::GL_TEXTURE_CUBE_MAP, ::GL_TEXTURE_3D, ::GL_TEXTURE_2D_ARRAY, 
 * or ::GL_RENDERBUFFER.
 *
 * The register flags \p flags specify the intended usage, as follows: 
 * - ::cudaGraphicsRegisterFlagsNone: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::cudaGraphicsRegisterFlagsReadOnly: Specifies that CUDA
 *   will not write to this resource.
 * - ::cudaGraphicsRegisterFlagsWriteDiscard: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 * - ::cudaGraphicsRegisterFlagsSurfaceLoadStore: Specifies that CUDA will
 *   bind this resource to a surface reference.
 * - ::cudaGraphicsRegisterFlagsTextureGather: Specifies that CUDA will perform
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
 * \param resource - Pointer to the returned object handle
 * \param image    - name of texture or renderbuffer object to be registered
 * \param target   - Identifies the type of object specified by \p image 
 * \param flags    - Register flags
 * 
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources, 
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsGLRegisterImage
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource **resource, GLuint image, GLenum target, unsigned int flags);

/**
 * \brief Registers an OpenGL buffer object
 *
 * Registers the buffer object specified by \p buffer for access by
 * CUDA.  A handle to the registered object is returned as \p
 * resource.  The register flags \p flags specify the intended usage,
 * as follows:
 *
 * - ::cudaGraphicsRegisterFlagsNone: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::cudaGraphicsRegisterFlagsReadOnly: Specifies that CUDA
 *   will not write to this resource.
 * - ::cudaGraphicsRegisterFlagsWriteDiscard: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * \param resource - Pointer to the returned object handle
 * \param buffer   - name of buffer object to be registered
 * \param flags    - Register flags
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources,
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGraphicsGLRegisterBuffer
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(struct cudaGraphicsResource **resource, GLuint buffer, unsigned int flags);

#ifdef _WIN32
#ifndef WGL_NV_gpu_affinity
typedef void* HGPUNV;
#endif

/**
 * \brief Gets the CUDA device associated with hGpu
 *
 * Returns the CUDA device associated with a hGpu, if applicable.
 *
 * \param device - Returns the device associated with hGpu, or -1 if hGpu is
 * not a compute device.
 * \param hGpu   - Handle to a GPU, as queried via WGL_NV_gpu_affinity
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa
 * ::WGL_NV_gpu_affinity,
 * ::cuWGLGetDevice
 */
extern __host__ cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu);
#endif

/** @} */ /* END CUDART_OPENGL */

/**
 * \addtogroup CUDART_OPENGL_DEPRECATED OpenGL Interoperability [DEPRECATED]
 * This section describes deprecated OpenGL interoperability functionality.
 *
 * @{
 */

/**
 * CUDA GL Map Flags
 */
enum cudaGLMapFlags
{
  cudaGLMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
  cudaGLMapFlagsReadOnly     = 1,  /**< CUDA kernels will not write to this resource */
  cudaGLMapFlagsWriteDiscard = 2   /**< CUDA kernels will only write to and will not read from this resource */
};

/**
 * \brief Sets a CUDA device to use OpenGL interoperability
 *
 * \deprecated This function is deprecated as of CUDA 5.0. 
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a CUDA device with an OpenGL
 * context in order to achieve maximum interoperability performance.
 *
 * \param device - Device to use for OpenGL interoperability
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorSetOnActiveProcess
 * \notefnerr
 *
 * \sa ::cudaGraphicsGLRegisterBuffer, ::cudaGraphicsGLRegisterImage
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLSetGLDevice(int device);

/**
 * \brief Registers a buffer object for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Registers the buffer object of ID \p bufObj for access by
 * CUDA. This function must be called before CUDA can map the buffer
 * object.  The OpenGL context used to create the buffer, or another
 * context from the same share group, must be bound to the current
 * thread when this is called.
 *
 * \param bufObj - Buffer object ID to register
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError
 * \notefnerr
 *
 * \sa ::cudaGraphicsGLRegisterBuffer
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLRegisterBufferObject(GLuint bufObj);

/**
 * \brief Maps a buffer object for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Maps the buffer object of ID \p bufObj into the address space of
 * CUDA and returns in \p *devPtr the base pointer of the resulting
 * mapping.  The buffer must have previously been registered by
 * calling ::cudaGLRegisterBufferObject().  While a buffer is mapped
 * by CUDA, any OpenGL operation which references the buffer will
 * result in undefined behavior.  The OpenGL context used to create
 * the buffer, or another context from the same share group, must be
 * bound to the current thread when this is called.
 *
 * All streams in the current thread are synchronized with the current
 * GL context.
 *
 * \param devPtr - Returned device pointer to CUDA object
 * \param bufObj - Buffer object ID to map
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed
 * \notefnerr
 *
 * \sa ::cudaGraphicsMapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLMapBufferObject(void **devPtr, GLuint bufObj);

/**
 * \brief Unmaps a buffer object for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Unmaps the buffer object of ID \p bufObj for access by CUDA.  When
 * a buffer is unmapped, the base address returned by
 * ::cudaGLMapBufferObject() is invalid and subsequent references to
 * the address result in undefined behavior.  The OpenGL context used
 * to create the buffer, or another context from the same share group,
 * must be bound to the current thread when this is called.
 *
 * All streams in the current thread are synchronized with the current
 * GL context.
 *
 * \param bufObj - Buffer object to unmap
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnmapBufferObjectFailed
 * \notefnerr
 *
 * \sa ::cudaGraphicsUnmapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLUnmapBufferObject(GLuint bufObj);

/**
 * \brief Unregisters a buffer object for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Unregisters the buffer object of ID \p bufObj for access by CUDA
 * and releases any CUDA resources associated with the buffer.  Once a
 * buffer is unregistered, it may no longer be mapped by CUDA.  The GL
 * context used to create the buffer, or another context from the
 * same share group, must be bound to the current thread when this is
 * called.
 *
 * \param bufObj - Buffer object to unregister
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 *
 * \sa ::cudaGraphicsUnregisterResource
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLUnregisterBufferObject(GLuint bufObj);

/**
 * \brief Set usage flags for mapping an OpenGL buffer
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Set flags for mapping the OpenGL buffer \p bufObj
 *
 * Changes to flags will take effect the next time \p bufObj is mapped.
 * The \p flags argument may be any of the following:
 *
 * - ::cudaGLMapFlagsNone: Specifies no hints about how this buffer will
 * be used. It is therefore assumed that this buffer will be read from and
 * written to by CUDA kernels. This is the default value.
 * - ::cudaGLMapFlagsReadOnly: Specifies that CUDA kernels which access this
 * buffer will not write to the buffer.
 * - ::cudaGLMapFlagsWriteDiscard: Specifies that CUDA kernels which access
 * this buffer will not read from the buffer and will write over the
 * entire contents of the buffer, so none of the data previously stored in
 * the buffer will be preserved.
 *
 * If \p bufObj has not been registered for use with CUDA, then
 * ::cudaErrorInvalidResourceHandle is returned. If \p bufObj is presently
 * mapped for access by CUDA, then ::cudaErrorUnknown is returned.
 *
 * \param bufObj    - Registered buffer object to set flags for
 * \param flags     - Parameters for buffer mapping
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsResourceSetMapFlags
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLSetBufferObjectMapFlags(GLuint bufObj, unsigned int flags); 

/**
 * \brief Maps a buffer object for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Maps the buffer object of ID \p bufObj into the address space of
 * CUDA and returns in \p *devPtr the base pointer of the resulting
 * mapping.  The buffer must have previously been registered by
 * calling ::cudaGLRegisterBufferObject().  While a buffer is mapped
 * by CUDA, any OpenGL operation which references the buffer will
 * result in undefined behavior.  The OpenGL context used to create
 * the buffer, or another context from the same share group, must be
 * bound to the current thread when this is called.
 *
 * Stream /p stream is synchronized with the current GL context.
 *
 * \param devPtr - Returned device pointer to CUDA object
 * \param bufObj - Buffer object ID to map
 * \param stream - Stream to synchronize
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed
 * \notefnerr
 *
 * \sa ::cudaGraphicsMapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj, cudaStream_t stream);

/**
 * \brief Unmaps a buffer object for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Unmaps the buffer object of ID \p bufObj for access by CUDA.  When
 * a buffer is unmapped, the base address returned by
 * ::cudaGLMapBufferObject() is invalid and subsequent references to
 * the address result in undefined behavior.  The OpenGL context used
 * to create the buffer, or another context from the same share group,
 * must be bound to the current thread when this is called.
 *
 * Stream /p stream is synchronized with the current GL context.
 *
 * \param bufObj - Buffer object to unmap
 * \param stream - Stream to synchronize
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnmapBufferObjectFailed
 * \notefnerr
 *
 * \sa ::cudaGraphicsUnmapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t stream);

/** @} */ /* END CUDART_OPENGL_DEPRECATED */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#undef __CUDA_DEPRECATED

#endif /* __CUDA_GL_INTEROP_H__ */

