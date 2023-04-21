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

#if !defined(__CUDA_D3D9_INTEROP_H__)
#define __CUDA_D3D9_INTEROP_H__

#include "cuda_runtime_api.h"

/** \cond impl_private */
#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v) \
        = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */
/** \endcond impl_private */

#include <d3d9.h>

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
 * \addtogroup CUDART_D3D9 Direct3D 9 Interoperability
 * This section describes the Direct3D 9 interoperability functions of the CUDA
 * runtime application programming interface. Note that mapping of Direct3D 9
 * resources is performed with the graphics API agnostic, resource mapping 
 * interface described in \ref CUDART_INTEROP "Graphics Interopability".
 *
 * @{
 */

/**
 * CUDA devices corresponding to a D3D9 device
 */
enum cudaD3D9DeviceList
{
  cudaD3D9DeviceListAll           = 1, /**< The CUDA devices for all GPUs used by a D3D9 device */
  cudaD3D9DeviceListCurrentFrame  = 2, /**< The CUDA devices for the GPUs used by a D3D9 device in its currently rendering frame */
  cudaD3D9DeviceListNextFrame     = 3  /**< The CUDA devices for the GPUs to be used by a D3D9 device in the next frame  */
};

/**
 * \brief Gets the Direct3D device against which the current CUDA context was
 * created
 *
 * Returns in \p *ppD3D9Device the Direct3D device against which this CUDA
 * context was created in ::cudaD3D9SetDirect3DDevice().
 *
 * \param ppD3D9Device - Returns the Direct3D device for this thread
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidGraphicsContext,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaD3D9SetDirect3DDevice,
 * ::cuD3D9GetDirect3DDevice
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D9GetDirect3DDevice(IDirect3DDevice9 **ppD3D9Device);

/**
 * \brief Register a Direct3D 9 resource for access by CUDA
 * 
 * Registers the Direct3D 9 resource \p pD3DResource for access by CUDA.  
 *
 * If this call is successful then the application will be able to map and
 * unmap this resource until it is unregistered through
 * ::cudaGraphicsUnregisterResource(). Also on success, this call will increase the
 * internal reference count on \p pD3DResource. This reference count will be
 * decremented when this resource is unregistered through
 * ::cudaGraphicsUnregisterResource().
 *
 * This call potentially has a high-overhead and should not be called every frame
 * in interactive applications.
 *
 * The type of \p pD3DResource must be one of the following.
 *
 * - ::IDirect3DVertexBuffer9: may be accessed through a device pointer
 * - ::IDirect3DIndexBuffer9: may be accessed through a device pointer
 * - ::IDirect3DSurface9: may be accessed through an array.
 *     Only stand-alone objects of type ::IDirect3DSurface9
 *     may be explicitly shared. In particular, individual mipmap levels and faces
 *     of cube maps may not be registered directly. To access individual surfaces
 *     associated with a texture, one must register the base texture object.
 * - ::IDirect3DBaseTexture9: individual surfaces on this texture may be accessed
 *     through an array.
 *
 * The \p flags argument may be used to specify additional parameters at register
 * time.  The valid values for this parameter are 
 *
 * - ::cudaGraphicsRegisterFlagsNone: Specifies no hints about how this
 *   resource will be used.
 * - ::cudaGraphicsRegisterFlagsSurfaceLoadStore: Specifies that CUDA will
 *   bind this resource to a surface reference.
 * - ::cudaGraphicsRegisterFlagsTextureGather: Specifies that CUDA will perform
 *   texture gather operations on this resource.
 *
 * Not all Direct3D resources of the above types may be used for
 * interoperability with CUDA.  The following are some limitations.
 *
 * - The primary rendertarget may not be registered with CUDA.
 * - Resources allocated as shared may not be registered with CUDA.
 * - Textures which are not of a format which is 1, 2, or 4 channels of 8, 16,
 *   or 32-bit integer or floating-point data cannot be shared.
 * - Surfaces of depth or stencil formats cannot be shared.
 *
 * A complete list of supported formats is as follows:
 * - D3DFMT_L8
 * - D3DFMT_L16
 * - D3DFMT_A8R8G8B8
 * - D3DFMT_X8R8G8B8
 * - D3DFMT_G16R16
 * - D3DFMT_A8B8G8R8
 * - D3DFMT_A8
 * - D3DFMT_A8L8
 * - D3DFMT_Q8W8V8U8
 * - D3DFMT_V16U16
 * - D3DFMT_A16B16G16R16F
 * - D3DFMT_A16B16G16R16
 * - D3DFMT_R32F
 * - D3DFMT_G16R16F
 * - D3DFMT_A32B32G32R32F
 * - D3DFMT_G32R32F
 * - D3DFMT_R16F
 *
 * If \p pD3DResource is of incorrect type or is already registered, then 
 * ::cudaErrorInvalidResourceHandle is returned. 
 * If \p pD3DResource cannot be registered, then ::cudaErrorUnknown is returned.
 *
 * \param resource - Pointer to returned resource handle
 * \param pD3DResource - Direct3D resource to register
 * \param flags        - Parameters for resource registration
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
 * ::cudaD3D9SetDirect3DDevice,
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources,
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGraphicsD3D9RegisterResource
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsD3D9RegisterResource(struct cudaGraphicsResource **resource, IDirect3DResource9 *pD3DResource, unsigned int flags);

/**
 * \brief Gets the device number for an adapter
 *
 * Returns in \p *device the CUDA-compatible device corresponding to the
 * adapter name \p pszAdapterName obtained from ::EnumDisplayDevices or
 * ::IDirect3D9::GetAdapterIdentifier(). If no device on the adapter with name
 * \p pszAdapterName is CUDA-compatible then the call will fail.
 *
 * \param device         - Returns the device corresponding to pszAdapterName
 * \param pszAdapterName - D3D9 adapter to get device for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaD3D9SetDirect3DDevice,
 * ::cudaGraphicsD3D9RegisterResource,
 * ::cuD3D9GetDevice
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D9GetDevice(int *device, const char *pszAdapterName);

/**
 * \brief Gets the CUDA devices corresponding to a Direct3D 9 device
 * 
 * Returns in \p *pCudaDeviceCount the number of CUDA-compatible devices corresponding 
 * to the Direct3D 9 device \p pD3D9Device.
 * Also returns in \p *pCudaDevices at most \p cudaDeviceCount of the the CUDA-compatible devices 
 * corresponding to the Direct3D 9 device \p pD3D9Device.
 *
 * If any of the GPUs being used to render \p pDevice are not CUDA capable then the
 * call will return ::cudaErrorNoDevice.
 *
 * \param pCudaDeviceCount - Returned number of CUDA devices corresponding to \p pD3D9Device
 * \param pCudaDevices     - Returned CUDA devices corresponding to \p pD3D9Device
 * \param cudaDeviceCount  - The size of the output device array \p pCudaDevices
 * \param pD3D9Device      - Direct3D 9 device to query for CUDA devices
 * \param deviceList       - The set of devices to return.  This set may be
 *                           ::cudaD3D9DeviceListAll for all devices, 
 *                           ::cudaD3D9DeviceListCurrentFrame for the devices used to
 *                           render the current frame (in SLI), or
 *                           ::cudaD3D9DeviceListNextFrame for the devices used to
 *                           render the next frame (in SLI).
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNoDevice,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources, 
 * ::cudaGraphicsSubResourceGetMappedArray, 
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuD3D9GetDevices 
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D9GetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, IDirect3DDevice9 *pD3D9Device, enum cudaD3D9DeviceList deviceList);

/**
 * \brief Sets the Direct3D 9 device to use for interoperability with 
 * a CUDA device
 *
 * Records \p pD3D9Device as the Direct3D 9 device to use for Direct3D 9
 * interoperability with the CUDA device \p device and sets \p device as 
 * the current device for the calling host thread.
 * 
 * If \p device has already been initialized then this call will fail with 
 * the error ::cudaErrorSetOnActiveProcess.  In this case it is necessary 
 * to reset \p device using ::cudaDeviceReset() before Direct3D 9 
 * interoperability on \p device may be enabled.
 *
 * Successfully initializing CUDA interoperability with \p pD3D9Device 
 * will increase the internal reference count on \p pD3D9Device.  This 
 * reference count will be decremented when \p device is reset using 
 * ::cudaDeviceReset().
 *
 * Note that this function is never required for correct functionality.  Use of 
 * this function will result in accelerated interoperability only when the
 * operating system is Windows Vista or Windows 7, and the device \p pD3DDdevice 
 * is not an IDirect3DDevice9Ex.  In all other cirumstances, this function is 
 * not necessary.
 *
 * \param pD3D9Device - Direct3D device to use for this thread
 * \param device      - The CUDA device to use.  This device must be among the devices
 *                      returned when querying ::cudaD3D9DeviceListAll from ::cudaD3D9GetDevices,
 *                      may be set to -1 to automatically select an appropriate CUDA device.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorSetOnActiveProcess
 * \notefnerr
 *
 * \sa 
 * ::cudaD3D9GetDevice,
 * ::cudaGraphicsD3D9RegisterResource,
 * ::cudaDeviceReset
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D9SetDirect3DDevice(IDirect3DDevice9 *pD3D9Device, int device __dv(-1));

/** @} */ /* END CUDART_D3D9 */

/**
 * \addtogroup CUDART_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED]
 * This section describes deprecated Direct3D 9 interoperability functions.
 *
 * @{
 */

/**
 * CUDA D3D9 Register Flags
 */
enum cudaD3D9RegisterFlags
{
  cudaD3D9RegisterFlagsNone  = 0,  /**< Default; Resource can be accessed througa void* */
  cudaD3D9RegisterFlagsArray = 1   /**< Resource can be accessed through a CUarray* */
};

/**
 * CUDA D3D9 Map Flags
 */
enum cudaD3D9MapFlags
{
  cudaD3D9MapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
  cudaD3D9MapFlagsReadOnly     = 1,  /**< CUDA kernels will not write to this resource */
  cudaD3D9MapFlagsWriteDiscard = 2   /**< CUDA kernels will only write to and will not read from this resource */
};

/**
 * \brief Registers a Direct3D resource for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Registers the Direct3D resource \p pResource for access by CUDA.
 *
 * If this call is successful, then the application will be able to map and
 * unmap this resource until it is unregistered through
 * ::cudaD3D9UnregisterResource(). Also on success, this call will increase
 * the internal reference count on \p pResource. This reference count will be
 * decremented when this resource is unregistered through
 * ::cudaD3D9UnregisterResource().
 *
 * This call potentially has a high-overhead and should not be called every frame
 * in interactive applications.
 *
 * The type of \p pResource must be one of the following.
 *
 * - ::IDirect3DVertexBuffer9: No notes.
 * - ::IDirect3DIndexBuffer9: No notes.
 * - ::IDirect3DSurface9: Only stand-alone objects of type ::IDirect3DSurface9
 * may be explicitly shared. In particular, individual mipmap levels and faces
 * of cube maps may not be registered directly. To access individual surfaces
 * associated with a texture, one must register the base texture object.
 * - ::IDirect3DBaseTexture9: When a texture is registered, all surfaces
 * associated with all mipmap levels of all faces of the texture will be
 * accessible to CUDA.
 *
 * The \p flags argument specifies the mechanism through which CUDA will
 * access the Direct3D resource. The following value is allowed:
 *
 * - ::cudaD3D9RegisterFlagsNone: Specifies that CUDA will access this
 * resource through a \p void*. The pointer, size, and pitch for each
 * subresource of this resource may be queried through
 * ::cudaD3D9ResourceGetMappedPointer(), ::cudaD3D9ResourceGetMappedSize(),
 * and ::cudaD3D9ResourceGetMappedPitch() respectively. This option is valid
 * for all resource types.
 *
 * Not all Direct3D resources of the above types may be used for
 * interoperability with CUDA. The following are some limitations:
 *
 * - The primary rendertarget may not be registered with CUDA.
 * - Resources allocated as shared may not be registered with CUDA.
 * - Any resources allocated in ::D3DPOOL_SYSTEMMEM or ::D3DPOOL_MANAGED may
 *   not be registered with CUDA.
 * - Textures which are not of a format which is 1, 2, or 4 channels of 8, 16,
 *   or 32-bit integer or floating-point data cannot be shared.
 * - Surfaces of depth or stencil formats cannot be shared.
 *
 * If Direct3D interoperability is not initialized on this context, then
 * ::cudaErrorInvalidDevice is returned. If \p pResource is of incorrect type
 * (e.g, is a non-stand-alone ::IDirect3DSurface9) or is already registered,
 * then ::cudaErrorInvalidResourceHandle is returned. If \p pResource cannot
 * be registered then ::cudaErrorUnknown is returned.
 *
 * \param pResource - Resource to register
 * \param flags     - Parameters for resource registration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsD3D9RegisterResource
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D9RegisterResource(IDirect3DResource9 *pResource, unsigned int flags);

/**
 * \brief Unregisters a Direct3D resource for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Unregisters the Direct3D resource \p pResource so it is not accessible by
 * CUDA unless registered again.
 *
 * If \p pResource is not registered, then ::cudaErrorInvalidResourceHandle is
 * returned.
 *
 * \param pResource - Resource to unregister
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsUnregisterResource
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9UnregisterResource(IDirect3DResource9 *pResource);

/**
 * \brief Map Direct3D resources for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Maps the \p count Direct3D resources in \p ppResources for access by CUDA.
 *
 * The resources in \p ppResources may be accessed in CUDA kernels until they
 * are unmapped. Direct3D should not access any resources while they are
 * mapped by CUDA. If an application does so, the results are undefined.
 *
 * This function provides the synchronization guarantee that any Direct3D
 * calls issued before ::cudaD3D9MapResources() will complete before any CUDA
 * kernels issued after ::cudaD3D9MapResources() begin.
 *
 * If any of \p ppResources have not been registered for use with CUDA or if
 * \p ppResources contains any duplicate entries then
 * ::cudaErrorInvalidResourceHandle is returned. If any of \p ppResources are
 * presently mapped for access by CUDA then ::cudaErrorUnknown is returned.
 *
 * \param count       - Number of resources to map for CUDA
 * \param ppResources - Resources to map for CUDA
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsMapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9MapResources(int count, IDirect3DResource9 **ppResources);

/**
 * \brief Unmap Direct3D resources for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Unmaps the \p count Direct3D resources in \p ppResources.  
 *
 * This function provides the synchronization guarantee that any CUDA kernels
 * issued before ::cudaD3D9UnmapResources() will complete before any Direct3D
 * calls issued after ::cudaD3D9UnmapResources() begin.
 *
 * If any of \p ppResources have not been registered for use with CUDA or if
 * \p ppResources contains any duplicate entries, then
 * ::cudaErrorInvalidResourceHandle is returned. If any of \p ppResources are
 * not presently mapped for access by CUDA then ::cudaErrorUnknown is returned.
 *
 * \param count       - Number of resources to unmap for CUDA
 * \param ppResources - Resources to unmap for CUDA
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
  * ::cudaGraphicsUnmapResources
  */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9UnmapResources(int count, IDirect3DResource9 **ppResources);

/**
 * \brief Set usage flags for mapping a Direct3D resource
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Set flags for mapping the Direct3D resource \p pResource.
 *
 * Changes to flags will take effect the next time \p pResource is mapped.
 * The \p flags argument may be any of the following:
 *
 * - ::cudaD3D9MapFlagsNone: Specifies no hints about how this resource will
 * be used. It is therefore assumed that this resource will be read from and
 * written to by CUDA kernels. This is the default value.
 * - ::cudaD3D9MapFlagsReadOnly: Specifies that CUDA kernels which access this
 * resource will not write to this resource.
 * - ::cudaD3D9MapFlagsWriteDiscard: Specifies that CUDA kernels which access
 * this resource will not read from this resource and will write over the
 * entire contents of the resource, so none of the data previously stored in
 * the resource will be preserved.
 *
 * If \p pResource has not been registered for use with CUDA, then
 * ::cudaErrorInvalidResourceHandle is returned. If \p pResource is presently
 * mapped for access by CUDA, then ::cudaErrorUnknown is returned.
 *
 * \param pResource - Registered resource to set flags for
 * \param flags     - Parameters for resource mapping
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa
 * ::cudaInteropResourceSetMapFlags
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9ResourceSetMapFlags(IDirect3DResource9 *pResource, unsigned int flags); 

/**
 * \brief Get the dimensions of a registered Direct3D surface
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pWidth, \p *pHeight, and \p *pDepth the dimensions of the
 * subresource of the mapped Direct3D resource \p pResource which corresponds
 * to \p face and \p level.
 *
 * Since anti-aliased surfaces may have multiple samples per pixel, it is
 * possible that the dimensions of a resource will be an integer factor larger
 * than the dimensions reported by the Direct3D runtime.
 *
 * The parameters \p pWidth, \p pHeight, and \p pDepth are optional. For 2D
 * surfaces, the value returned in \p *pDepth will be 0.
 *
 * If \p pResource is not of type ::IDirect3DBaseTexture9 or
 * ::IDirect3DSurface9 or if \p pResource has not been registered for use with
 * CUDA, then ::cudaErrorInvalidResourceHandle is returned.
 *
 * For usage requirements of \p face and \p level parameters, see
 * ::cudaD3D9ResourceGetMappedPointer.
 *
 * \param pWidth    - Returned width of surface
 * \param pHeight   - Returned height of surface
 * \param pDepth    - Returned depth of surface
 * \param pResource - Registered resource to access
 * \param face      - Face of resource to access
 * \param level     - Level of resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsSubResourceGetMappedArray
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9ResourceGetSurfaceDimensions(size_t *pWidth, size_t *pHeight, size_t *pDepth, IDirect3DResource9 *pResource, unsigned int face, unsigned int level); 

/**
 * \brief Get an array through which to access a subresource of a Direct3D
 * resource which has been mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pArray an array through which the subresource of the mapped
 * Direct3D resource \p pResource, which corresponds to \p face and \p level
 * may be accessed. The value set in \p pArray may change every time that
 * \p pResource is mapped.
 *
 * If \p pResource is not registered then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource was not registered with usage flags
 * ::cudaD3D9RegisterFlagsArray, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource is not mapped, then ::cudaErrorUnknown is
 * returned.
 *
 * For usage requirements of \p face and \p level parameters, see
 * ::cudaD3D9ResourceGetMappedPointer().
 *
 * \param ppArray   - Returned array corresponding to subresource
 * \param pResource - Mapped resource to access
 * \param face      - Face of resource to access
 * \param level     - Level of resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsSubResourceGetMappedArray
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9ResourceGetMappedArray(cudaArray **ppArray, IDirect3DResource9 *pResource, unsigned int face, unsigned int level);

/**
 * \brief Get a pointer through which to access a subresource of a Direct3D
 * resource which has been mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pPointer the base pointer of the subresource of the mapped
 * Direct3D resource \p pResource, which corresponds to \p face and \p level.
 * The value set in \p pPointer may change every time that \p pResource is
 * mapped.
 *
 * If \p pResource is not registered, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource was not registered with usage flags
 * ::cudaD3D9RegisterFlagsNone, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource is not mapped, then ::cudaErrorUnknown is
 * returned.
 *
 * If \p pResource is of type ::IDirect3DCubeTexture9, then \p face must one
 * of the values enumerated by type ::D3DCUBEMAP_FACES. For all other types,
 * \p face must be 0. If \p face is invalid, then ::cudaErrorInvalidValue is
 * returned.
 *
 * If \p pResource is of type ::IDirect3DBaseTexture9, then \p level must
 * correspond to a valid mipmap level. Only mipmap level 0 is supported for
 * now. For all other types \p level must be 0. If \p level is invalid, then
 * ::cudaErrorInvalidValue is returned.
 *
 * \param pPointer  - Returned pointer corresponding to subresource
 * \param pResource - Mapped resource to access
 * \param face      - Face of resource to access
 * \param level     - Level of resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsResourceGetMappedPointer
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9ResourceGetMappedPointer(void **pPointer, IDirect3DResource9 *pResource, unsigned int face, unsigned int level);

/**
 * \brief Get the size of a subresource of a Direct3D resource which has been
 * mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pSize the size of the subresource of the mapped Direct3D
 * resource \p pResource, which corresponds to \p face and \p level. The value
 * set in \p pSize may change every time that \p pResource is mapped.
 *
 * If \p pResource has not been registered for use with CUDA then
 * ::cudaErrorInvalidResourceHandle is returned. If \p pResource was not
 * registered with usage flags ::cudaD3D9RegisterFlagsNone, then
 * ::cudaErrorInvalidResourceHandle is returned. If \p pResource is not mapped
 * for access by CUDA then ::cudaErrorUnknown is returned.
 *
 * For usage requirements of \p face and \p level parameters, see
 * ::cudaD3D9ResourceGetMappedPointer().
 *
 * \param pSize     - Returned size of subresource
 * \param pResource - Mapped resource to access
 * \param face      - Face of resource to access
 * \param level     - Level of resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsResourceGetMappedPointer
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9ResourceGetMappedSize(size_t *pSize, IDirect3DResource9 *pResource, unsigned int face, unsigned int level);

/**
 * \brief Get the pitch of a subresource of a Direct3D resource which has been
 * mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pPitch and \p *pPitchSlice the pitch and Z-slice pitch of
 * the subresource of the mapped Direct3D resource \p pResource, which
 * corresponds to \p face and \p level. The values set in \p pPitch and
 * \p pPitchSlice may change every time that \p pResource is mapped.
 *
 * The pitch and Z-slice pitch values may be used to compute the location of a
 * sample on a surface as follows.
 *
 * For a 2D surface, the byte offset of the sample at position \b x, \b y from
 * the base pointer of the surface is:
 *
 * \b y * \b pitch + (<b>bytes per pixel</b>) * \b x
 *
 * For a 3D surface, the byte offset of the sample at position \b x, \b y,
 * \b z from the base pointer of the surface is:
 *
 * \b z* \b slicePitch + \b y * \b pitch + (<b>bytes per pixel</b>) * \b x
 *
 * Both parameters \p pPitch and \p pPitchSlice are optional and may be set to
 * NULL.
 *
 * If \p pResource is not of type ::IDirect3DBaseTexture9 or one of its
 * sub-types or if \p pResource has not been registered for use with CUDA,
 * then ::cudaErrorInvalidResourceHandle is returned. If \p pResource was not
 * registered with usage flags ::cudaD3D9RegisterFlagsNone, then
 * ::cudaErrorInvalidResourceHandle is returned. If \p pResource is not mapped
 * for access by CUDA then ::cudaErrorUnknown is returned.
 *
 * For usage requirements of \p face and \p level parameters, see
 * ::cudaD3D9ResourceGetMappedPointer().
 *
 * \param pPitch      - Returned pitch of subresource
 * \param pPitchSlice - Returned Z-slice pitch of subresource
 * \param pResource   - Mapped resource to access
 * \param face        - Face of resource to access
 * \param level       - Level of resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsResourceGetMappedPointer
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9ResourceGetMappedPitch(size_t *pPitch, size_t *pPitchSlice, IDirect3DResource9 *pResource, unsigned int face, unsigned int level);

/* D3D9 1.x interop interface */

extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9Begin(IDirect3DDevice9 *pDevice);
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9End(void);
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9RegisterVertexBuffer(IDirect3DVertexBuffer9 *pVB);
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9UnregisterVertexBuffer(IDirect3DVertexBuffer9 *pVB);
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9MapVertexBuffer(void **dptr, IDirect3DVertexBuffer9 *pVB);
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D9UnmapVertexBuffer(IDirect3DVertexBuffer9 *pVB);

/** @} */ /* END CUDART_D3D9_DEPRECATED */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#undef __dv
#undef __CUDA_DEPRECATED

#endif /* __CUDA_D3D9_INTEROP_H__ */
