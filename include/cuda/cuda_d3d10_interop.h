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

#if !defined(__CUDA_D3D10_INTEROP_H__)
#define __CUDA_D3D10_INTEROP_H__

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

#include <d3d10_1.h>

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
 * \addtogroup CUDART_D3D10 Direct3D 10 Interoperability
 * This section describes the Direct3D 10 interoperability functions of the CUDA
 * runtime application programming interface. Note that mapping of Direct3D 10
 * resources is performed with the graphics API agnostic, resource mapping 
 * interface described in \ref CUDART_INTEROP "Graphics Interopability".
 *
 * @{
 */

/**
 * CUDA devices corresponding to a D3D10 device
 */
enum cudaD3D10DeviceList
{
  cudaD3D10DeviceListAll           = 1, /**< The CUDA devices for all GPUs used by a D3D10 device */
  cudaD3D10DeviceListCurrentFrame  = 2, /**< The CUDA devices for the GPUs used by a D3D10 device in its currently rendering frame */
  cudaD3D10DeviceListNextFrame     = 3  /**< The CUDA devices for the GPUs to be used by a D3D10 device in the next frame  */
};

/**
 * \brief Registers a Direct3D 10 resource for access by CUDA
 * 
 * Registers the Direct3D 10 resource \p pD3DResource for access by CUDA.  
 *
 * If this call is successful, then the application will be able to map and
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
 * - ::ID3D10Buffer: may be accessed via a device pointer
 * - ::ID3D10Texture1D: individual subresources of the texture may be accessed via arrays
 * - ::ID3D10Texture2D: individual subresources of the texture may be accessed via arrays
 * - ::ID3D10Texture3D: individual subresources of the texture may be accessed via arrays
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
 * - Textures which are not of a format which is 1, 2, or 4 channels of 8, 16,
 *   or 32-bit integer or floating-point data cannot be shared.
 * - Surfaces of depth or stencil formats cannot be shared.
 *
 * A complete list of supported DXGI formats is as follows. For compactness the
 * notation A_{B,C,D} represents A_B, A_C, and A_D.
 * - DXGI_FORMAT_A8_UNORM
 * - DXGI_FORMAT_B8G8R8A8_UNORM
 * - DXGI_FORMAT_B8G8R8X8_UNORM
 * - DXGI_FORMAT_R16_FLOAT
 * - DXGI_FORMAT_R16G16B16A16_{FLOAT,SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R16G16_{FLOAT,SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R16_{SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R32_FLOAT
 * - DXGI_FORMAT_R32G32B32A32_{FLOAT,SINT,UINT}
 * - DXGI_FORMAT_R32G32_{FLOAT,SINT,UINT}
 * - DXGI_FORMAT_R32_{SINT,UINT}
 * - DXGI_FORMAT_R8G8B8A8_{SINT,SNORM,UINT,UNORM,UNORM_SRGB}
 * - DXGI_FORMAT_R8G8_{SINT,SNORM,UINT,UNORM}
 * - DXGI_FORMAT_R8_{SINT,SNORM,UINT,UNORM}
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
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources, 
 * ::cudaGraphicsSubResourceGetMappedArray, 
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGraphicsD3D10RegisterResource 
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsD3D10RegisterResource(struct cudaGraphicsResource **resource, ID3D10Resource *pD3DResource, unsigned int flags);

/**
 * \brief Gets the device number for an adapter
 *
 * Returns in \p *device the CUDA-compatible device corresponding to the
 * adapter \p pAdapter obtained from ::IDXGIFactory::EnumAdapters. This call
 * will succeed only if a device on adapter \p pAdapter is CUDA-compatible.
 *
 * \param device   - Returns the device corresponding to pAdapter
 * \param pAdapter - D3D10 adapter to get device for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsD3D10RegisterResource,
 * ::cuD3D10GetDevice 
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D10GetDevice(int *device, IDXGIAdapter *pAdapter);

/**
 * \brief Gets the CUDA devices corresponding to a Direct3D 10 device
 * 
 * Returns in \p *pCudaDeviceCount the number of CUDA-compatible devices corresponding 
 * to the Direct3D 10 device \p pD3D10Device.
 * Also returns in \p *pCudaDevices at most \p cudaDeviceCount of the the CUDA-compatible devices 
 * corresponding to the Direct3D 10 device \p pD3D10Device.
 *
 * If any of the GPUs being used to render \p pDevice are not CUDA capable then the
 * call will return ::cudaErrorNoDevice.
 *
 * \param pCudaDeviceCount - Returned number of CUDA devices corresponding to \p pD3D10Device
 * \param pCudaDevices     - Returned CUDA devices corresponding to \p pD3D10Device
 * \param cudaDeviceCount  - The size of the output device array \p pCudaDevices
 * \param pD3D10Device     - Direct3D 10 device to query for CUDA devices
 * \param deviceList       - The set of devices to return.  This set may be
 *                           ::cudaD3D10DeviceListAll for all devices, 
 *                           ::cudaD3D10DeviceListCurrentFrame for the devices used to
 *                           render the current frame (in SLI), or
 *                           ::cudaD3D10DeviceListNextFrame for the devices used to
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
 * ::cuD3D10GetDevices 
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D10GetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, ID3D10Device *pD3D10Device, enum cudaD3D10DeviceList deviceList);

/** @} */ /* END CUDART_D3D10 */

/**
 * \addtogroup CUDART_D3D10_DEPRECATED Direct3D 10 Interoperability [DEPRECATED]
 * This section describes deprecated Direct3D 10 interoperability functions.
 *
 * @{
 */

/**
 * CUDA D3D10 Register Flags
 */
enum cudaD3D10RegisterFlags
{
  cudaD3D10RegisterFlagsNone  = 0,  /**< Default; Resource can be accessed through a void* */
  cudaD3D10RegisterFlagsArray = 1   /**< Resource can be accessed through a CUarray* */
};

/**
 * CUDA D3D10 Map Flags
 */
enum cudaD3D10MapFlags
{
  cudaD3D10MapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
  cudaD3D10MapFlagsReadOnly     = 1,  /**< CUDA kernels will not write to this resource */
  cudaD3D10MapFlagsWriteDiscard = 2   /**< CUDA kernels will only write to and will not read from this resource */
};

/**
 * \brief Gets the Direct3D device against which the current CUDA context was
 * created
 *
 * \deprecated This function is deprecated as of CUDA 5.0.
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a CUDA device with a D3D10
 * device in order to achieve maximum interoperability performance.
 *
 * \param ppD3D10Device - Returns the Direct3D device for this thread
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaD3D10SetDirect3DDevice
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10GetDirect3DDevice(ID3D10Device **ppD3D10Device);

/**
 * \brief Sets the Direct3D 10 device to use for interoperability with 
 * a CUDA device
 *
 * \deprecated This function is deprecated as of CUDA 5.0.
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a CUDA device with a D3D10
 * device in order to achieve maximum interoperability performance.
 *
 * \param pD3D10Device - Direct3D device to use for interoperability
 * \param device       - The CUDA device to use.  This device must be among the devices
 *                       returned when querying ::cudaD3D10DeviceListAll from ::cudaD3D10GetDevices,
 *                       may be set to -1 to automatically select an appropriate CUDA device.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorSetOnActiveProcess
 * \notefnerr
 *
 * \sa 
 * ::cudaD3D10GetDevice,
 * ::cudaGraphicsD3D10RegisterResource,
 * ::cudaDeviceReset
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10SetDirect3DDevice(ID3D10Device *pD3D10Device, int device __dv(-1));

/**
 * \brief Registers a Direct3D 10 resource for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Registers the Direct3D resource \p pResource for access by CUDA.
 *
 * If this call is successful, then the application will be able to map and
 * unmap this resource until it is unregistered through
 * ::cudaD3D10UnregisterResource(). Also on success, this call will increase
 * the internal reference count on \p pResource. This reference count will be
 * decremented when this resource is unregistered through
 * ::cudaD3D10UnregisterResource().
 *
 * This call potentially has a high-overhead and should not be called every frame
 * in interactive applications.
 *
 * The type of \p pResource must be one of the following:
 *
 * - ::ID3D10Buffer: Cannot be used with \p flags set to
 * \p cudaD3D10RegisterFlagsArray.
 * - ::ID3D10Texture1D: No restrictions.
 * - ::ID3D10Texture2D: No restrictions.
 * - ::ID3D10Texture3D: No restrictions.
 *
 * The \p flags argument specifies the mechanism through which CUDA will
 * access the Direct3D resource. The following values are allowed.
 *
 * - ::cudaD3D10RegisterFlagsNone: Specifies that CUDA will access this
 * resource through a \p void*. The pointer, size, and pitch for each
 * subresource of this resource may be queried through
 * ::cudaD3D10ResourceGetMappedPointer(), ::cudaD3D10ResourceGetMappedSize(),
 * and ::cudaD3D10ResourceGetMappedPitch() respectively. This option is valid
 * for all resource types.
 * - ::cudaD3D10RegisterFlagsArray: Specifies that CUDA will access this
 * resource through a \p CUarray queried on a sub-resource basis through
 * ::cudaD3D10ResourceGetMappedArray(). This option is only valid for resources
 * of type ::ID3D10Texture1D, ::ID3D10Texture2D, and ::ID3D10Texture3D.
 *
 * Not all Direct3D resources of the above types may be used for
 * interoperability with CUDA. The following are some limitations.
 *
 * - The primary rendertarget may not be registered with CUDA.
 * - Resources allocated as shared may not be registered with CUDA.
 * - Textures which are not of a format which is 1, 2, or 4 channels of 8, 16,
 *   or 32-bit integer or floating-point data cannot be shared.
 * - Surfaces of depth or stencil formats cannot be shared.
 *
 * If Direct3D interoperability is not initialized on this context then
 * ::cudaErrorInvalidDevice is returned. If \p pResource is of incorrect type
 * or is already registered then ::cudaErrorInvalidResourceHandle is returned.
 * If \p pResource cannot be registered then ::cudaErrorUnknown is returned.
 *
 * \param pResource - Resource to register
 * \param flags     - Parameters for resource registration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsD3D10RegisterResource
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10RegisterResource(ID3D10Resource *pResource, unsigned int flags);

/**
 * \brief Unregisters a Direct3D resource
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Unregisters the Direct3D resource \p resource so it is not accessible by
 * CUDA unless registered again.
 *
 * If \p pResource is not registered, then ::cudaErrorInvalidResourceHandle
 * is returned.
 *
 * \param pResource - Resource to unregister
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsUnregisterResource
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10UnregisterResource(ID3D10Resource *pResource);

/**
 * \brief Maps Direct3D Resources for access by CUDA
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
 * calls issued before ::cudaD3D10MapResources() will complete before any CUDA
 * kernels issued after ::cudaD3D10MapResources() begin.
 *
 * If any of \p ppResources have not been registered for use with CUDA or if
 * \p ppResources contains any duplicate entries then ::cudaErrorInvalidResourceHandle
 * is returned. If any of \p ppResources are presently mapped for access by
 * CUDA then ::cudaErrorUnknown is returned.
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
 * \sa ::cudaGraphicsMapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10MapResources(int count, ID3D10Resource **ppResources);

/**
 * \brief Unmaps Direct3D resources
 *
 * \deprecated This function is deprecated as of CUDA 3.0.   
 *
 * Unmaps the \p count Direct3D resource in \p ppResources.
 *
 * This function provides the synchronization guarantee that any CUDA kernels
 * issued before ::cudaD3D10UnmapResources() will complete before any Direct3D
 * calls issued after ::cudaD3D10UnmapResources() begin.
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
 * \sa ::cudaGraphicsUnmapResources
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10UnmapResources(int count, ID3D10Resource **ppResources);

/**
 * \brief Gets an array through which to access a subresource of a Direct3D
 * resource which has been mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Returns in \p *ppArray an array through which the subresource of the mapped
 * Direct3D resource \p pResource which corresponds to \p subResource may be
 * accessed. The value set in \p ppArray may change every time that
 * \p pResource is mapped.
 *
 * If \p pResource is not registered, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource was not registered with usage flags
 * ::cudaD3D10RegisterFlagsArray, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource is not mapped then ::cudaErrorUnknown is returned.
 *
 * For usage requirements of the \p subResource parameter, see
 * ::cudaD3D10ResourceGetMappedPointer().
 *
 * \param ppArray     - Returned array corresponding to subresource
 * \param pResource   - Mapped resource to access
 * \param subResource - Subresource of pResource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsSubResourceGetMappedArray
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10ResourceGetMappedArray(cudaArray **ppArray, ID3D10Resource *pResource, unsigned int subResource);

/**
 * \brief Set usage flags for mapping a Direct3D resource
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Set usage flags for mapping the Direct3D resource \p pResource.  
 *
 * Changes to flags will take effect the next time \p pResource is mapped.
 * The \p flags argument may be any of the following:
 *
 * - ::cudaD3D10MapFlagsNone: Specifies no hints about how this resource will
 * be used. It is therefore assumed that this resource will be read from and
 * written to by CUDA kernels. This is the default value.
 * - ::cudaD3D10MapFlagsReadOnly: Specifies that CUDA kernels which access
 * this resource will not write to this resource.
 * - ::cudaD3D10MapFlagsWriteDiscard: Specifies that CUDA kernels which access
 * this resource will not read from this resource and will write over the
 * entire contents of the resource, so none of the data previously stored in
 * the resource will be preserved.
 *
 * If \p pResource has not been registered for use with CUDA then
 * ::cudaErrorInvalidHandle is returned. If \p pResource is presently mapped
 * for access by CUDA then ::cudaErrorUnknown is returned.
 *
 * \param pResource - Registered resource to set flags for
 * \param flags     - Parameters for resource mapping
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown,
 * \notefnerr
 *
 * \sa ::cudaGraphicsResourceSetMapFlags
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10ResourceSetMapFlags(ID3D10Resource *pResource, unsigned int flags); 

/**
 * \brief Gets the dimensions of a registered Direct3D surface
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Returns in \p *pWidth, \p *pHeight, and \p *pDepth the dimensions of the
 * subresource of the mapped Direct3D resource \p pResource which corresponds
 * to \p subResource.
 *
 * Since anti-aliased surfaces may have multiple samples per pixel, it is
 * possible that the dimensions of a resource will be an integer factor larger
 * than the dimensions reported by the Direct3D runtime.
 *
 * The parameters \p pWidth, \p pHeight, and \p pDepth are optional. For 2D
 * surfaces, the value returned in \p *pDepth will be 0.
 *
 * If \p pResource is not of type ::ID3D10Texture1D, ::ID3D10Texture2D, or
 * ::ID3D10Texture3D, or if \p pResource has not been registered for use with
 * CUDA, then ::cudaErrorInvalidHandle is returned.

 * For usage requirements of \p subResource parameters see
 * ::cudaD3D10ResourceGetMappedPointer().
 *
 * \param pWidth      - Returned width of surface
 * \param pHeight     - Returned height of surface
 * \param pDepth      - Returned depth of surface
 * \param pResource   - Registered resource to access
 * \param subResource - Subresource of pResource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * \notefnerr
 *
 * \sa ::cudaGraphicsSubResourceGetMappedArray
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10ResourceGetSurfaceDimensions(size_t *pWidth, size_t *pHeight, size_t *pDepth, ID3D10Resource *pResource, unsigned int subResource); 

/**
 * \brief Gets a pointer through which to access a subresource of a Direct3D
 * resource which has been mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Returns in \p *pPointer the base pointer of the subresource of the mapped
 * Direct3D resource \p pResource which corresponds to \p subResource. The
 * value set in \p pPointer may change every time that \p pResource is mapped.
 *
 * If \p pResource is not registered, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource was not registered with usage flags
 * ::cudaD3D9RegisterFlagsNone, then ::cudaErrorInvalidResourceHandle is
 * returned. If \p pResource is not mapped then ::cudaErrorUnknown is returned.
 *
 * If \p pResource is of type ::ID3D10Buffer then \p subResource must be 0.
 * If \p pResource is of any other type, then the value of \p subResource must
 * come from the subresource calculation in ::D3D10CalcSubResource().
 *
 * \param pPointer    - Returned pointer corresponding to subresource
 * \param pResource   - Mapped resource to access
 * \param subResource - Subresource of pResource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsResourceGetMappedPointer
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10ResourceGetMappedPointer(void **pPointer, ID3D10Resource *pResource, unsigned int subResource);

/**
 * \brief Gets the size of a subresource of a Direct3D resource which has been
 * mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Returns in \p *pSize the size of the subresource of the mapped Direct3D
 * resource \p pResource which corresponds to \p subResource. The value set in
 * \p pSize may change every time that \p pResource is mapped.
 *
 * If \p pResource has not been registered for use with CUDA then
 * ::cudaErrorInvalidHandle is returned. If \p pResource was not registered
 * with usage flags ::cudaD3D10RegisterFlagsNone, then
 * ::cudaErrorInvalidResourceHandle is returned. If \p pResource is not mapped for
 * access by CUDA then ::cudaErrorUnknown is returned.
 *
 * For usage requirements of the \p subResource parameter see
 * ::cudaD3D10ResourceGetMappedPointer().
 *
 * \param pSize       - Returned size of subresource
 * \param pResource   - Mapped resource to access
 * \param subResource - Subresource of pResource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsResourceGetMappedPointer
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10ResourceGetMappedSize(size_t *pSize, ID3D10Resource *pResource, unsigned int subResource);

/**
 * \brief Gets the pitch of a subresource of a Direct3D resource which has been
 * mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0. 
 *
 * Returns in \p *pPitch and \p *pPitchSlice the pitch and Z-slice pitch of
 * the subresource of the mapped Direct3D resource \p pResource, which
 * corresponds to \p subResource. The values set in \p pPitch and
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
 * If \p pResource is not of type ::ID3D10Texture1D, ::ID3D10Texture2D, or
 * ::ID3D10Texture3D, or if \p pResource has not been registered for use with
 * CUDA, then ::cudaErrorInvalidResourceHandle is returned. If \p pResource was
 * not registered with usage flags ::cudaD3D10RegisterFlagsNone, then
 * ::cudaErrorInvalidResourceHandle is returned. If \p pResource is not mapped
 * for access by CUDA then ::cudaErrorUnknown is returned.
 *
 * For usage requirements of the \p subResource parameter see
 * ::cudaD3D10ResourceGetMappedPointer().
 *
 * \param pPitch      - Returned pitch of subresource
 * \param pPitchSlice - Returned Z-slice pitch of subresource
 * \param pResource   - Mapped resource to access
 * \param subResource - Subresource of pResource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa ::cudaGraphicsSubResourceGetMappedArray
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D10ResourceGetMappedPitch(size_t *pPitch, size_t *pPitchSlice, ID3D10Resource *pResource, unsigned int subResource);

/** @} */ /* END CUDART_D3D10_DEPRECATED */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#undef __dv
#undef __CUDA_DEPRECATED

#endif /* __CUDA_D3D10_INTEROP_H__ */
