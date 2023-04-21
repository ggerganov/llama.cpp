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

#if !defined(__CUDA_D3D11_INTEROP_H__)
#define __CUDA_D3D11_INTEROP_H__

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

#include <d3d11.h>

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
 * \addtogroup CUDART_D3D11 Direct3D 11 Interoperability
 * This section describes the Direct3D 11 interoperability functions of the CUDA
 * runtime application programming interface. Note that mapping of Direct3D 11
 * resources is performed with the graphics API agnostic, resource mapping 
 * interface described in \ref CUDART_INTEROP "Graphics Interopability".
 *
 * @{
 */

/**
 * CUDA devices corresponding to a D3D11 device
 */
enum cudaD3D11DeviceList
{
  cudaD3D11DeviceListAll           = 1, /**< The CUDA devices for all GPUs used by a D3D11 device */
  cudaD3D11DeviceListCurrentFrame  = 2, /**< The CUDA devices for the GPUs used by a D3D11 device in its currently rendering frame */
  cudaD3D11DeviceListNextFrame     = 3  /**< The CUDA devices for the GPUs to be used by a D3D11 device in the next frame  */
};

/**
 * \brief Register a Direct3D 11 resource for access by CUDA
 * 
 * Registers the Direct3D 11 resource \p pD3DResource for access by CUDA.  
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
 * - ::ID3D11Buffer: may be accessed via a device pointer
 * - ::ID3D11Texture1D: individual subresources of the texture may be accessed via arrays
 * - ::ID3D11Texture2D: individual subresources of the texture may be accessed via arrays
 * - ::ID3D11Texture3D: individual subresources of the texture may be accessed via arrays
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
 * ::cuGraphicsD3D11RegisterResource 
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsD3D11RegisterResource(struct cudaGraphicsResource **resource, ID3D11Resource *pD3DResource, unsigned int flags);

/**
 * \brief Gets the device number for an adapter
 *
 * Returns in \p *device the CUDA-compatible device corresponding to the
 * adapter \p pAdapter obtained from ::IDXGIFactory::EnumAdapters. This call
 * will succeed only if a device on adapter \p pAdapter is CUDA-compatible.
 *
 * \param device   - Returns the device corresponding to pAdapter
 * \param pAdapter - D3D11 adapter to get device for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsMapResources, 
 * ::cudaGraphicsSubResourceGetMappedArray, 
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuD3D11GetDevice 
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D11GetDevice(int *device, IDXGIAdapter *pAdapter);

/**
 * \brief Gets the CUDA devices corresponding to a Direct3D 11 device
 * 
 * Returns in \p *pCudaDeviceCount the number of CUDA-compatible devices corresponding 
 * to the Direct3D 11 device \p pD3D11Device.
 * Also returns in \p *pCudaDevices at most \p cudaDeviceCount of the the CUDA-compatible devices 
 * corresponding to the Direct3D 11 device \p pD3D11Device.
 *
 * If any of the GPUs being used to render \p pDevice are not CUDA capable then the
 * call will return ::cudaErrorNoDevice.
 *
 * \param pCudaDeviceCount - Returned number of CUDA devices corresponding to \p pD3D11Device
 * \param pCudaDevices     - Returned CUDA devices corresponding to \p pD3D11Device
 * \param cudaDeviceCount  - The size of the output device array \p pCudaDevices
 * \param pD3D11Device     - Direct3D 11 device to query for CUDA devices
 * \param deviceList       - The set of devices to return.  This set may be
 *                           ::cudaD3D11DeviceListAll for all devices, 
 *                           ::cudaD3D11DeviceListCurrentFrame for the devices used to
 *                           render the current frame (in SLI), or
 *                           ::cudaD3D11DeviceListNextFrame for the devices used to
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
 * ::cuD3D11GetDevices 
 */
extern __host__ cudaError_t CUDARTAPI cudaD3D11GetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, ID3D11Device *pD3D11Device, enum cudaD3D11DeviceList deviceList);

/** @} */ /* END CUDART_D3D11 */

/**
 * \addtogroup CUDART_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED]
 * This section describes deprecated Direct3D 11 interoperability functions.
 *
 * @{
 */

/**
 * \brief Gets the Direct3D device against which the current CUDA context was
 * created
 *
 * \deprecated This function is deprecated as of CUDA 5.0. 
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a CUDA device with a D3D11
 * device in order to achieve maximum interoperability performance.
 *
 * \param ppD3D11Device - Returns the Direct3D device for this thread
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnknown
 * \notefnerr
 *
 * \sa 
 * ::cudaD3D11SetDirect3DDevice
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D11GetDirect3DDevice(ID3D11Device **ppD3D11Device);

/**
 * \brief Sets the Direct3D 11 device to use for interoperability with 
 * a CUDA device
 *
 * \deprecated This function is deprecated as of CUDA 5.0. 
 *
 * This function is deprecated and should no longer be used.  It is
 * no longer necessary to associate a CUDA device with a D3D11
 * device in order to achieve maximum interoperability performance.
 *
 * \param pD3D11Device - Direct3D device to use for interoperability
 * \param device       - The CUDA device to use.  This device must be among the devices
 *                       returned when querying ::cudaD3D11DeviceListAll from ::cudaD3D11GetDevices,
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
 * ::cudaD3D11GetDevice,
 * ::cudaGraphicsD3D11RegisterResource,
 * ::cudaDeviceReset
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaD3D11SetDirect3DDevice(ID3D11Device *pD3D11Device, int device __dv(-1));

/** @} */ /* END CUDART_D3D11_DEPRECATED */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#undef __dv
#undef __CUDA_DEPRECATED

#endif /* __CUDA_D3D11_INTEROP_H__ */
