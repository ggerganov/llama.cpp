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

#ifndef CUDAD3D9_H
#define CUDAD3D9_H

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

#define cuD3D9CtxCreate                    cuD3D9CtxCreate_v2
#define cuD3D9ResourceGetSurfaceDimensions cuD3D9ResourceGetSurfaceDimensions_v2
#define cuD3D9ResourceGetMappedPointer     cuD3D9ResourceGetMappedPointer_v2
#define cuD3D9ResourceGetMappedSize        cuD3D9ResourceGetMappedSize_v2
#define cuD3D9ResourceGetMappedPitch       cuD3D9ResourceGetMappedPitch_v2
#define cuD3D9MapVertexBuffer              cuD3D9MapVertexBuffer_v2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \file cudaD3D9.h
 * \brief Header file for the Direct3D 9 interoperability functions of the
 * low-level CUDA driver application programming interface.
 */

/**
 * \defgroup CUDA_D3D9 Direct3D 9 Interoperability
 * \ingroup CUDA_DRIVER
 *
 * ___MANBRIEF___ Direct3D 9 interoperability functions of the low-level CUDA
 * driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the Direct3D 9 interoperability functions of the
 * low-level CUDA driver application programming interface. Note that mapping 
 * of Direct3D 9 resources is performed with the graphics API agnostic, resource 
 * mapping interface described in \ref CUDA_GRAPHICS "Graphics Interoperability".
 *
 * @{
 */

/**
 * CUDA devices corresponding to a D3D9 device
 */
typedef enum CUd3d9DeviceList_enum {
    CU_D3D9_DEVICE_LIST_ALL            = 0x01, /**< The CUDA devices for all GPUs used by a D3D9 device */
    CU_D3D9_DEVICE_LIST_CURRENT_FRAME  = 0x02, /**< The CUDA devices for the GPUs used by a D3D9 device in its currently rendering frame */
    CU_D3D9_DEVICE_LIST_NEXT_FRAME     = 0x03, /**< The CUDA devices for the GPUs to be used by a D3D9 device in the next frame */
} CUd3d9DeviceList;

/**
 * \brief Gets the CUDA device corresponding to a display adapter.
 *
 * Returns in \p *pCudaDevice the CUDA-compatible device corresponding to the
 * adapter name \p pszAdapterName obtained from ::EnumDisplayDevices() or
 * ::IDirect3D9::GetAdapterIdentifier().
 *
 * If no device on the adapter with name \p pszAdapterName is CUDA-compatible,
 * then the call will fail.
 *
 * \param pCudaDevice    - Returned CUDA device corresponding to pszAdapterName
 * \param pszAdapterName - Adapter name to query for device
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuD3D9CtxCreate,
 * ::cudaD3D9GetDevice
 */
CUresult CUDAAPI cuD3D9GetDevice(CUdevice *pCudaDevice, const char *pszAdapterName);

/**
 * \brief Gets the CUDA devices corresponding to a Direct3D 9 device
 *
 * Returns in \p *pCudaDeviceCount the number of CUDA-compatible device corresponding
 * to the Direct3D 9 device \p pD3D9Device.
 * Also returns in \p *pCudaDevices at most \p cudaDeviceCount of the CUDA-compatible devices
 * corresponding to the Direct3D 9 device \p pD3D9Device.
 *
 * If any of the GPUs being used to render \p pDevice are not CUDA capable then the
 * call will return ::CUDA_ERROR_NO_DEVICE.
 *
 * \param pCudaDeviceCount - Returned number of CUDA devices corresponding to \p pD3D9Device
 * \param pCudaDevices     - Returned CUDA devices corresponding to \p pD3D9Device
 * \param cudaDeviceCount  - The size of the output device array \p pCudaDevices
 * \param pD3D9Device      - Direct3D 9 device to query for CUDA devices
 * \param deviceList       - The set of devices to return.  This set may be
 *                           ::CU_D3D9_DEVICE_LIST_ALL for all devices,
 *                           ::CU_D3D9_DEVICE_LIST_CURRENT_FRAME for the devices used to
 *                           render the current frame (in SLI), or
 *                           ::CU_D3D9_DEVICE_LIST_NEXT_FRAME for the devices used to
 *                           render the next frame (in SLI).
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_NO_DEVICE,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_FOUND,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuD3D9CtxCreate,
 * ::cudaD3D9GetDevices
 */
CUresult CUDAAPI cuD3D9GetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, IDirect3DDevice9 *pD3D9Device, CUd3d9DeviceList deviceList);

/**
 * \brief Create a CUDA context for interoperability with Direct3D 9
 *
 * Creates a new CUDA context, enables interoperability for that context with
 * the Direct3D device \p pD3DDevice, and associates the created CUDA context
 * with the calling thread.
 * The created ::CUcontext will be returned in \p *pCtx.
 * Direct3D resources from this device may be registered and mapped through the
 * lifetime of this CUDA context.
 * If \p pCudaDevice is non-NULL then the ::CUdevice on which this CUDA context was
 * created will be returned in \p *pCudaDevice.
 *
 * On success, this call will increase the internal reference count on
 * \p pD3DDevice. This reference count will be decremented upon destruction of
 * this context through ::cuCtxDestroy().
 * This context will cease to function if \p pD3DDevice is destroyed or encounters
 * an error.
 *
 * Note that this function is never required for correct functionality.  Use of 
 * this function will result in accelerated interoperability only when the
 * operating system is Windows Vista or Windows 7, and the device \p pD3DDdevice 
 * is not an IDirect3DDevice9Ex.  In all other circumstances, this function is 
 * not necessary.
 *
 * \param pCtx        - Returned newly created CUDA context
 * \param pCudaDevice - Returned pointer to the device on which the context was created
 * \param Flags       - Context creation flags (see ::cuCtxCreate() for details)
 * \param pD3DDevice  - Direct3D device to create interoperability context with
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuD3D9GetDevice,
 * ::cuGraphicsD3D9RegisterResource
 */
CUresult CUDAAPI cuD3D9CtxCreate(CUcontext *pCtx, CUdevice *pCudaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice);

/**
 * \brief Create a CUDA context for interoperability with Direct3D 9
 *
 * Creates a new CUDA context, enables interoperability for that context with
 * the Direct3D device \p pD3DDevice, and associates the created CUDA context
 * with the calling thread.
 * The created ::CUcontext will be returned in \p *pCtx.
 * Direct3D resources from this device may be registered and mapped through the
 * lifetime of this CUDA context.
 *
 * On success, this call will increase the internal reference count on
 * \p pD3DDevice. This reference count will be decremented upon destruction of
 * this context through ::cuCtxDestroy().
 * This context will cease to function if \p pD3DDevice is destroyed or encounters
 * an error.
 *
 * Note that this function is never required for correct functionality.  Use of 
 * this function will result in accelerated interoperability only when the
 * operating system is Windows Vista or Windows 7, and the device \p pD3DDdevice 
 * is not an IDirect3DDevice9Ex.  In all other circumstances, this function is 
 * not necessary.
 *
 * \param pCtx        - Returned newly created CUDA context
 * \param flags       - Context creation flags (see ::cuCtxCreate() for details)
 * \param pD3DDevice  - Direct3D device to create interoperability context with
 * \param cudaDevice  - The CUDA device on which to create the context.  This device
 *                      must be among the devices returned when querying
 *                      ::CU_D3D9_DEVICES_ALL from  ::cuD3D9GetDevices.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuD3D9GetDevices,
 * ::cuGraphicsD3D9RegisterResource
 */
CUresult CUDAAPI cuD3D9CtxCreateOnDevice(CUcontext *pCtx, unsigned int flags, IDirect3DDevice9 *pD3DDevice, CUdevice cudaDevice);

/**
 * \brief Get the Direct3D 9 device against which the current CUDA context was
 * created
 *
 * Returns in \p *ppD3DDevice the Direct3D device against which this CUDA context
 * was created in ::cuD3D9CtxCreate().
 *
 * \param ppD3DDevice - Returned Direct3D device corresponding to CUDA context
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * ::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::cuD3D9GetDevice,
 * ::cudaD3D9GetDirect3DDevice
 */
CUresult CUDAAPI cuD3D9GetDirect3DDevice(IDirect3DDevice9 **ppD3DDevice);

/**
 * \brief Register a Direct3D 9 resource for access by CUDA
 *
 * Registers the Direct3D 9 resource \p pD3DResource for access by CUDA and
 * returns a CUDA handle to \p pD3Dresource in \p pCudaResource.
 * The handle returned in \p pCudaResource may be used to map and unmap this
 * resource until it is unregistered.
 * On success this call will increase the internal reference count on
 * \p pD3DResource. This reference count will be decremented when this
 * resource is unregistered through ::cuGraphicsUnregisterResource().
 *
 * This call is potentially high-overhead and should not be called every frame
 * in interactive applications.
 *
 * The type of \p pD3DResource must be one of the following.
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
 * The \p Flags argument may be used to specify additional parameters at register
 * time.  The valid values for this parameter are
 * - ::CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will
 *   bind this resource to a surface reference.
 * - ::CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform
 *   texture gather operations on this resource.
 *
 * Not all Direct3D resources of the above types may be used for
 * interoperability with CUDA.  The following are some limitations.
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
 * If Direct3D interoperability is not initialized for this context using
 * ::cuD3D9CtxCreate then ::CUDA_ERROR_INVALID_CONTEXT is returned.
 * If \p pD3DResource is of incorrect type or is already registered then
 * ::CUDA_ERROR_INVALID_HANDLE is returned.
 * If \p pD3DResource cannot be registered then ::CUDA_ERROR_UNKNOWN is returned.
 * If \p Flags is not one of the above specified value then ::CUDA_ERROR_INVALID_VALUE
 * is returned.
 *
 * \param pCudaResource - Returned graphics resource handle
 * \param pD3DResource  - Direct3D resource to register
 * \param Flags         - Parameters for resource registration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuD3D9CtxCreate,
 * ::cuGraphicsUnregisterResource,
 * ::cuGraphicsMapResources,
 * ::cuGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsD3D9RegisterResource
 */
CUresult CUDAAPI cuGraphicsD3D9RegisterResource(CUgraphicsResource *pCudaResource, IDirect3DResource9 *pD3DResource, unsigned int Flags);

/**
 * \defgroup CUDA_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated Direct3D 9 interoperability functions of the
 * low-level CUDA driver API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated Direct3D 9 interoperability functionality.
 * @{
 */

/** Flags to register a resource */
typedef enum CUd3d9register_flags_enum {
    CU_D3D9_REGISTER_FLAGS_NONE  = 0x00,
    CU_D3D9_REGISTER_FLAGS_ARRAY = 0x01,
} CUd3d9register_flags;

/** Flags to map or unmap a resource */
typedef enum CUd3d9map_flags_enum {
    CU_D3D9_MAPRESOURCE_FLAGS_NONE         = 0x00,
    CU_D3D9_MAPRESOURCE_FLAGS_READONLY     = 0x01,
    CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02,
} CUd3d9map_flags;

/**
 * \brief Register a Direct3D resource for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Registers the Direct3D resource \p pResource for access by CUDA.
 *
 * If this call is successful, then the application will be able to map and
 * unmap this resource until it is unregistered through
 * ::cuD3D9UnregisterResource(). Also on success, this call will increase the
 * internal reference count on \p pResource. This reference count will be
 * decremented when this resource is unregistered through
 * ::cuD3D9UnregisterResource().
 *
 * This call is potentially high-overhead and should not be called every frame
 * in interactive applications.
 *
 * The type of \p pResource must be one of the following.
 *
 * - ::IDirect3DVertexBuffer9: Cannot be used with \p Flags set to
 *   ::CU_D3D9_REGISTER_FLAGS_ARRAY.
 * - ::IDirect3DIndexBuffer9: Cannot be used with \p Flags set to
 *   ::CU_D3D9_REGISTER_FLAGS_ARRAY.
 * - ::IDirect3DSurface9: Only stand-alone objects of type ::IDirect3DSurface9
 *   may be explicitly shared. In particular, individual mipmap levels and
 *   faces of cube maps may not be registered directly. To access individual
 *   surfaces associated with a texture, one must register the base texture
 *   object. For restrictions on the \p Flags parameter, see type
 *   ::IDirect3DBaseTexture9.
 * - ::IDirect3DBaseTexture9: When a texture is registered, all surfaces
 *   associated with the all mipmap levels of all faces of the texture will be
 *   accessible to CUDA.
 *
 * The \p Flags argument specifies the mechanism through which CUDA will access
 * the Direct3D resource. The following values are allowed.
 *
 * - CU_D3D9_REGISTER_FLAGS_NONE: Specifies that CUDA will access this resource
 *   through a ::CUdeviceptr. The pointer, size, and (for textures), pitch for
 *   each subresource of this allocation may be queried through
 *   ::cuD3D9ResourceGetMappedPointer(), ::cuD3D9ResourceGetMappedSize(), and
 *   ::cuD3D9ResourceGetMappedPitch() respectively. This option is valid for
 *   all resource types.
 * - ::CU_D3D9_REGISTER_FLAGS_ARRAY: Specifies that CUDA will access this
 *   resource through a ::CUarray queried on a sub-resource basis through
 *   ::cuD3D9ResourceGetMappedArray(). This option is only valid for resources
 *   of type ::IDirect3DSurface9 and subtypes of ::IDirect3DBaseTexture9.
 *
 * Not all Direct3D resources of the above types may be used for
 * interoperability with CUDA. The following are some limitations.
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
 * ::CUDA_ERROR_INVALID_CONTEXT is returned. If \p pResource is of incorrect
 * type (e.g. is a non-stand-alone ::IDirect3DSurface9) or is already
 * registered, then ::CUDA_ERROR_INVALID_HANDLE is returned. If \p pResource
 * cannot be registered then ::CUDA_ERROR_UNKNOWN is returned.
 *
 * \param pResource - Resource to register for CUDA access
 * \param Flags     - Flags for resource registration
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_OUT_OF_MEMORY,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsD3D9RegisterResource
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9RegisterResource(IDirect3DResource9 *pResource, unsigned int Flags);

/**
 * \brief Unregister a Direct3D resource
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Unregisters the Direct3D resource \p pResource so it is not accessible by
 * CUDA unless registered again.
 *
 * If \p pResource is not registered, then ::CUDA_ERROR_INVALID_HANDLE is
 * returned.
 *
 * \param pResource - Resource to unregister
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsUnregisterResource
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9UnregisterResource(IDirect3DResource9 *pResource);

/**
 * \brief Map Direct3D resources for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Maps the \p count Direct3D resources in \p ppResource for access by CUDA.
 *
 * The resources in \p ppResource may be accessed in CUDA kernels until they
 * are unmapped. Direct3D should not access any resources while they are mapped
 * by CUDA. If an application does so the results are undefined.
 *
 * This function provides the synchronization guarantee that any Direct3D calls
 * issued before ::cuD3D9MapResources() will complete before any CUDA kernels
 * issued after ::cuD3D9MapResources() begin.
 *
 * If any of \p ppResource have not been registered for use with CUDA or if
 * \p ppResource contains any duplicate entries, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned.  If any of \p ppResource are
 * presently mapped for access by CUDA, then ::CUDA_ERROR_ALREADY_MAPPED is
 * returned.
 *
 * \param count      - Number of resources in ppResource
 * \param ppResource - Resources to map for CUDA usage
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsMapResources
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9MapResources(unsigned int count, IDirect3DResource9 **ppResource);

/**
 * \brief Unmaps Direct3D resources
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Unmaps the \p count Direct3D resources in \p ppResource.
 *
 * This function provides the synchronization guarantee that any CUDA kernels
 * issued before ::cuD3D9UnmapResources() will complete before any Direct3D
 * calls issued after ::cuD3D9UnmapResources() begin.
 *
 * If any of \p ppResource have not been registered for use with CUDA or if
 * \p ppResource contains any duplicate entries, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If any of \p ppResource are not
 * presently mapped for access by CUDA, then ::CUDA_ERROR_NOT_MAPPED is
 * returned.
 *
 * \param count      - Number of resources to unmap for CUDA
 * \param ppResource - Resources to unmap for CUDA
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED,
 * ::CUDA_ERROR_UNKNOWN
 * \notefnerr
 *
 * \sa
 * ::cuGraphicsUnmapResources
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9UnmapResources(unsigned int count, IDirect3DResource9 **ppResource);

/**
 * \brief Set usage flags for mapping a Direct3D resource
 *
 * \deprecated This function is deprecated as of Cuda 3.0.
 *
 * Set \p Flags for mapping the Direct3D resource \p pResource.
 *
 * Changes to \p Flags will take effect the next time \p pResource is mapped.
 * The \p Flags argument may be any of the following:
 * - ::CU_D3D9_MAPRESOURCE_FLAGS_NONE: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA kernels. This is the default value.
 * - ::CU_D3D9_MAPRESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which
 *   access this resource will not write to this resource.
 * - ::CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels
 *   which access this resource will not read from this resource and will
 *   write over the entire contents of the resource, so none of the data
 *   previously stored in the resource will be preserved.
 *
 * If \p pResource has not been registered for use with CUDA, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If \p pResource is presently
 * mapped for access by CUDA, then ::CUDA_ERROR_ALREADY_MAPPED is returned.
 *
 * \param pResource - Registered resource to set flags for
 * \param Flags     - Parameters for resource mapping
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_ALREADY_MAPPED
 * \notefnerr
 *
 * \sa ::cuGraphicsResourceSetMapFlags
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9ResourceSetMapFlags(IDirect3DResource9 *pResource, unsigned int Flags);

/**
 * \brief Get the dimensions of a registered surface
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pWidth, \p *pHeight, and \p *pDepth the dimensions of the
 * subresource of the mapped Direct3D resource \p pResource, which corresponds
 * to \p Face and \p Level.
 *
 * Because anti-aliased surfaces may have multiple samples per pixel, it is
 * possible that the dimensions of a resource will be an integer factor larger
 * than the dimensions reported by the Direct3D runtime.
 *
 * The parameters \p pWidth, \p pHeight, and \p pDepth are optional. For 2D
 * surfaces, the value returned in \p *pDepth will be 0.
 *
 * If \p pResource is not of type ::IDirect3DBaseTexture9 or
 * ::IDirect3DSurface9 or if \p pResource has not been registered for use with
 * CUDA, then ::CUDA_ERROR_INVALID_HANDLE is returned.
 *
 * For usage requirements of \p Face and \p Level parameters, see
 * ::cuD3D9ResourceGetMappedPointer().
 *
 * \param pWidth    - Returned width of surface
 * \param pHeight   - Returned height of surface
 * \param pDepth    - Returned depth of surface
 * \param pResource - Registered resource to access
 * \param Face      - Face of resource to access
 * \param Level     - Level of resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE
 * \notefnerr
 *
 * \sa ::cuGraphicsSubResourceGetMappedArray
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9ResourceGetSurfaceDimensions(size_t *pWidth, size_t *pHeight, size_t *pDepth, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);

/**
 * \brief Get an array through which to access a subresource of a Direct3D
 * resource which has been mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pArray an array through which the subresource of the mapped
 * Direct3D resource \p pResource which corresponds to \p Face and \p Level may
 * be accessed. The value set in \p pArray may change every time that
 * \p pResource is mapped.
 *
 * If \p pResource is not registered then ::CUDA_ERROR_INVALID_HANDLE is
 * returned. If \p pResource was not registered with usage flags
 * ::CU_D3D9_REGISTER_FLAGS_ARRAY then ::CUDA_ERROR_INVALID_HANDLE is
 * returned. If \p pResource is not mapped then ::CUDA_ERROR_NOT_MAPPED is
 * returned.
 *
 * For usage requirements of \p Face and \p Level parameters, see
 * ::cuD3D9ResourceGetMappedPointer().
 *
 * \param pArray    - Returned array corresponding to subresource
 * \param pResource - Mapped resource to access
 * \param Face      - Face of resource to access
 * \param Level     - Level of resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED
 * \notefnerr
 *
 * \sa ::cuGraphicsSubResourceGetMappedArray
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9ResourceGetMappedArray(CUarray *pArray, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);

/**
 * \brief Get the pointer through which to access a subresource of a Direct3D
 * resource which has been mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pDevPtr the base pointer of the subresource of the mapped
 * Direct3D resource \p pResource, which corresponds to \p Face and \p Level.
 * The value set in \p pDevPtr may change every time that \p pResource is
 * mapped.
 *
 * If \p pResource is not registered, then ::CUDA_ERROR_INVALID_HANDLE is
 * returned. If \p pResource was not registered with usage flags
 * ::CU_D3D9_REGISTER_FLAGS_NONE, then ::CUDA_ERROR_INVALID_HANDLE is returned.
 * If \p pResource is not mapped, then ::CUDA_ERROR_NOT_MAPPED is returned.
 *
 * If \p pResource is of type ::IDirect3DCubeTexture9, then \p Face must one
 * of the values enumerated by type ::D3DCUBEMAP_FACES.  For all other types
 * \p Face must be 0. If \p Face is invalid, then ::CUDA_ERROR_INVALID_VALUE
 * is returned.
 *
 * If \p pResource is of type ::IDirect3DBaseTexture9, then \p Level must
 * correspond to a valid mipmap level. At present only mipmap level 0 is
 * supported. For all other types \p Level must be 0. If \p Level is invalid,
 * then ::CUDA_ERROR_INVALID_VALUE is returned.
 *
 * \param pDevPtr     - Returned pointer corresponding to subresource
 * \param pResource   - Mapped resource to access
 * \param Face        - Face of resource to access
 * \param Level       - Level of resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED
 * \notefnerr
 *
 * \sa ::cuGraphicsResourceGetMappedPointer
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9ResourceGetMappedPointer(CUdeviceptr *pDevPtr, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);

/**
 * \brief Get the size of a subresource of a Direct3D resource which has been
 * mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pSize the size of the subresource of the mapped Direct3D
 * resource \p pResource, which corresponds to \p Face and \p Level. The value
 * set in \p pSize may change every time that \p pResource is mapped.
 *
 * If \p pResource has not been registered for use with CUDA, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If \p pResource was not registered
 * with usage flags ::CU_D3D9_REGISTER_FLAGS_NONE, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If \p pResource is not mapped for
 * access by CUDA, then ::CUDA_ERROR_NOT_MAPPED is returned.
 *
 * For usage requirements of \p Face and \p Level parameters, see
 * ::cuD3D9ResourceGetMappedPointer.
 *
 * \param pSize       - Returned size of subresource
 * \param pResource   - Mapped resource to access
 * \param Face        - Face of resource to access
 * \param Level       - Level of resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED
 * \notefnerr
 *
 * \sa ::cuGraphicsResourceGetMappedPointer
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9ResourceGetMappedSize(size_t *pSize, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);

/**
 * \brief Get the pitch of a subresource of a Direct3D resource which has been
 * mapped for access by CUDA
 *
 * \deprecated This function is deprecated as of CUDA 3.0.
 *
 * Returns in \p *pPitch and \p *pPitchSlice the pitch and Z-slice pitch of
 * the subresource of the mapped Direct3D resource \p pResource, which
 * corresponds to \p Face and \p Level. The values set in \p pPitch and
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
 * registered with usage flags ::CU_D3D9_REGISTER_FLAGS_NONE, then
 * ::CUDA_ERROR_INVALID_HANDLE is returned. If \p pResource is not mapped
 * for access by CUDA then ::CUDA_ERROR_NOT_MAPPED is returned.
 *
 * For usage requirements of \p Face and \p Level parameters, see
 * ::cuD3D9ResourceGetMappedPointer().
 *
 * \param pPitch      - Returned pitch of subresource
 * \param pPitchSlice - Returned Z-slice pitch of subresource
 * \param pResource   - Mapped resource to access
 * \param Face        - Face of resource to access
 * \param Level       - Level of resource to access
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_INVALID_HANDLE,
 * ::CUDA_ERROR_NOT_MAPPED
 * \notefnerr
 *
 * \sa ::cuGraphicsSubResourceGetMappedArray
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9ResourceGetMappedPitch(size_t *pPitch, size_t *pPitchSlice, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);

/* CUDA 1.x compatibility API. These functions are deprecated, please use the ones above. */
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9Begin(IDirect3DDevice9 *pDevice);
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9End(void);
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9RegisterVertexBuffer(IDirect3DVertexBuffer9 *pVB);
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9MapVertexBuffer(CUdeviceptr *pDevPtr, size_t *pSize, IDirect3DVertexBuffer9 *pVB);
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9UnmapVertexBuffer(IDirect3DVertexBuffer9 *pVB);
__CUDA_DEPRECATED CUresult CUDAAPI cuD3D9UnregisterVertexBuffer(IDirect3DVertexBuffer9 *pVB);

/** @} */ /* END CUDA_D3D9_DEPRECATED */
/** @} */ /* END CUDA_D3D9 */


/**
 * CUDA API versioning support
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
    #undef cuD3D9CtxCreate
    #undef cuD3D9ResourceGetSurfaceDimensions
    #undef cuD3D9ResourceGetMappedPointer
    #undef cuD3D9ResourceGetMappedSize
    #undef cuD3D9ResourceGetMappedPitch
    #undef cuD3D9MapVertexBuffer

    CUresult CUDAAPI cuD3D9CtxCreate(CUcontext *pCtx, CUdevice *pCudaDevice, unsigned int Flags, IDirect3DDevice9 *pD3DDevice);
    CUresult CUDAAPI cuD3D9ResourceGetSurfaceDimensions(unsigned int *pWidth, unsigned int *pHeight, unsigned int *pDepth, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    CUresult CUDAAPI cuD3D9ResourceGetMappedPointer(CUdeviceptr_v1 *pDevPtr, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    CUresult CUDAAPI cuD3D9ResourceGetMappedSize(unsigned int *pSize, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    CUresult CUDAAPI cuD3D9ResourceGetMappedPitch(unsigned int *pPitch, unsigned int *pPitchSlice, IDirect3DResource9 *pResource, unsigned int Face, unsigned int Level);
    CUresult CUDAAPI cuD3D9MapVertexBuffer(CUdeviceptr_v1 *pDevPtr, unsigned int *pSize, IDirect3DVertexBuffer9 *pVB);
#endif /* __CUDA_API_VERSION_INTERNAL */

#ifdef __cplusplus
};
#endif

#undef __CUDA_DEPRECATED

#endif

