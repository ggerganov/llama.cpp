/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__DRIVER_TYPES_H__)
#define __DRIVER_TYPES_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#ifndef __DOXYGEN_ONLY__
#include "crt/host_defines.h"
#endif
#include "vector_types.h"



/**
 * \defgroup CUDART_TYPES Data types used by CUDA Runtime
 * \ingroup CUDART
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDA_INTERNAL_COMPILATION__)

#if !defined(__CUDACC_RTC__)
#include <limits.h>
#include <stddef.h>
#endif /* !defined(__CUDACC_RTC__) */

#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
#define cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */

#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define cudaStreamDefault                   0x00  /**< Default stream flag */
#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

 /**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamLegacy                    ((cudaStream_t)0x1)

/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamPerThread                 ((cudaStream_t)0x2)

#define cudaEventDefault                    0x00  /**< Default event flag */
#define cudaEventBlockingSync               0x01  /**< Event uses blocking synchronization */
#define cudaEventDisableTiming              0x02  /**< Event will not record timing data */
#define cudaEventInterprocess               0x04  /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */

#define cudaEventRecordDefault              0x00  /**< Default event record flag */
#define cudaEventRecordExternal             0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define cudaEventWaitDefault                0x00  /**< Default event wait flag */
#define cudaEventWaitExternal               0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define cudaDeviceScheduleAuto              0x00  /**< Device flag - Automatic scheduling */
#define cudaDeviceScheduleSpin              0x01  /**< Device flag - Spin default scheduling */
#define cudaDeviceScheduleYield             0x02  /**< Device flag - Yield default scheduling */
#define cudaDeviceScheduleBlockingSync      0x04  /**< Device flag - Use blocking synchronization */
#define cudaDeviceBlockingSync              0x04  /**< Device flag - Use blocking synchronization 
                                                    *  \deprecated This flag was deprecated as of CUDA 4.0 and
                                                    *  replaced with ::cudaDeviceScheduleBlockingSync. */
#define cudaDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define cudaDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define cudaDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define cudaDeviceMask                      0x1f  /**< Device flags mask */

#define cudaArrayDefault                    0x00  /**< Default CUDA array allocation flag */
#define cudaArrayLayered                    0x01  /**< Must be set in cudaMalloc3DArray to create a layered CUDA array */
#define cudaArraySurfaceLoadStore           0x02  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
#define cudaArrayCubemap                    0x04  /**< Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
#define cudaArrayTextureGather              0x08  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */
#define cudaArrayColorAttachment            0x20  /**< Must be set in cudaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API */
#define cudaArraySparse                     0x40  /**< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a sparse CUDA array or CUDA mipmapped array */

#define cudaArrayDeferredMapping            0x80  /**< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a deferred mapping CUDA array or CUDA mipmapped array */


#define cudaIpcMemLazyEnablePeerAccess      0x01  /**< Automatically enable peer access between remote devices as needed */

#define cudaMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define cudaMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define cudaMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

#define cudaOccupancyDefault                0x00  /**< Default behavior */
#define cudaOccupancyDisableCachingOverride 0x01  /**< Assume global caching is enabled and cannot be automatically turned off */

#define cudaCpuDeviceId                     ((int)-1) /**< Device id that represents the CPU */
#define cudaInvalidDeviceId                 ((int)-2) /**< Device id that represents an invalid device */

/**
 * If set, each kernel launched as part of ::cudaLaunchCooperativeKernelMultiDevice only
 * waits for prior work in the stream corresponding to that GPU to complete before the
 * kernel begins execution.
 */
#define cudaCooperativeLaunchMultiDeviceNoPreSync  0x01

/**
 * If set, any subsequent work pushed in a stream that participated in a call to
 * ::cudaLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
 * the GPU corresponding to that stream to complete before it begins execution.
 */
#define cudaCooperativeLaunchMultiDeviceNoPostSync 0x02

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

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

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * CUDA error types
 */
enum __device_builtin__ cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 =     1,
  
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    cudaErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          =      3,
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       =    8,
  
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         =      9,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                =     13,
  
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidDevicePointer         =     17,
  
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       =     21,
  
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         =     25,
  
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           =     27,
  
    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          =     32,
  
    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    cudaErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer CUDA driver than the one
     * currently installed. Users should install an updated NVIDIA CUDA driver
     * to allow the API call to succeed.
     */
    cudaErrorCallRequiresNewerDriver      =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               =     37,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    =     49,
    
    /**
     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         =      52,
  
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           =      53,

    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    cudaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which ::cudaDeviceSynchronize will be called must be specified 
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    cudaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   =     69,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        =      98,
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     =     100,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     */
    cudaErrorInvalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    cudaErrorDeviceNotLicensed            =     102,

   /**
    * By default, the CUDA runtime may perform a minimal set of self-tests,
    * as well as CUDA driver tests, to establish the validity of both.
    * Introduced in CUDA 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   cudaErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =    127,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    cudaErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        =     205,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    cudaErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    cudaErrorAlreadyMapped                =     208,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    cudaErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    cudaErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    cudaErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    cudaErrorNotMappedAsPointer           =     213,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             =     214,
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             =     215,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    cudaErrorNvlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    cudaErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the CUDA driver and PTX JIT compiler.
     */
    cudaErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    cudaErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    cudaErrorUnsupportedExecAffinity      =     224,

    /**
     * This indicates that the device kernel source is invalid.
     */
    cudaErrorInvalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    cudaErrorFileNotFound                 =     301,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   =     302,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              =     304,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    cudaErrorIllegalState                 =     401,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    cudaErrorSymbolNotFound               =     500,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalAddress               =     700,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         =      701,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    cudaErrorLaunchIncompatibleTexturing  =     703,
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     =     704,
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         =     705,
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    cudaErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorAssert                        =    710,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 =     711,
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  =     712,
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorMisalignedAddress            =     716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidPc                    =     718,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    cudaErrorCooperativeLaunchTooLarge    =     720,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    cudaErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    cudaErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    cudaErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    cudaErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    cudaErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    cudaErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    cudaErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    cudaErrorMpsMaxConnectionsReached     =     809,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    cudaErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    cudaErrorStreamCaptureInvalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    cudaErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    cudaErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    cudaErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    cudaErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from cudaStreamLegacy.
     */
    cudaErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    cudaErrorCapturedEvent                =    907,
  
    /**
     * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
     * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
     * different thread.
     */
    cudaErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    cudaErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    cudaErrorGraphExecUpdateFailure       =    910,

    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    cudaErrorExternalDevice               =    911,









    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      =    999,

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorApiFailureBase               =  10000
};

/**
 * Channel format kind
 */
enum __device_builtin__ cudaChannelFormatKind
{
    cudaChannelFormatKindSigned                         =   0,      /**< Signed channel format */
    cudaChannelFormatKindUnsigned                       =   1,      /**< Unsigned channel format */
    cudaChannelFormatKindFloat                          =   2,      /**< Float channel format */
    cudaChannelFormatKindNone                           =   3,      /**< No channel format */
    cudaChannelFormatKindNV12                           =   4,      /**< Unsigned 8-bit integers, planar 4:2:0 YUV format */
    cudaChannelFormatKindUnsignedNormalized8X1          =   5,      /**< 1 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized8X2          =   6,      /**< 2 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized8X4          =   7,      /**< 4 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X1         =   8,      /**< 1 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X2         =   9,      /**< 2 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X4         =   10,     /**< 4 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X1            =   11,     /**< 1 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X2            =   12,     /**< 2 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X4            =   13,     /**< 4 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X1           =   14,     /**< 1 channel signed 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X2           =   15,     /**< 2 channel signed 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X4           =   16,     /**< 4 channel signed 16-bit normalized integer */
    cudaChannelFormatKindUnsignedBlockCompressed1       =   17,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB   =   18,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*/
    cudaChannelFormatKindUnsignedBlockCompressed2       =   19,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB   =   20,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedBlockCompressed3       =   21,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB   =   22,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedBlockCompressed4       =   23,     /**< 1 channel unsigned normalized block-compressed (BC4 compression) format */
    cudaChannelFormatKindSignedBlockCompressed4         =   24,     /**< 1 channel signed normalized block-compressed (BC4 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed5       =   25,     /**< 2 channel unsigned normalized block-compressed (BC5 compression) format */
    cudaChannelFormatKindSignedBlockCompressed5         =   26,     /**< 2 channel signed normalized block-compressed (BC5 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed6H      =   27,     /**< 3 channel unsigned half-float block-compressed (BC6H compression) format */
    cudaChannelFormatKindSignedBlockCompressed6H        =   28,     /**< 3 channel signed half-float block-compressed (BC6H compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed7       =   29,     /**< 4 channel unsigned normalized block-compressed (BC7 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB   =   30      /**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding */
};

/**
 * CUDA Channel format descriptor
 */
struct __device_builtin__ cudaChannelFormatDesc
{
    int                        x; /**< x */
    int                        y; /**< y */
    int                        z; /**< z */
    int                        w; /**< w */
    enum cudaChannelFormatKind f; /**< Channel format kind */
};

/**
 * CUDA array
 */
typedef struct cudaArray *cudaArray_t;

/**
 * CUDA array (as source copy argument)
 */
typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;

/**
 * CUDA mipmapped array
 */
typedef struct cudaMipmappedArray *cudaMipmappedArray_t;

/**
 * CUDA mipmapped array (as source argument)
 */
typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;

/**
 * Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers
 */
#define cudaArraySparsePropertiesSingleMipTail   0x1

/**
 * Sparse CUDA array and CUDA mipmapped array properties
 */
struct __device_builtin__ cudaArraySparseProperties {
    struct {
        unsigned int width;             /**< Tile width in elements */
        unsigned int height;            /**< Tile height in elements */
        unsigned int depth;             /**< Tile depth in elements */
    } tileExtent;
    unsigned int miptailFirstLevel;     /**< First mip level at which the mip tail begins */   
    unsigned long long miptailSize;     /**< Total size of the mip tail. */
    unsigned int flags;                 /**< Flags will either be zero or ::cudaArraySparsePropertiesSingleMipTail */
    unsigned int reserved[4];
};


/**
 * CUDA array and CUDA mipmapped array memory requirements
 */
struct __device_builtin__ cudaArrayMemoryRequirements {
    size_t size;                    /**< Total size of the array. */
    size_t alignment;               /**< Alignment necessary for mapping the array. */
    unsigned int reserved[4];
};


/**
 * CUDA memory types
 */
enum __device_builtin__ cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
    cudaMemoryTypeHost         = 1, /**< Host memory */
    cudaMemoryTypeDevice       = 2, /**< Device memory */
    cudaMemoryTypeManaged      = 3  /**< Managed memory */
};

/**
 * CUDA memory copy types
 */
enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * CUDA Pitched memory pointer
 *
 * \sa ::make_cudaPitchedPtr
 */
struct __device_builtin__ cudaPitchedPtr
{
    void   *ptr;      /**< Pointer to allocated memory */
    size_t  pitch;    /**< Pitch of allocated memory in bytes */
    size_t  xsize;    /**< Logical width of allocation in elements */
    size_t  ysize;    /**< Logical height of allocation in elements */
};

/**
 * CUDA extent
 *
 * \sa ::make_cudaExtent
 */
struct __device_builtin__ cudaExtent
{
    size_t width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
    size_t height;    /**< Height in elements */
    size_t depth;     /**< Depth in elements */
};

/**
 * CUDA 3D position
 *
 * \sa ::make_cudaPos
 */
struct __device_builtin__ cudaPos
{
    size_t x;     /**< x */
    size_t y;     /**< y */
    size_t z;     /**< z */
};

/**
 * CUDA 3D memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
    enum cudaMemcpyKind    kind;      /**< Type of transfer */
};

/**
 * CUDA 3D cross-device memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
    int                    srcDevice; /**< Source device */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
    int                    dstDevice; /**< Destination device */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
};

/**
 * CUDA Memset node parameters
 */
struct __device_builtin__  cudaMemsetParams {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
 */
enum __device_builtin__  cudaAccessProperty {
    cudaAccessPropertyNormal = 0,       /**< Normal cache persistence. */
    cudaAccessPropertyStreaming = 1,    /**< Streaming access is less likely to persit from cache. */
    cudaAccessPropertyPersisting = 2    /**< Persisting access is more likely to persist in cache.*/
};

/**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 */
struct __device_builtin__ cudaAccessPolicyWindow {
    void *base_ptr;                     /**< Starting address of the access policy window. CUDA driver may align it. */
    size_t num_bytes;                   /**< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. */
    float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    enum cudaAccessProperty hitProp;    /**< ::CUaccessProperty set for hit. */
    enum cudaAccessProperty missProp;   /**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING. */
};

#ifdef _WIN32
#define CUDART_CB __stdcall
#else
#define CUDART_CB
#endif

/**
 * CUDA host function
 * \param userData Argument value passed to the function
 */
typedef void (CUDART_CB *cudaHostFn_t)(void *userData);

/**
 * CUDA host node parameters
 */
struct __device_builtin__ cudaHostNodeParams {
    cudaHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * Possible stream capture statuses returned by ::cudaStreamIsCapturing
 */
enum __device_builtin__ cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
    cudaStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
    cudaStreamCaptureStatusInvalidated = 2  /**< Stream is part of a capture sequence that
                                                   has been invalidated, but not terminated */
};

/**
 * Possible modes for stream capture thread interactions. For more details see
 * ::cudaStreamBeginCapture and ::cudaThreadExchangeStreamCaptureMode
 */
enum __device_builtin__ cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2
};

enum __device_builtin__ cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4
};

/**
 * Stream Attributes
 */
enum __device_builtin__ cudaStreamAttrID {
    cudaStreamAttributeAccessPolicyWindow     = 1,  /**< Identifier for ::cudaStreamAttrValue::accessPolicyWindow. */
    cudaStreamAttributeSynchronizationPolicy  = 3   /**< ::cudaSynchronizationPolicy for work queued up in this stream */
};

/**
 * Stream attributes union used with ::cudaStreamSetAttribute/::cudaStreamGetAttribute
 */
union __device_builtin__ cudaStreamAttrValue {
    struct cudaAccessPolicyWindow accessPolicyWindow;
    enum cudaSynchronizationPolicy syncPolicy;
};

/**
 * Flags for ::cudaStreamUpdateCaptureDependencies
 */
enum __device_builtin__ cudaStreamUpdateCaptureDependenciesFlags {
    cudaStreamAddCaptureDependencies = 0x0, /**< Add new nodes to the dependency set */
    cudaStreamSetCaptureDependencies = 0x1  /**< Replace the dependency set with the new nodes */
};

/**
 * Flags for user objects for graphs
 */
enum __device_builtin__ cudaUserObjectFlags {
    cudaUserObjectNoDestructorSync = 0x1  /**< Indicates the destructor execution is not synchronized by any CUDA handle. */
};

/**
 * Flags for retaining user object references for graphs
 */
enum __device_builtin__ cudaUserObjectRetainFlags {
    cudaGraphUserObjectMove = 0x1  /**< Transfer references from the caller rather than creating new references. */
};

/**
 * CUDA graphics interop resource
 */
struct cudaGraphicsResource;

/**
 * CUDA graphics interop register flags
 */
enum __device_builtin__ cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  /**< Default */
    cudaGraphicsRegisterFlagsReadOnly         = 1,  /**< CUDA will not write to this resource */ 
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  /**< CUDA will only write to and will not read from this resource */
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< CUDA will bind this resource to a surface reference */
    cudaGraphicsRegisterFlagsTextureGather    = 8   /**< CUDA will perform texture gather operations on this resource */
};

/**
 * CUDA graphics interop map flags
 */
enum __device_builtin__ cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
    cudaGraphicsMapFlagsReadOnly     = 1,  /**< CUDA will not write to this resource */
    cudaGraphicsMapFlagsWriteDiscard = 2   /**< CUDA will only write to and will not read from this resource */
};

/**
 * CUDA graphics interop array indices for cube maps
 */
enum __device_builtin__ cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, /**< Positive X face of cubemap */
    cudaGraphicsCubeFaceNegativeX = 0x01, /**< Negative X face of cubemap */
    cudaGraphicsCubeFacePositiveY = 0x02, /**< Positive Y face of cubemap */
    cudaGraphicsCubeFaceNegativeY = 0x03, /**< Negative Y face of cubemap */
    cudaGraphicsCubeFacePositiveZ = 0x04, /**< Positive Z face of cubemap */
    cudaGraphicsCubeFaceNegativeZ = 0x05  /**< Negative Z face of cubemap */
};

/**
 * Graph kernel node Attributes
 */
enum __device_builtin__ cudaKernelNodeAttrID {
    cudaKernelNodeAttributeAccessPolicyWindow   = 1,   /**< Identifier for ::cudaKernelNodeAttrValue::accessPolicyWindow. */
    cudaKernelNodeAttributeCooperative          = 2    /**< Allows a kernel node to be cooperative (see ::cudaLaunchCooperativeKernel). */
};

/**
 * Graph kernel node attributes union, used with ::cudaGraphKernelNodeSetAttribute/::cudaGraphKernelNodeGetAttribute
 */
union __device_builtin__ cudaKernelNodeAttrValue {
    struct cudaAccessPolicyWindow accessPolicyWindow;          /**< Attribute ::CUaccessPolicyWindow. */
    int cooperative;
};

/**
 * CUDA resource types
 */
enum __device_builtin__ cudaResourceType
{
    cudaResourceTypeArray          = 0x00, /**< Array resource */
    cudaResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
    cudaResourceTypeLinear         = 0x02, /**< Linear resource */
    cudaResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
};

/**
 * CUDA texture resource view formats
 */
enum __device_builtin__ cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, /**< No resource view format (use underlying resource format) */
    cudaResViewFormatUnsignedChar1             = 0x01, /**< 1 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar2             = 0x02, /**< 2 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar4             = 0x03, /**< 4 channel unsigned 8-bit integers */
    cudaResViewFormatSignedChar1               = 0x04, /**< 1 channel signed 8-bit integers */
    cudaResViewFormatSignedChar2               = 0x05, /**< 2 channel signed 8-bit integers */
    cudaResViewFormatSignedChar4               = 0x06, /**< 4 channel signed 8-bit integers */
    cudaResViewFormatUnsignedShort1            = 0x07, /**< 1 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort2            = 0x08, /**< 2 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort4            = 0x09, /**< 4 channel unsigned 16-bit integers */
    cudaResViewFormatSignedShort1              = 0x0a, /**< 1 channel signed 16-bit integers */
    cudaResViewFormatSignedShort2              = 0x0b, /**< 2 channel signed 16-bit integers */
    cudaResViewFormatSignedShort4              = 0x0c, /**< 4 channel signed 16-bit integers */
    cudaResViewFormatUnsignedInt1              = 0x0d, /**< 1 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt2              = 0x0e, /**< 2 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt4              = 0x0f, /**< 4 channel unsigned 32-bit integers */
    cudaResViewFormatSignedInt1                = 0x10, /**< 1 channel signed 32-bit integers */
    cudaResViewFormatSignedInt2                = 0x11, /**< 2 channel signed 32-bit integers */
    cudaResViewFormatSignedInt4                = 0x12, /**< 4 channel signed 32-bit integers */
    cudaResViewFormatHalf1                     = 0x13, /**< 1 channel 16-bit floating point */
    cudaResViewFormatHalf2                     = 0x14, /**< 2 channel 16-bit floating point */
    cudaResViewFormatHalf4                     = 0x15, /**< 4 channel 16-bit floating point */
    cudaResViewFormatFloat1                    = 0x16, /**< 1 channel 32-bit floating point */
    cudaResViewFormatFloat2                    = 0x17, /**< 2 channel 32-bit floating point */
    cudaResViewFormatFloat4                    = 0x18, /**< 4 channel 32-bit floating point */
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, /**< Block compressed 1 */
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, /**< Block compressed 2 */
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, /**< Block compressed 3 */
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, /**< Block compressed 4 unsigned */
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, /**< Block compressed 4 signed */
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, /**< Block compressed 5 unsigned */
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, /**< Block compressed 5 signed */
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, /**< Block compressed 6 unsigned half-float */
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, /**< Block compressed 6 signed half-float */
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  /**< Block compressed 7 */
};

/**
 * CUDA resource descriptor
 */
struct __device_builtin__ cudaResourceDesc {
    enum cudaResourceType resType;             /**< Resource type */
    
    union {
        struct {
            cudaArray_t array;                 /**< CUDA array */
        } array;
        struct {
            cudaMipmappedArray_t mipmap;       /**< CUDA mipmapped array */
        } mipmap;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct cudaChannelFormatDesc desc; /**< Channel descriptor */
            size_t sizeInBytes;                /**< Size in bytes */
        } linear;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct cudaChannelFormatDesc desc; /**< Channel descriptor */
            size_t width;                      /**< Width of the array in elements */
            size_t height;                     /**< Height of the array in elements */
            size_t pitchInBytes;               /**< Pitch between two rows in bytes */
        } pitch2D;
    } res;
};

/**
 * CUDA resource view descriptor
 */
struct __device_builtin__ cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           /**< Resource view format */
    size_t                      width;            /**< Width of the resource view */
    size_t                      height;           /**< Height of the resource view */
    size_t                      depth;            /**< Depth of the resource view */
    unsigned int                firstMipmapLevel; /**< First defined mipmap level */
    unsigned int                lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int                firstLayer;       /**< First layer index */
    unsigned int                lastLayer;        /**< Last layer index */
};

/**
 * CUDA pointer attributes
 */
struct __device_builtin__ cudaPointerAttributes
{
    /**
     * The type of memory - ::cudaMemoryTypeUnregistered, ::cudaMemoryTypeHost,
     * ::cudaMemoryTypeDevice or ::cudaMemoryTypeManaged.
     */
    enum cudaMemoryType type;

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::cudaMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::cudaMemoryTypeHost or::cudaMemoryTypeManaged then
     * this identifies the device which was current when the memory was allocated
     * or registered (and if that device is deinitialized then this allocation
     * will vanish with that device's state).
     */
    int device;

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    void *devicePointer;

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     *
     * \note CUDA doesn't check if unregistered memory is allocated so this field
     * may contain invalid pointer if an invalid pointer has been passed to CUDA.
     */
    void *hostPointer;
};

/**
 * CUDA function attributes
 */
struct __device_builtin__ cudaFuncAttributes
{
   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   size_t sharedSizeBytes;

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   size_t constSizeBytes;

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   size_t localSizeBytes;

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is currently loaded.
    */
   int maxThreadsPerBlock;

   /**
    * The number of registers used by each thread of this function.
    */
   int numRegs;

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   int ptxVersion;

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   int binaryVersion;

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   int cacheModeCA;

   /**
    * The maximum size in bytes of dynamic shared memory per block for 
    * this function. Any launch must have a dynamic shared memory size
    * smaller than this value.
    */
   int maxDynamicSharedSizeBytes;

   /**
    * On devices where the L1 cache and shared memory use the same hardware resources, 
    * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
    * Refer to ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
    * This is only a hint, and the driver can choose a different ratio if required to execute the function.
    * See ::cudaFuncSetAttribute
    */
   int preferredShmemCarveout;

















































};

/**
 * CUDA function attributes that can be set using ::cudaFuncSetAttribute
 */
enum __device_builtin__ cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    cudaFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */








    cudaFuncAttributeMax
};

/**
 * CUDA function cache configurations
 */
enum __device_builtin__ cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
    cudaFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
    cudaFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
    cudaFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
};

/**
 * CUDA shared memory configuration
 */

enum __device_builtin__ cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

/** 
 * Shared memory carveout configurations. These may be passed to cudaFuncSetAttribute
 */
enum __device_builtin__ cudaSharedCarveout {
    cudaSharedmemCarveoutDefault      = -1,  /**< No preference for shared memory or L1 (default) */
    cudaSharedmemCarveoutMaxShared    = 100, /**< Prefer maximum available shared memory, minimum L1 cache */
    cudaSharedmemCarveoutMaxL1        = 0    /**< Prefer maximum available L1 cache, minimum shared memory */
};












/**
 * CUDA device compute modes
 */
enum __device_builtin__ cudaComputeMode
{
    cudaComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
};

/**
 * CUDA Limits
 */
enum __device_builtin__ cudaLimit
{
    cudaLimitStackSize                    = 0x00, /**< GPU thread stack size */
    cudaLimitPrintfFifoSize               = 0x01, /**< GPU printf FIFO size */
    cudaLimitMallocHeapSize               = 0x02, /**< GPU malloc heap size */
    cudaLimitDevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
    cudaLimitDevRuntimePendingLaunchCount = 0x04, /**< GPU device runtime pending launch count */
    cudaLimitMaxL2FetchGranularity        = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    cudaLimitPersistingL2CacheSize        = 0x06  /**< A size in bytes for L2 persisting lines cache size */
};

/**
 * CUDA Memory Advise values
 */
enum __device_builtin__ cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly          = 1, /**< Data will mostly be read and only occassionally be written to */
    cudaMemAdviseUnsetReadMostly        = 2, /**< Undo the effect of ::cudaMemAdviseSetReadMostly */
    cudaMemAdviseSetPreferredLocation   = 3, /**< Set the preferred location for the data as the specified device */
    cudaMemAdviseUnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
    cudaMemAdviseSetAccessedBy          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    cudaMemAdviseUnsetAccessedBy        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
};

/**
 * CUDA range attributes
 */
enum __device_builtin__ cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly           = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    cudaMemRangeAttributePreferredLocation    = 2, /**< The preferred location of the range */
    cudaMemRangeAttributeAccessedBy           = 3, /**< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device */
    cudaMemRangeAttributeLastPrefetchLocation = 4  /**< The last location to which the range was prefetched */
};

/**
 * CUDA Profiler Output modes
 */
enum __device_builtin__ cudaOutputMode
{
    cudaKeyValuePair    = 0x00, /**< Output mode Key-Value pair format. */
    cudaCSV             = 0x01  /**< Output mode Comma separated values format. */
};

/**
 * CUDA GPUDirect RDMA flush writes APIs supported on the device
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesOptions {
    cudaFlushGPUDirectRDMAWritesOptionHost   = 1<<0, /**< ::cudaDeviceFlushGPUDirectRDMAWrites() and its CUDA Driver API counterpart are supported on the device. */
    cudaFlushGPUDirectRDMAWritesOptionMemOps = 1<<1  /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the CUDA device. */
};

/**
 * CUDA GPUDirect RDMA flush writes ordering features of the device
 */
enum __device_builtin__ cudaGPUDirectRDMAWritesOrdering {
    cudaGPUDirectRDMAWritesOrderingNone       = 0,   /**< The device does not natively support ordering of GPUDirect RDMA writes. ::cudaFlushGPUDirectRDMAWrites() can be leveraged if supported. */
    cudaGPUDirectRDMAWritesOrderingOwner      = 100, /**< Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA devices may not. */
    cudaGPUDirectRDMAWritesOrderingAllDevices = 200  /**< Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device. */
};

/**
 * CUDA GPUDirect RDMA flush writes scopes
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesScope {
    cudaFlushGPUDirectRDMAWritesToOwner      = 100, /**< Blocks until remote writes are visible to the CUDA device context owning the data. */
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200  /**< Blocks until remote writes are visible to all CUDA device contexts. */
};

/**
 * CUDA GPUDirect RDMA flush writes targets
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesTarget {
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice /**< Sets the target for ::cudaDeviceFlushGPUDirectRDMAWrites() to the currently active CUDA device context. */
};


/**
 * CUDA device attributes
 */
enum __device_builtin__ cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode                    = 20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    cudaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    cudaDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    cudaDevAttrConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    cudaDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    cudaDevAttrReserved92                     = 92,
    cudaDevAttrReserved93                     = 93,
    cudaDevAttrReserved94                     = 94,
    cudaDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
    cudaDevAttrCooperativeMultiDeviceLaunch   = 96, /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
    cudaDevAttrCanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
    cudaDevAttrHostRegisterSupported          = 99, /**< Device supports host memory registration via ::cudaHostRegister. */
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    cudaDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    cudaDevAttrMaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
    cudaDevAttrMaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    cudaDevAttrMaxAccessPolicyWindowSize      = 109, /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
    cudaDevAttrReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by CUDA driver per block in bytes */
    cudaDevAttrSparseCudaArraySupported       = 112, /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    cudaDevAttrHostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,  /**< Deprecated, External timeline semaphore interop is supported on the device */
    cudaDevAttrMemoryPoolsSupported           = 115, /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrGPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    cudaDevAttrGPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */




    cudaDevAttrDeferredMappingCudaArraySupported = 121, /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */

    cudaDevAttrMax
};

/**
 * CUDA memory pool attributes
 */
enum __device_builtin__ cudaMemPoolAttr
{
    /**
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    cudaMemPoolReuseFollowEventDependencies   = 0x1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    cudaMemPoolReuseAllowOpportunistic        = 0x2,

    /**
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    cudaMemPoolReuseAllowInternalDependencies = 0x3,


    /**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    cudaMemPoolAttrReleaseThreshold           = 0x4,

    /**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    cudaMemPoolAttrReservedMemCurrent         = 0x5,

    /**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrReservedMemHigh            = 0x6,

    /**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    cudaMemPoolAttrUsedMemCurrent             = 0x7,

    /**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrUsedMemHigh                = 0x8
};

/**
 * Specifies the type of location 
 */
enum __device_builtin__ cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1  /**< Location is a device location, thus id is a device ordinal */
};

/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = ::cudaMemLocationTypeDevice and set id = the gpu's device ordinal.
 */
struct __device_builtin__ cudaMemLocation {
    enum cudaMemLocationType type;  /**< Specifies the location type, which modifies the meaning of id. */
    int id;                         /**< identifier for a given this location's ::CUmemLocationType. */
};

/**
 * Specifies the memory protection flags for mapping.
 */
enum __device_builtin__ cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone      = 0,  /**< Default, make the address range not accessible */
    cudaMemAccessFlagsProtRead      = 1,  /**< Make the address range read accessible */
    cudaMemAccessFlagsProtReadWrite = 3   /**< Make the address range read-write accessible */
};

/**
 * Memory access descriptor
 */
struct __device_builtin__ cudaMemAccessDesc {
    struct cudaMemLocation  location; /**< Location on which the request is to change it's accessibility */
    enum cudaMemAccessFlags flags;    /**< ::CUmemProt accessibility flags to set on the request */
};

/**
 * Defines the allocation types available
 */
enum __device_builtin__ cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    cudaMemAllocationTypePinned  = 0x1,
    cudaMemAllocationTypeMax     = 0x7FFFFFFF 
};

/**
 * Flags for specifying particular handle types
 */
enum __device_builtin__ cudaMemAllocationHandleType {
    cudaMemHandleTypeNone                    = 0x0,  /**< Does not allow any export mechanism. > */
    cudaMemHandleTypePosixFileDescriptor     = 0x1,  /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
    cudaMemHandleTypeWin32                   = 0x2,  /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
    cudaMemHandleTypeWin32Kmt                = 0x4   /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
};

/**
 * Specifies the properties of allocations made from the pool.
 */
struct __device_builtin__ cudaMemPoolProps {
    enum cudaMemAllocationType         allocType;   /**< Allocation type. Currently must be specified as cudaMemAllocationTypePinned */
    enum cudaMemAllocationHandleType   handleTypes; /**< Handle types that will be supported by allocations from the pool. */
    struct cudaMemLocation             location;    /**< Location allocations should reside. */
    /**
     * Windows-specific LPSECURITYATTRIBUTES required when
     * ::cudaMemHandleTypeWin32 is specified.  This security attribute defines
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     */
    void                              *win32SecurityAttributes;
    unsigned char                      reserved[64]; /**< reserved for future use, must be 0 */
};

/**
 * Opaque data for exporting a pool allocation
 */
struct __device_builtin__ cudaMemPoolPtrExportData {
    unsigned char reserved[64];
};

/**
 * Memory allocation node parameters
 */
struct __device_builtin__ cudaMemAllocNodeParams {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::cudaMemHandleTypeNone. IPC is not supported.
    */
    struct cudaMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct cudaMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by CUDA */
};

/**
 * Graph memory attributes
 */
enum __device_builtin__ cudaGraphMemAttributeType {
    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently associated with graphs.
     */
    cudaGraphMemAttrUsedMemCurrent      = 0x0,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, associated with graphs since the
     * last time it was reset.  High watermark can only be reset to zero.
     */
    cudaGraphMemAttrUsedMemHigh         = 0x1,

    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     */
    cudaGraphMemAttrReservedMemCurrent  = 0x2,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     */
    cudaGraphMemAttrReservedMemHigh     = 0x3
};

/**
 * CUDA device P2P attributes
 */

enum __device_builtin__ cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
    cudaDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
    cudaDevP2PAttrNativeAtomicSupported        = 3, /**< Native atomic operation over the link supported */
    cudaDevP2PAttrCudaArrayAccessSupported     = 4  /**< Accessing CUDA arrays over the link supported */
};

/**
 * CUDA UUID types
 */
#ifndef CU_UUID_HAS_BEEN_DEFINED
#define CU_UUID_HAS_BEEN_DEFINED
struct __device_builtin__ CUuuid_st {     /**< CUDA definition of UUID */
    char bytes[16];
};
typedef __device_builtin__ struct CUuuid_st CUuuid;
#endif
typedef __device_builtin__ struct CUuuid_st cudaUUID_t;

/**
 * CUDA device properties
 */
struct __device_builtin__ cudaDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    int          clockRate;                  /**< Clock frequency in kilohertz */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int          computeMode;                /**< Compute mode (See ::cudaComputeMode) */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture1DLinear;         /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          singleToDoublePrecisionPerfRatio; /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    int          cooperativeMultiDeviceLaunch; /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
};

#define cudaDevicePropDontCare                                 \
        {                                                      \
          {'\0'},    /* char         name[256];               */ \
          {{0}},     /* cudaUUID_t   uuid;                    */ \
          {'\0'},    /* char         luid[8];                 */ \
          0,         /* unsigned int luidDeviceNodeMask       */ \
          0,         /* size_t       totalGlobalMem;          */ \
          0,         /* size_t       sharedMemPerBlock;       */ \
          0,         /* int          regsPerBlock;            */ \
          0,         /* int          warpSize;                */ \
          0,         /* size_t       memPitch;                */ \
          0,         /* int          maxThreadsPerBlock;      */ \
          {0, 0, 0}, /* int          maxThreadsDim[3];        */ \
          {0, 0, 0}, /* int          maxGridSize[3];          */ \
          0,         /* int          clockRate;               */ \
          0,         /* size_t       totalConstMem;           */ \
          -1,        /* int          major;                   */ \
          -1,        /* int          minor;                   */ \
          0,         /* size_t       textureAlignment;        */ \
          0,         /* size_t       texturePitchAlignment    */ \
          -1,        /* int          deviceOverlap;           */ \
          0,         /* int          multiProcessorCount;     */ \
          0,         /* int          kernelExecTimeoutEnabled */ \
          0,         /* int          integrated               */ \
          0,         /* int          canMapHostMemory         */ \
          0,         /* int          computeMode              */ \
          0,         /* int          maxTexture1D             */ \
          0,         /* int          maxTexture1DMipmap       */ \
          0,         /* int          maxTexture1DLinear       */ \
          {0, 0},    /* int          maxTexture2D[2]          */ \
          {0, 0},    /* int          maxTexture2DMipmap[2]    */ \
          {0, 0, 0}, /* int          maxTexture2DLinear[3]    */ \
          {0, 0},    /* int          maxTexture2DGather[2]    */ \
          {0, 0, 0}, /* int          maxTexture3D[3]          */ \
          {0, 0, 0}, /* int          maxTexture3DAlt[3]       */ \
          0,         /* int          maxTextureCubemap        */ \
          {0, 0},    /* int          maxTexture1DLayered[2]   */ \
          {0, 0, 0}, /* int          maxTexture2DLayered[3]   */ \
          {0, 0},    /* int          maxTextureCubemapLayered[2] */ \
          0,         /* int          maxSurface1D             */ \
          {0, 0},    /* int          maxSurface2D[2]          */ \
          {0, 0, 0}, /* int          maxSurface3D[3]          */ \
          {0, 0},    /* int          maxSurface1DLayered[2]   */ \
          {0, 0, 0}, /* int          maxSurface2DLayered[3]   */ \
          0,         /* int          maxSurfaceCubemap        */ \
          {0, 0},    /* int          maxSurfaceCubemapLayered[2] */ \
          0,         /* size_t       surfaceAlignment         */ \
          0,         /* int          concurrentKernels        */ \
          0,         /* int          ECCEnabled               */ \
          0,         /* int          pciBusID                 */ \
          0,         /* int          pciDeviceID              */ \
          0,         /* int          pciDomainID              */ \
          0,         /* int          tccDriver                */ \
          0,         /* int          asyncEngineCount         */ \
          0,         /* int          unifiedAddressing        */ \
          0,         /* int          memoryClockRate          */ \
          0,         /* int          memoryBusWidth           */ \
          0,         /* int          l2CacheSize              */ \
          0,         /* int          persistingL2CacheMaxSize   */ \
          0,         /* int          maxThreadsPerMultiProcessor */ \
          0,         /* int          streamPrioritiesSupported */ \
          0,         /* int          globalL1CacheSupported   */ \
          0,         /* int          localL1CacheSupported    */ \
          0,         /* size_t       sharedMemPerMultiprocessor; */ \
          0,         /* int          regsPerMultiprocessor;   */ \
          0,         /* int          managedMemory            */ \
          0,         /* int          isMultiGpuBoard          */ \
          0,         /* int          multiGpuBoardGroupID     */ \
          0,         /* int          hostNativeAtomicSupported */ \
          0,         /* int          singleToDoublePrecisionPerfRatio */ \
          0,         /* int          pageableMemoryAccess     */ \
          0,         /* int          concurrentManagedAccess  */ \
          0,         /* int          computePreemptionSupported */ \
          0,         /* int          canUseHostPointerForRegisteredMem */ \
          0,         /* int          cooperativeLaunch */ \
          0,         /* int          cooperativeMultiDeviceLaunch */ \
          0,         /* size_t       sharedMemPerBlockOptin */ \
          0,         /* int          pageableMemoryAccessUsesHostPageTables */ \
          0,         /* int          directManagedMemAccessFromHost */ \
          0,         /* int          accessPolicyMaxWindowSize */ \
          0,         /* size_t       reservedSharedMemPerBlock */ \
        } /**< Empty device properties */






/**
 * CUDA IPC Handle Size
 */
#define CUDA_IPC_HANDLE_SIZE 64

/**
 * CUDA IPC event handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcEventHandle_st
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcEventHandle_t;

/**
 * CUDA IPC memory handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcMemHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcMemHandle_t;

/**
 * External memory handle types
 */
enum __device_builtin__ cudaExternalMemoryHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    cudaExternalMemoryHandleTypeOpaqueFd         = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    cudaExternalMemoryHandleTypeOpaqueWin32      = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
    /**
     * Handle is a D3D12 heap object
     */
    cudaExternalMemoryHandleTypeD3D12Heap        = 4,
    /**
     * Handle is a D3D12 committed resource
     */
    cudaExternalMemoryHandleTypeD3D12Resource    = 5,
    /**
    *  Handle is a shared NT handle to a D3D11 resource
    */
    cudaExternalMemoryHandleTypeD3D11Resource    = 6,
    /**
    *  Handle is a globally shared handle to a D3D11 resource
    */
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    /**
    *  Handle is an NvSciBuf object
    */
    cudaExternalMemoryHandleTypeNvSciBuf         = 8
};

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define cudaExternalMemoryDedicated   0x1

/** When the /p flags parameter of ::cudaExternalSemaphoreSignalParams
 * contains this flag, it indicates that signaling an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define cudaExternalSemaphoreSignalSkipNvSciBufMemSync     0x01

/** When the /p flags parameter of ::cudaExternalSemaphoreWaitParams
 * contains this flag, it indicates that waiting an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define cudaExternalSemaphoreWaitSkipNvSciBufMemSync       0x02

/**
 * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need signaler specific NvSciSyncAttr
 * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
 */
#define cudaNvSciSyncAttrSignal       0x1

/**
 * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need waiter specific NvSciSyncAttr
 * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
 */
#define cudaNvSciSyncAttrWait         0x2

/**
 * External memory handle descriptor
 */
struct __device_builtin__ cudaExternalMemoryHandleDesc {
    /**
     * Type of the handle
     */
    enum cudaExternalMemoryHandleType type;
    union {
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::cudaExternalMemoryHandleTypeOpaqueFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalMemoryHandleTypeD3D12Heap 
         * - ::cudaExternalMemoryHandleTypeD3D12Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following: 
         * ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * A handle representing NvSciBuf Object. Valid when type
         * is ::cudaExternalMemoryHandleTypeNvSciBuf
         */
        const void *nvSciBufObject;
    } handle;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
    /**
     * Flags must either be zero or ::cudaExternalMemoryDedicated
     */
    unsigned int flags;
};

/**
 * External memory buffer descriptor
 */
struct __device_builtin__ cudaExternalMemoryBufferDesc {
    /**
     * Offset into the memory object where the buffer's base is
     */
    unsigned long long offset;
    /**
     * Size of the buffer
     */
    unsigned long long size;
    /**
     * Flags reserved for future use. Must be zero.
     */
    unsigned int flags;
};
 
/**
 * External memory mipmap descriptor
 */
struct __device_builtin__ cudaExternalMemoryMipmappedArrayDesc {
    /**
     * Offset into the memory object where the base level of the
     * mipmap chain is.
     */
    unsigned long long offset;
    /**
     * Format of base level of the mipmap chain
     */
    struct cudaChannelFormatDesc formatDesc;
    /**
     * Dimensions of base level of the mipmap chain
     */
    struct cudaExtent extent;
    /**
     * Flags associated with CUDA mipmapped arrays.
     * See ::cudaMallocMipmappedArray
     */
    unsigned int flags;
    /**
     * Total number of levels in the mipmap chain
     */
    unsigned int numLevels;
};
 
/**
 * External semaphore handle types
 */
enum __device_builtin__ cudaExternalSemaphoreHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    cudaExternalSemaphoreHandleTypeOpaqueFd       = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    cudaExternalSemaphoreHandleTypeOpaqueWin32    = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    /**
     * Handle is a shared NT handle referencing a D3D12 fence object
     */
    cudaExternalSemaphoreHandleTypeD3D12Fence     = 4,
    /**
     * Handle is a shared NT handle referencing a D3D11 fence object
     */
    cudaExternalSemaphoreHandleTypeD3D11Fence     = 5,
    /**
     * Opaque handle to NvSciSync Object
     */
     cudaExternalSemaphoreHandleTypeNvSciSync     = 6,
    /**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     */
    cudaExternalSemaphoreHandleTypeKeyedMutex     = 7,
    /**
     * Handle is a shared KMT handle referencing a D3D11 keyed mutex object
     */
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt  = 8,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd  = 9,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
};

/**
 * External semaphore handle descriptor
 */
struct __device_builtin__ cudaExternalSemaphoreHandleDesc {
    /**
     * Type of the handle
     */
    enum cudaExternalSemaphoreHandleType type;
    union {
        /**
         * File descriptor referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalSemaphoreHandleTypeOpaqueFd
         * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32
         * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalSemaphoreHandleTypeD3D12Fence
         * - ::cudaExternalSemaphoreHandleTypeD3D11Fence
         * - ::cudaExternalSemaphoreHandleTypeKeyedMutex
         * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid synchronization primitive.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * Valid NvSciSyncObj. Must be non NULL
         */
        const void* nvSciSyncObj;
    } handle;
    /**
     * Flags reserved for the future. Must be zero.
     */
    unsigned int flags;
};

/**
 * External semaphore signal parameters(deprecated)
 */
struct __device_builtin__ cudaExternalSemaphoreSignalParams_v1 {
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
* External semaphore wait parameters(deprecated)
*/
struct __device_builtin__ cudaExternalSemaphoreWaitParams_v1 {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
};

/**
 * External semaphore signal parameters, compatible with driver type
 */
struct __device_builtin__ cudaExternalSemaphoreSignalParams{
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/**
 * External semaphore wait parameters, compatible with driver type
 */
struct __device_builtin__ cudaExternalSemaphoreWaitParams {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};


/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * CUDA Error types
 */
typedef __device_builtin__ enum cudaError cudaError_t;

/**
 * CUDA stream
 */
typedef __device_builtin__ struct CUstream_st *cudaStream_t;

/**
 * CUDA event types
 */
typedef __device_builtin__ struct CUevent_st *cudaEvent_t;

/**
 * CUDA graphics resource types
 */
typedef __device_builtin__ struct cudaGraphicsResource *cudaGraphicsResource_t;

/**
 * CUDA output file modes
 */
typedef __device_builtin__ enum cudaOutputMode cudaOutputMode_t;

/**
 * CUDA external memory
 */
typedef __device_builtin__ struct CUexternalMemory_st *cudaExternalMemory_t;

/**
 * CUDA external semaphore
 */
typedef __device_builtin__ struct CUexternalSemaphore_st *cudaExternalSemaphore_t;

/**
 * CUDA graph
 */
typedef __device_builtin__ struct CUgraph_st *cudaGraph_t;

/**
 * CUDA graph node.
 */
typedef __device_builtin__ struct CUgraphNode_st *cudaGraphNode_t;

/**
 * CUDA user object for graphs
 */
typedef __device_builtin__ struct CUuserObject_st *cudaUserObject_t;

/**
 * CUDA function
 */
typedef __device_builtin__ struct CUfunc_st *cudaFunction_t;

/**
 * CUDA memory pool
 */
typedef __device_builtin__ struct CUmemPoolHandle_st *cudaMemPool_t;

/**
 * CUDA cooperative group scope
 */
enum __device_builtin__ cudaCGScope {
    cudaCGScopeInvalid   = 0, /**< Invalid cooperative group scope */
    cudaCGScopeGrid      = 1, /**< Scope represented by a grid_group */
    cudaCGScopeMultiGrid = 2  /**< Scope represented by a multi_grid_group */
};

/**
 * CUDA launch parameters
 */
struct __device_builtin__ cudaLaunchParams
{
    void *func;          /**< Device function symbol */
    dim3 gridDim;        /**< Grid dimentions */
    dim3 blockDim;       /**< Block dimentions */
    void **args;         /**< Arguments */
    size_t sharedMem;    /**< Shared memory */
    cudaStream_t stream; /**< Stream identifier */
};

/**
 * CUDA GPU kernel node parameters
 */
struct __device_builtin__ cudaKernelNodeParams {
    void* func;                     /**< Kernel to launch */
    dim3 gridDim;                   /**< Grid dimensions */
    dim3 blockDim;                  /**< Block dimensions */
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * External semaphore signal node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreSignalNodeParams {
    cudaExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreWaitNodeParams {
    cudaExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
* CUDA Graph node types
*/
enum __device_builtin__ cudaGraphNodeType {
    cudaGraphNodeTypeKernel      = 0x00, /**< GPU kernel node */
    cudaGraphNodeTypeMemcpy      = 0x01, /**< Memcpy node */
    cudaGraphNodeTypeMemset      = 0x02, /**< Memset node */
    cudaGraphNodeTypeHost        = 0x03, /**< Host (executable) node */
    cudaGraphNodeTypeGraph       = 0x04, /**< Node which executes an embedded graph */
    cudaGraphNodeTypeEmpty       = 0x05, /**< Empty (no-op) node */
    cudaGraphNodeTypeWaitEvent   = 0x06, /**< External event wait node */
    cudaGraphNodeTypeEventRecord = 0x07, /**< External event record node */
    cudaGraphNodeTypeExtSemaphoreSignal = 0x08, /**< External semaphore signal node */
    cudaGraphNodeTypeExtSemaphoreWait = 0x09, /**< External semaphore wait node */
    cudaGraphNodeTypeMemAlloc    = 0x0a, /**< Memory allocation node */
    cudaGraphNodeTypeMemFree     = 0x0b, /**< Memory free node */
    cudaGraphNodeTypeCount
};

/**
 * CUDA executable (launchable) graph
 */
typedef struct CUgraphExec_st* cudaGraphExec_t;

/**
* CUDA Graph Update error types
*/
enum __device_builtin__ cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess                = 0x0, /**< The update succeeded */
    cudaGraphExecUpdateError                  = 0x1, /**< The update failed for an unexpected reason which is described in the return value of the function */
    cudaGraphExecUpdateErrorTopologyChanged   = 0x2, /**< The update failed because the topology changed */
    cudaGraphExecUpdateErrorNodeTypeChanged   = 0x3, /**< The update failed because a node type changed */
    cudaGraphExecUpdateErrorFunctionChanged   = 0x4, /**< The update failed because the function of a kernel node changed (CUDA driver < 11.2) */
    cudaGraphExecUpdateErrorParametersChanged = 0x5, /**< The update failed because the parameters changed in a way that is not supported */
    cudaGraphExecUpdateErrorNotSupported      = 0x6, /**< The update failed because something about the node is not supported */
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7, /**< The update failed because the function of a kernel node changed in an unsupported way */
    cudaGraphExecUpdateErrorAttributesChanged = 0x8 /**< The update failed because the node attributes changed in a way that is not supported */
};

/**
 * Flags to specify search options to be used with ::cudaGetDriverEntryPoint
 * For more details see ::cuGetProcAddress
 */ 
enum __device_builtin__ cudaGetDriverEntryPointFlags {
    cudaEnableDefault                = 0x0, /**< Default search mode for driver symbols. */
    cudaEnableLegacyStream           = 0x1, /**< Search for legacy versions of driver symbols. */
    cudaEnablePerThreadDefaultStream = 0x2  /**< Search for per-thread versions of driver symbols. */
};

/**
 * CUDA Graph debug write options
 */
enum __device_builtin__ cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose                  = 1<<0,  /**< Output all debug data as if every debug flag is enabled */
    cudaGraphDebugDotFlagsKernelNodeParams         = 1<<2,  /**< Adds cudaKernelNodeParams to output */
    cudaGraphDebugDotFlagsMemcpyNodeParams         = 1<<3,  /**< Adds cudaMemcpy3DParms to output */
    cudaGraphDebugDotFlagsMemsetNodeParams         = 1<<4,  /**< Adds cudaMemsetParams to output */
    cudaGraphDebugDotFlagsHostNodeParams           = 1<<5,  /**< Adds cudaHostNodeParams to output */
    cudaGraphDebugDotFlagsEventNodeParams          = 1<<6,  /**< Adds cudaEvent_t handle from record and wait nodes to output */
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7,  /**< Adds cudaExternalSemaphoreSignalNodeParams values to output */
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams   = 1<<8,  /**< Adds cudaExternalSemaphoreWaitNodeParams to output */
    cudaGraphDebugDotFlagsKernelNodeAttributes     = 1<<9,  /**< Adds cudaKernelNodeAttrID values to output */
    cudaGraphDebugDotFlagsHandles                  = 1<<10  /**< Adds node handles and every kernel function handle to output */
};

/**
 * Flags for instantiating a graph
 */
enum __device_builtin__ cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1 /**< Automatically free memory allocated in a graph before relaunching. */
};

/** @} */
/** @} */ /* END CUDART_TYPES */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#undef __CUDA_DEPRECATED

#endif /* !__DRIVER_TYPES_H__ */
