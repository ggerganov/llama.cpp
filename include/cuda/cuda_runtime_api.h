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



#if !defined(__CUDA_RUNTIME_API_H__)
#define __CUDA_RUNTIME_API_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_API_H__
#endif

/**
 * \latexonly
 * \page sync_async API synchronization behavior
 *
 * \section memcpy_sync_async_behavior Memcpy
 * The API provides memcpy/memset functions in both synchronous and asynchronous forms,
 * the latter having an \e "Async" suffix. This is a misnomer as each function
 * may exhibit synchronous or asynchronous behavior depending on the arguments
 * passed to the function. In the reference documentation, each memcpy function is
 * categorized as \e synchronous or \e asynchronous, corresponding to the definitions
 * below.
 * 
 * \subsection MemcpySynchronousBehavior Synchronous
 * 
 * <ol>
 * <li> For transfers from pageable host memory to device memory, a stream sync is performed
 * before the copy is initiated. The function will return once the pageable
 * buffer has been copied to the staging memory for DMA transfer to device memory,
 * but the DMA to final destination may not have completed.
 * 
 * <li> For transfers from pinned host memory to device memory, the function is synchronous
 * with respect to the host.
 *
 * <li> For transfers from device to either pageable or pinned host memory, the function returns
 * only once the copy has completed.
 * 
 * <li> For transfers from device memory to device memory, no host-side synchronization is
 * performed.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * </ol>
 * 
 * \subsection MemcpyAsynchronousBehavior Asynchronous
 *
 * <ol>
 * <li> For transfers from device memory to pageable host memory, the function
 * will return only once the copy has completed.
 *
 * <li> For transfers from any host memory to any host memory, the function is fully
 * synchronous with respect to the host.
 * 
 * <li> For all other transfers, the function is fully asynchronous. If pageable
 * memory must first be staged to pinned memory, this will be handled
 * asynchronously with a worker thread.
 * </ol>
 *
 * \section memset_sync_async_behavior Memset
 * The cudaMemset functions are asynchronous with respect to the host
 * except when the target memory is pinned host memory. The \e Async
 * versions are always asynchronous with respect to the host.
 *
 * \section kernel_launch_details Kernel Launches
 * Kernel launches are asynchronous with respect to the host. Details of
 * concurrent kernel execution and data transfers can be found in the CUDA
 * Programmers Guide.
 *
 * \endlatexonly
 */

/**
 * There are two levels for the runtime API.
 *
 * The C API (<i>cuda_runtime_api.h</i>) is
 * a C-style interface that does not require compiling with \p nvcc.
 *
 * The \ref CUDART_HIGHLEVEL "C++ API" (<i>cuda_runtime.h</i>) is a
 * C++-style interface built on top of the C API. It wraps some of the
 * C API routines, using overloading, references and default arguments.
 * These wrappers can be used from C++ code and can be compiled with any C++
 * compiler. The C++ API also has some CUDA-specific wrappers that wrap
 * C API routines that deal with symbols, textures, and device functions.
 * These wrappers require the use of \p nvcc because they depend on code being
 * generated by the compiler. For example, the execution configuration syntax
 * to invoke kernels is only available in source code compiled with \p nvcc.
 */

/** CUDA Runtime API Version */
#define CUDART_VERSION  11060

#if defined(__CUDA_API_VER_MAJOR__) && defined(__CUDA_API_VER_MINOR__)
# define __CUDART_API_VERSION ((__CUDA_API_VER_MAJOR__ * 1000) + (__CUDA_API_VER_MINOR__ * 10))
#else
# define __CUDART_API_VERSION CUDART_VERSION
#endif

#ifndef __DOXYGEN_ONLY__
#include "crt/host_defines.h"
#endif
#include "builtin_types.h"

#include "cuda_device_runtime_api.h"

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) || defined(__CUDA_API_VERSION_INTERNAL)
    #define __CUDART_API_PER_THREAD_DEFAULT_STREAM
    #define __CUDART_API_PTDS(api) api ## _ptds
    #define __CUDART_API_PTSZ(api) api ## _ptsz
#else
    #define __CUDART_API_PTDS(api) api
    #define __CUDART_API_PTSZ(api) api
#endif

#define cudaSignalExternalSemaphoresAsync  __CUDART_API_PTSZ(cudaSignalExternalSemaphoresAsync_v2)
#define cudaWaitExternalSemaphoresAsync    __CUDART_API_PTSZ(cudaWaitExternalSemaphoresAsync_v2)

#if defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)
    #define cudaMemcpy                     __CUDART_API_PTDS(cudaMemcpy)
    #define cudaMemcpyToSymbol             __CUDART_API_PTDS(cudaMemcpyToSymbol)
    #define cudaMemcpyFromSymbol           __CUDART_API_PTDS(cudaMemcpyFromSymbol)
    #define cudaMemcpy2D                   __CUDART_API_PTDS(cudaMemcpy2D)
    #define cudaMemcpyToArray              __CUDART_API_PTDS(cudaMemcpyToArray)
    #define cudaMemcpy2DToArray            __CUDART_API_PTDS(cudaMemcpy2DToArray)
    #define cudaMemcpyFromArray            __CUDART_API_PTDS(cudaMemcpyFromArray)
    #define cudaMemcpy2DFromArray          __CUDART_API_PTDS(cudaMemcpy2DFromArray)
    #define cudaMemcpyArrayToArray         __CUDART_API_PTDS(cudaMemcpyArrayToArray)
    #define cudaMemcpy2DArrayToArray       __CUDART_API_PTDS(cudaMemcpy2DArrayToArray)
    #define cudaMemcpy3D                   __CUDART_API_PTDS(cudaMemcpy3D)
    #define cudaMemcpy3DPeer               __CUDART_API_PTDS(cudaMemcpy3DPeer)
    #define cudaMemset                     __CUDART_API_PTDS(cudaMemset)
    #define cudaMemset2D                   __CUDART_API_PTDS(cudaMemset2D)
    #define cudaMemset3D                   __CUDART_API_PTDS(cudaMemset3D)
    #define cudaGraphUpload                __CUDART_API_PTSZ(cudaGraphUpload)
    #define cudaGraphLaunch                __CUDART_API_PTSZ(cudaGraphLaunch)
    #define cudaStreamBeginCapture         __CUDART_API_PTSZ(cudaStreamBeginCapture)
    #define cudaStreamEndCapture           __CUDART_API_PTSZ(cudaStreamEndCapture)
    #define cudaStreamGetCaptureInfo       __CUDART_API_PTSZ(cudaStreamGetCaptureInfo)
    #define cudaStreamGetCaptureInfo_v2    __CUDART_API_PTSZ(cudaStreamGetCaptureInfo_v2)
    #define cudaStreamIsCapturing          __CUDART_API_PTSZ(cudaStreamIsCapturing)
    #define cudaMemcpyAsync                __CUDART_API_PTSZ(cudaMemcpyAsync)
    #define cudaMemcpyToSymbolAsync        __CUDART_API_PTSZ(cudaMemcpyToSymbolAsync)
    #define cudaMemcpyFromSymbolAsync      __CUDART_API_PTSZ(cudaMemcpyFromSymbolAsync)
    #define cudaMemcpy2DAsync              __CUDART_API_PTSZ(cudaMemcpy2DAsync)
    #define cudaMemcpyToArrayAsync         __CUDART_API_PTSZ(cudaMemcpyToArrayAsync)
    #define cudaMemcpy2DToArrayAsync       __CUDART_API_PTSZ(cudaMemcpy2DToArrayAsync)
    #define cudaMemcpyFromArrayAsync       __CUDART_API_PTSZ(cudaMemcpyFromArrayAsync)
    #define cudaMemcpy2DFromArrayAsync     __CUDART_API_PTSZ(cudaMemcpy2DFromArrayAsync)
    #define cudaMemcpy3DAsync              __CUDART_API_PTSZ(cudaMemcpy3DAsync)
    #define cudaMemcpy3DPeerAsync          __CUDART_API_PTSZ(cudaMemcpy3DPeerAsync)
    #define cudaMemsetAsync                __CUDART_API_PTSZ(cudaMemsetAsync)
    #define cudaMemset2DAsync              __CUDART_API_PTSZ(cudaMemset2DAsync)
    #define cudaMemset3DAsync              __CUDART_API_PTSZ(cudaMemset3DAsync)
    #define cudaStreamQuery                __CUDART_API_PTSZ(cudaStreamQuery)
    #define cudaStreamGetFlags             __CUDART_API_PTSZ(cudaStreamGetFlags)
    #define cudaStreamGetPriority          __CUDART_API_PTSZ(cudaStreamGetPriority)
    #define cudaEventRecord                __CUDART_API_PTSZ(cudaEventRecord)
    #define cudaEventRecordWithFlags       __CUDART_API_PTSZ(cudaEventRecordWithFlags)
    #define cudaStreamWaitEvent            __CUDART_API_PTSZ(cudaStreamWaitEvent)
    #define cudaStreamAddCallback          __CUDART_API_PTSZ(cudaStreamAddCallback)
    #define cudaStreamAttachMemAsync       __CUDART_API_PTSZ(cudaStreamAttachMemAsync)
    #define cudaStreamSynchronize          __CUDART_API_PTSZ(cudaStreamSynchronize)
    #define cudaLaunchKernel               __CUDART_API_PTSZ(cudaLaunchKernel)
    #define cudaLaunchHostFunc             __CUDART_API_PTSZ(cudaLaunchHostFunc)
    #define cudaMemPrefetchAsync           __CUDART_API_PTSZ(cudaMemPrefetchAsync)
    #define cudaLaunchCooperativeKernel    __CUDART_API_PTSZ(cudaLaunchCooperativeKernel)
    #define cudaStreamCopyAttributes       __CUDART_API_PTSZ(cudaStreamCopyAttributes)
    #define cudaStreamGetAttribute         __CUDART_API_PTSZ(cudaStreamGetAttribute)
    #define cudaStreamSetAttribute         __CUDART_API_PTSZ(cudaStreamSetAttribute)
    #define cudaMallocAsync                __CUDART_API_PTSZ(cudaMallocAsync)
    #define cudaFreeAsync                  __CUDART_API_PTSZ(cudaFreeAsync)
    #define cudaMallocFromPoolAsync        __CUDART_API_PTSZ(cudaMallocFromPoolAsync)
    #define cudaGetDriverEntryPoint        __CUDART_API_PTSZ(cudaGetDriverEntryPoint)
#endif

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

#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350))   /** Visible to SM>=3.5 and "__host__ __device__" only **/

#define CUDART_DEVICE __device__ 

#else

#define CUDART_DEVICE

#endif /** CUDART_DEVICE */

#if !defined(__CUDACC_RTC__)
#define EXCLUDE_FROM_RTC

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
 * \defgroup CUDART_DEVICE Device Management
 *
 * ___MANBRIEF___ device management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the device management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Destroy all allocations and reset all state on the current device
 * in the current process.
 *
 * Explicitly destroys and cleans up all resources associated with the current
 * device in the current process. It is the caller's responsibility to ensure
 * that the resources are not accessed or passed in subsequent API calls and
 * doing so will result in undefined behavior. These resources include CUDA types
 * such as ::cudaStream_t, ::cudaEvent_t, ::cudaArray_t, ::cudaMipmappedArray_t,
 * ::cudaTextureObject_t, ::cudaSurfaceObject_t, ::textureReference, ::surfaceReference,
 * ::cudaExternalMemory_t, ::cudaExternalSemaphore_t and ::cudaGraphicsResource_t.
 * Any subsequent API call to this device will reinitialize the device.
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSynchronize
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceReset(void);

/**
 * \brief Wait for compute device to finish
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::cudaDeviceSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::cudaDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::cudaSuccess
 * \note_device_sync_deprecated
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceReset,
 * ::cuCtxSynchronize
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void);

/**
 * \brief Set resource limits
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::cudaDeviceGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::cudaLimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::cudaLimitStackSize controls the stack size in bytes of each GPU thread.
 *
 * - ::cudaLimitPrintfFifoSize controls the size in bytes of the shared FIFO
 *   used by the ::printf() device system call. Setting
 *   ::cudaLimitPrintfFifoSize must not be performed after launching any kernel
 *   that uses the ::printf() device system call - in such case
 *   ::cudaErrorInvalidValue will be returned.
 *
 * - ::cudaLimitMallocHeapSize controls the size in bytes of the heap used by
 *   the ::malloc() and ::free() device system calls. Setting
 *   ::cudaLimitMallocHeapSize must not be performed after launching any kernel
 *   that uses the ::malloc() or ::free() device system calls - in such case
 *   ::cudaErrorInvalidValue will be returned.
 *
 * - ::cudaLimitDevRuntimeSyncDepth controls the maximum nesting depth of a
 *   grid at which a thread can safely call ::cudaDeviceSynchronize(). Setting
 *   this limit must be performed before any launch of a kernel that uses the
 *   device runtime and calls ::cudaDeviceSynchronize() above the default sync
 *   depth, two levels of grids. Calls to ::cudaDeviceSynchronize() will fail
 *   with error code ::cudaErrorSyncDepthExceeded if the limitation is
 *   violated. This limit can be set smaller than the default or up the maximum
 *   launch depth of 24. When setting this limit, keep in mind that additional
 *   levels of sync depth require the runtime to reserve large amounts of
 *   device memory which can no longer be used for user allocations. If these
 *   reservations of device memory fail, ::cudaDeviceSetLimit will return
 *   ::cudaErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::cudaErrorUnsupportedLimit being
 *   returned.
 *
 * - ::cudaLimitDevRuntimePendingLaunchCount controls the maximum number of
 *   outstanding device runtime launches that can be made from the current
 *   device. A grid is outstanding from the point of launch up until the grid
 *   is known to have been completed. Device runtime launches which violate 
 *   this limitation fail and return ::cudaErrorLaunchPendingCountExceeded when
 *   ::cudaGetLastError() is called after launch. If more pending launches than
 *   the default (2048 launches) are needed for a module using the device
 *   runtime, this limit can be increased. Keep in mind that being able to
 *   sustain additional pending launches will require the runtime to reserve
 *   larger amounts of device memory upfront which can no longer be used for
 *   allocations. If these reservations fail, ::cudaDeviceSetLimit will return
 *   ::cudaErrorMemoryAllocation, and the limit can be reset to a lower value.
 *   This limit is only applicable to devices of compute capability 3.5 and
 *   higher. Attempting to set this limit on devices of compute capability less
 *   than 3.5 will result in the error ::cudaErrorUnsupportedLimit being
 *   returned. 
 *
 * - ::cudaLimitMaxL2FetchGranularity controls the L2 cache fetch granularity.
 *   Values can range from 0B to 128B. This is purely a performance hint and
 *   it can be ignored or clamped depending on the platform.
 *
 * - ::cudaLimitPersistingL2CacheSize controls size in bytes available
 *   for persisting L2 cache. This is purely a performance hint and it
 *   can be ignored or clamped depending on the platform.
 *
 * \param limit - Limit to set
 * \param value - Size of limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceGetLimit,
 * ::cuCtxSetLimit
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value);

/**
 * \brief Returns resource limits
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::cudaLimit values are:
 * - ::cudaLimitStackSize: stack size in bytes of each GPU thread;
 * - ::cudaLimitPrintfFifoSize: size in bytes of the shared FIFO used by the
 *   ::printf() device system call.
 * - ::cudaLimitMallocHeapSize: size in bytes of the heap used by the
 *   ::malloc() and ::free() device system calls;
 * - ::cudaLimitDevRuntimeSyncDepth: maximum grid depth at which a
 *   thread can isssue the device runtime call ::cudaDeviceSynchronize()
 *   to wait on child grid launches to complete.
 * - ::cudaLimitDevRuntimePendingLaunchCount: maximum number of outstanding
 *   device runtime launches.
 * - ::cudaLimitMaxL2FetchGranularity: L2 cache fetch granularity.
 * - ::cudaLimitPersistingL2CacheSize: Persisting L2 cache size in bytes
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size of the limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceSetLimit,
 * ::cuCtxGetLimit
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);

/**
 * \brief Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.
 *
 * Returns in \p maxWidthInElements the maximum number of elements allocatable in a 1D linear texture
 * for given format descriptor \p fmtDesc.
 *
 * \param maxWidthInElements    - Returns maximum number of texture elements allocatable for given \p fmtDesc.
 * \param fmtDesc               - Texture format description.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuDeviceGetMaxTexture1DLinear,
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device);
#endif

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::cudaFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSetCacheConfig,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * ::cuCtxGetCacheConfig
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);

/**
 * \brief Returns numerical values that correspond to the least and
 * greatest stream priorities.
 *
 * Returns in \p *leastPriority and \p *greatestPriority the numerical values that correspond
 * to the least and greatest stream priorities respectively. Stream priorities
 * follow a convention where lower numbers imply greater priorities. The range of
 * meaningful stream priorities is given by [\p *greatestPriority, \p *leastPriority].
 * If the user attempts to create a stream with a priority value that is
 * outside the the meaningful range as specified by this API, the priority is
 * automatically clamped down or up to either \p *leastPriority or \p *greatestPriority
 * respectively. See ::cudaStreamCreateWithPriority for details on creating a
 * priority stream.
 * A NULL may be passed in for \p *leastPriority or \p *greatestPriority if the value
 * is not desired.
 *
 * This function will return '0' in both \p *leastPriority and \p *greatestPriority if
 * the current context's device does not support stream priorities
 * (see ::cudaDeviceGetAttribute).
 *
 * \param leastPriority    - Pointer to an int in which the numerical value for least
 *                           stream priority is returned
 * \param greatestPriority - Pointer to an int in which the numerical value for greatest
 *                           stream priority is returned
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaStreamGetPriority,
 * ::cuCtxGetStreamPriorityRange
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)"
 * or
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::cudaFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceGetCacheConfig,
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * ::cuCtxSetCacheConfig
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);

/**
 * \brief Returns the shared memory configuration for the current device.
 *
 * This function will return in \p pConfig the current size of shared memory banks
 * on the current device. On devices with configurable shared memory banks, 
 * ::cudaDeviceSetSharedMemConfig can be used to change this setting, so that all 
 * subsequent kernel launches will by default use the new bank size. When 
 * ::cudaDeviceGetSharedMemConfig is called on devices without configurable shared 
 * memory, it will return the fixed bank size of the hardware.
 *
 * The returned bank configurations can be either:
 * - ::cudaSharedMemBankSizeFourByte - shared memory bank width is four bytes.
 * - ::cudaSharedMemBankSizeEightByte - shared memory bank width is eight bytes.
 *
 * \param pConfig - Returned cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSetCacheConfig,
 * ::cudaDeviceGetCacheConfig,
 * ::cudaDeviceSetSharedMemConfig,
 * ::cudaFuncSetCacheConfig,
 * ::cuCtxGetSharedMemConfig
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);

/**
 * \brief Sets the shared memory configuration for the current device.
 *
 * On devices with configurable shared memory banks, this function will set
 * the shared memory bank size which is used for all subsequent kernel launches.
 * Any per-function setting of shared memory set via ::cudaFuncSetSharedMemConfig
 * will override the device wide setting.
 *
 * Changing the shared memory configuration between launches may introduce
 * a device side synchronization point.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * The supported bank configurations are:
 * - ::cudaSharedMemBankSizeDefault: set bank width the device default (currently,
 *   four bytes)
 * - ::cudaSharedMemBankSizeFourByte: set shared memory bank width to be four bytes
 *   natively.
 * - ::cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively.
 *
 * \param config - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSetCacheConfig,
 * ::cudaDeviceGetCacheConfig,
 * ::cudaDeviceGetSharedMemConfig,
 * ::cudaFuncSetCacheConfig,
 * ::cuCtxSetSharedMemConfig
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);

/**
 * \brief Returns a handle to a compute device
 *
 * Returns in \p *device a device ordinal given a PCI bus ID string.
 *
 * \param device   - Returned device ordinal
 *
 * \param pciBusId - String in one of the following forms: 
 * [domain]:[bus]:[device].[function]
 * [domain]:[bus]:[device]
 * [bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceGetPCIBusId,
 * ::cuDeviceGetByPCIBusId
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);

/**
 * \brief Returns a PCI Bus Id string for the device
 *
 * Returns an ASCII string identifying the device \p dev in the NULL-terminated
 * string pointed to by \p pciBusId. \p len specifies the maximum length of the
 * string that may be returned.
 *
 * \param pciBusId - Returned identifier string for the device in the following format
 * [domain]:[bus]:[device].[function]
 * where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
 * pciBusId should be large enough to store 13 characters including the NULL-terminator.
 *
 * \param len      - Maximum length of string to store in \p name
 *
 * \param device   - Device to get identifier string for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceGetByPCIBusId,
 * ::cuDeviceGetPCIBusId
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);

/**
 * \brief Gets an interprocess handle for a previously allocated event
 *
 * Takes as input a previously allocated event. This event must have been 
 * created with the ::cudaEventInterprocess and ::cudaEventDisableTiming
 * flags set. This opaque handle may be copied into other processes and
 * opened with ::cudaIpcOpenEventHandle to allow efficient hardware
 * synchronization between GPU work in different processes.
 *
 * After the event has been been opened in the importing process, 
 * ::cudaEventRecord, ::cudaEventSynchronize, ::cudaStreamWaitEvent and 
 * ::cudaEventQuery may be used in either process. Performing operations 
 * on the imported event after the exported event has been freed 
 * with ::cudaEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on Tegra platforms.
 *
 * \param handle - Pointer to a user allocated cudaIpcEventHandle
 *                    in which to return the opaque event handle
 * \param event   - Event allocated with ::cudaEventInterprocess and 
 *                    ::cudaEventDisableTiming flags.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaEventCreate,
 * ::cudaEventDestroy,
 * ::cudaEventSynchronize,
 * ::cudaEventQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle,
 * ::cuIpcGetEventHandle
 */
extern __host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);

/**
 * \brief Opens an interprocess event handle for use in the current process
 *
 * Opens an interprocess event handle exported from another process with 
 * ::cudaIpcGetEventHandle. This function returns a ::cudaEvent_t that behaves like 
 * a locally created event with the ::cudaEventDisableTiming flag specified. 
 * This event must be freed with ::cudaEventDestroy.
 *
 * Performing operations on the imported event after the exported event has 
 * been freed with ::cudaEventDestroy will result in undefined behavior.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on Tegra platforms.
 *
 * \param event - Returns the imported event
 * \param handle  - Interprocess handle to open
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorDeviceUninitialized
 * \note_init_rt
 * \note_callback
 *
  * \sa
 * ::cudaEventCreate,
 * ::cudaEventDestroy,
 * ::cudaEventSynchronize,
 * ::cudaEventQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle,
 * ::cuIpcOpenEventHandle
 */
extern __host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);


/**
 * \brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created 
 * with ::cudaMalloc and exports it for use in another process. This is a 
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects. 
 *
 * If a region of memory is freed with ::cudaFree and a subsequent call
 * to ::cudaMalloc returns memory with the same device address,
 * ::cudaIpcGetMemHandle will return a unique handle for the
 * new memory. 
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on Tegra platforms.
 *
 * \param handle - Pointer to user allocated ::cudaIpcMemHandle to return
 *                    the handle in.
 * \param devPtr - Base pointer to previously allocated device memory 
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMalloc,
 * ::cudaFree,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cudaIpcCloseMemHandle,
 * ::cuIpcGetMemHandle
 */
extern __host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);

/**
 * \brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with ::cudaIpcGetMemHandle into
 * the current device address space. For contexts on different devices 
 * ::cudaIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called ::cudaDeviceEnablePeerAccess. This behavior is 
 * controlled by the ::cudaIpcMemLazyEnablePeerAccess flag. 
 * ::cudaDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * ::cudaIpcOpenMemHandle can open handles to devices that may not be visible
 * in the process calling the API.
 *
 * Contexts that may open ::cudaIpcMemHandles are restricted in the following way.
 * ::cudaIpcMemHandles from each device in a given process may only be opened 
 * by one context per device per other process.
 *
 * If the memory handle has already been opened by the current context, the
 * reference count on the handle is incremented by 1 and the existing device pointer
 * is returned.
 *
 * Memory returned from ::cudaIpcOpenMemHandle must be freed with
 * ::cudaIpcCloseMemHandle.
 *
 * Calling ::cudaFree on an exported memory region before calling
 * ::cudaIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 * 
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on Tegra platforms.
 *
 * \param devPtr - Returned device pointer
 * \param handle - ::cudaIpcMemHandle to open
 * \param flags  - Flags for this operation. Must be specified as ::cudaIpcMemLazyEnablePeerAccess
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorDeviceUninitialized,
 * ::cudaErrorTooManyPeers,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \note No guarantees are made about the address returned in \p *devPtr.  
 * In particular, multiple processes may not receive the same address for the same \p handle.
 *
 * \sa
 * ::cudaMalloc,
 * ::cudaFree,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcCloseMemHandle,
 * ::cudaDeviceEnablePeerAccess,
 * ::cudaDeviceCanAccessPeer,
 * ::cuIpcOpenMemHandle
 */
extern __host__ cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags);

/**
 * \brief Attempts to close memory mapped with cudaIpcOpenMemHandle
 * 
 * Decrements the reference count of the memory returnd by ::cudaIpcOpenMemHandle by 1.
 * When the reference count reaches 0, this API unmaps the memory. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * IPC functionality is restricted to devices with support for unified 
 * addressing on Linux operating systems. IPC functionality is not supported
 * on Tegra platforms.
 *
 * \param devPtr - Device pointer returned by ::cudaIpcOpenMemHandle
 * 
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorMapBufferObjectFailed,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMalloc,
 * ::cudaFree,
 * ::cudaIpcGetEventHandle,
 * ::cudaIpcOpenEventHandle,
 * ::cudaIpcGetMemHandle,
 * ::cudaIpcOpenMemHandle,
 * ::cuIpcCloseMemHandle
 */
extern __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr);

/**
 * \brief Blocks until remote writes are visible to the specified scope
 *
 * Blocks until remote writes to the target context via mappings created
 * through GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see
 * https://docs.nvidia.com/cuda/gpudirect-rdma for more information), are
 * visible to the specified scope.
 *
 * If the scope equals or lies within the scope indicated by
 * ::cudaDevAttrGPUDirectRDMAWritesOrdering, the call will be a no-op and
 * can be safely omitted for performance. This can be determined by
 * comparing the numerical values between the two enums, with smaller
 * scopes having smaller values.
 *
 * Users may query support for this API via ::cudaDevAttrGPUDirectRDMAFlushWritesOptions.
 *
 * \param target - The target of the operation, see cudaFlushGPUDirectRDMAWritesTarget
 * \param scope  - The scope of the operation, see cudaFlushGPUDirectRDMAWritesScope
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotSupported,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuFlushGPUDirectRDMAWrites
 */
#if __CUDART_API_VERSION >= 11030
extern __host__ cudaError_t CUDARTAPI cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope);
#endif

/** @} */ /* END CUDART_DEVICE */


/**
 * \defgroup CUDART_THREAD_DEPRECATED Thread Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated thread management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated thread management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Exit and clean up from CUDA launches
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceReset(), which should be used
 * instead.
 *
 * Explicitly destroys all cleans up all resources associated with the current
 * device in the current process.  Any subsequent API call to this device will 
 * reinitialize the device.  
 *
 * Note that this function will reset the device immediately.  It is the caller's
 * responsibility to ensure that the device is not being accessed by any 
 * other host threads from the process when this function is called.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceReset
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadExit(void);

/**
 * \brief Wait for compute device to finish
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is similar to the 
 * non-deprecated function ::cudaDeviceSynchronize(), which should be used
 * instead.
 *
 * Blocks until the device has completed all preceding requested tasks.
 * ::cudaThreadSynchronize() returns an error if one of the preceding tasks
 * has failed. If the ::cudaDeviceScheduleBlockingSync flag was set for 
 * this device, the host thread will block until the device has finished 
 * its work.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSynchronize
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void);

/**
 * \brief Set resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceSetLimit(), which should be used
 * instead.
 *
 * Setting \p limit to \p value is a request by the application to update
 * the current limit maintained by the device.  The driver is free to
 * modify the requested value to meet h/w requirements (this could be
 * clamping to minimum or maximum values, rounding up to nearest element
 * size, etc).  The application can use ::cudaThreadGetLimit() to find out
 * exactly what the limit has been set to.
 *
 * Setting each ::cudaLimit has its own specific restrictions, so each is
 * discussed here.
 *
 * - ::cudaLimitStackSize controls the stack size of each GPU thread.
 *
 * - ::cudaLimitPrintfFifoSize controls the size of the shared FIFO
 *   used by the ::printf() device system call.
 *   Setting ::cudaLimitPrintfFifoSize must be performed before
 *   launching any kernel that uses the ::printf() device
 *   system call, otherwise ::cudaErrorInvalidValue will be returned.
 *
 * - ::cudaLimitMallocHeapSize controls the size of the heap used
 *   by the ::malloc() and ::free() device system calls.  Setting
 *   ::cudaLimitMallocHeapSize must be performed before launching
 *   any kernel that uses the ::malloc() or ::free() device system calls,
 *   otherwise ::cudaErrorInvalidValue will be returned.
 *
 * \param limit - Limit to set
 * \param value - Size in bytes of limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSetLimit
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSetLimit(enum cudaLimit limit, size_t value);

/**
 * \brief Returns resource limits
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceGetLimit(), which should be used
 * instead.
 *
 * Returns in \p *pValue the current size of \p limit.  The supported
 * ::cudaLimit values are:
 * - ::cudaLimitStackSize: stack size of each GPU thread;
 * - ::cudaLimitPrintfFifoSize: size of the shared FIFO used by the
 *   ::printf() device system call.
 * - ::cudaLimitMallocHeapSize: size of the heap used by the
 *   ::malloc() and ::free() device system calls;
 *
 * \param limit  - Limit to query
 * \param pValue - Returned size in bytes of limit
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorUnsupportedLimit,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceGetLimit
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit);

/**
 * \brief Returns the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceGetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this returns through \p pCacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute functions.
 *
 * This will return a \p pCacheConfig of ::cudaFuncCachePreferNone on devices
 * where the size of the L1 cache and shared memory are fixed.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param pCacheConfig - Returned cache configuration
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceGetCacheConfig
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig);

/**
 * \brief Sets the preferred cache configuration for the current device.
 *
 * \deprecated
 *
 * Note that this function is deprecated because its name does not 
 * reflect its behavior.  Its functionality is identical to the 
 * non-deprecated function ::cudaDeviceSetCacheConfig(), which should be 
 * used instead.
 * 
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache
 * configuration for the current device. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute the function. Any
 * function preference set via
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)"
 * or
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)"
 * will be preferred over this device-wide setting. Setting the device-wide
 * cache configuration to ::cudaFuncCachePreferNone will cause subsequent
 * kernel launches to prefer to not change the cache configuration unless
 * required to launch the kernel.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSetCacheConfig
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);

/** @} */ /* END CUDART_THREAD_DEPRECATED */



/**
 * \defgroup CUDART_ERROR Error Handling
 *
 * ___MANBRIEF___ error handling functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the error handling functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same host thread and resets it to ::cudaSuccess.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMissingConfiguration,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorInitializationError,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorUnmapBufferObjectFailed,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture,
 * ::cudaErrorInvalidTextureBinding,
 * ::cudaErrorInvalidChannelDescriptor,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorInvalidFilterSetting,
 * ::cudaErrorInvalidNormSetting,
 * ::cudaErrorUnknown,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInsufficientDriver,
 * ::cudaErrorNoDevice,
 * ::cudaErrorSetOnActiveProcess,
 * ::cudaErrorStartupFailure,
 * ::cudaErrorInvalidPtx,
 * ::cudaErrorUnsupportedPtxVersion,
 * ::cudaErrorNoKernelImageForDevice,
 * ::cudaErrorJitCompilerNotFound,
 * ::cudaErrorJitCompilationDisabled
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaPeekAtLastError, ::cudaGetErrorName, ::cudaGetErrorString, ::cudaError
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void);

/**
 * \brief Returns the last error from a runtime call
 *
 * Returns the last error that has been produced by any of the runtime calls
 * in the same host thread. Note that this call does not reset the error to
 * ::cudaSuccess like ::cudaGetLastError().
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMissingConfiguration,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorInitializationError,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorUnmapBufferObjectFailed,
 * ::cudaErrorInvalidDevicePointer,
 * ::cudaErrorInvalidTexture,
 * ::cudaErrorInvalidTextureBinding,
 * ::cudaErrorInvalidChannelDescriptor,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorInvalidFilterSetting,
 * ::cudaErrorInvalidNormSetting,
 * ::cudaErrorUnknown,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInsufficientDriver,
 * ::cudaErrorNoDevice,
 * ::cudaErrorSetOnActiveProcess,
 * ::cudaErrorStartupFailure,
 * ::cudaErrorInvalidPtx,
 * ::cudaErrorUnsupportedPtxVersion,
 * ::cudaErrorNoKernelImageForDevice,
 * ::cudaErrorJitCompilerNotFound,
 * ::cudaErrorJitCompilationDisabled
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetLastError, ::cudaGetErrorName, ::cudaGetErrorString, ::cudaError
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void);

/**
 * \brief Returns the string representation of an error code enum name
 *
 * Returns a string containing the name of an error code in the enum.  If the error
 * code is not recognized, "unrecognized error code" is returned.
 *
 * \param error - Error code to convert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::cudaGetErrorString, ::cudaGetLastError, ::cudaPeekAtLastError, ::cudaError,
 * ::cuGetErrorName
 */
extern __host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error);

/**
 * \brief Returns the description string for an error code
 *
 * Returns the description string for an error code.  If the error
 * code is not recognized, "unrecognized error code" is returned.
 *
 * \param error - Error code to convert to string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::cudaGetErrorName, ::cudaGetLastError, ::cudaPeekAtLastError, ::cudaError,
 * ::cuGetErrorString
 */
extern __host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error);
/** @} */ /* END CUDART_ERROR */

/**
 * \addtogroup CUDART_DEVICE 
 *
 * @{
 */

/**
 * \brief Returns the number of compute-capable devices
 *
 * Returns in \p *count the number of devices with compute capability greater
 * or equal to 2.0 that are available for execution.
 *
 * \param count - Returns the number of devices with compute capability
 * greater or equal to 2.0
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDevice, ::cudaSetDevice, ::cudaGetDeviceProperties,
 * ::cudaChooseDevice,
 * ::cuDeviceGetCount
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count);

/**
 * \brief Returns information about the compute-device
 *
 * Returns in \p *prop the properties of device \p dev. The ::cudaDeviceProp
 * structure is defined as:
 * \code
    struct cudaDeviceProp {
        char name[256];
        cudaUUID_t uuid;
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int persistingL2CacheMaxSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemory;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
        int singleToDoublePrecisionPerfRatio;
        int pageableMemoryAccess;
        int concurrentManagedAccess;
        int computePreemptionSupported;
        int canUseHostPointerForRegisteredMem;
        int cooperativeLaunch;
        int cooperativeMultiDeviceLaunch;
        int pageableMemoryAccessUsesHostPageTables;
        int directManagedMemAccessFromHost;
        int accessPolicyMaxWindowSize;
    }
 \endcode
 * where:
 * - \ref ::cudaDeviceProp::name "name[256]" is an ASCII string identifying
 *   the device;
 * - \ref ::cudaDeviceProp::uuid "uuid" is a 16-byte unique identifier.
 * - \ref ::cudaDeviceProp::totalGlobalMem "totalGlobalMem" is the total
 *   amount of global memory available on the device in bytes;
 * - \ref ::cudaDeviceProp::sharedMemPerBlock "sharedMemPerBlock" is the
 *   maximum amount of shared memory available to a thread block in bytes;
 * - \ref ::cudaDeviceProp::regsPerBlock "regsPerBlock" is the maximum number
 *   of 32-bit registers available to a thread block;
 * - \ref ::cudaDeviceProp::warpSize "warpSize" is the warp size in threads;
 * - \ref ::cudaDeviceProp::memPitch "memPitch" is the maximum pitch in
 *   bytes allowed by the memory copy functions that involve memory regions
 *   allocated through ::cudaMallocPitch();
 * - \ref ::cudaDeviceProp::maxThreadsPerBlock "maxThreadsPerBlock" is the
 *   maximum number of threads per block;
 * - \ref ::cudaDeviceProp::maxThreadsDim "maxThreadsDim[3]" contains the
 *   maximum size of each dimension of a block;
 * - \ref ::cudaDeviceProp::maxGridSize "maxGridSize[3]" contains the
 *   maximum size of each dimension of a grid;
 * - \ref ::cudaDeviceProp::clockRate "clockRate" is the clock frequency in
 *   kilohertz;
 * - \ref ::cudaDeviceProp::totalConstMem "totalConstMem" is the total amount
 *   of constant memory available on the device in bytes;
 * - \ref ::cudaDeviceProp::major "major",
 *   \ref ::cudaDeviceProp::minor "minor" are the major and minor revision
 *   numbers defining the device's compute capability;
 * - \ref ::cudaDeviceProp::textureAlignment "textureAlignment" is the
 *   alignment requirement; texture base addresses that are aligned to
 *   \ref ::cudaDeviceProp::textureAlignment "textureAlignment" bytes do not
 *   need an offset applied to texture fetches;
 * - \ref ::cudaDeviceProp::texturePitchAlignment "texturePitchAlignment" is the
 *   pitch alignment requirement for 2D texture references that are bound to 
 *   pitched memory;
 * - \ref ::cudaDeviceProp::deviceOverlap "deviceOverlap" is 1 if the device
 *   can concurrently copy memory between host and device while executing a
 *   kernel, or 0 if not.  Deprecated, use instead asyncEngineCount.
 * - \ref ::cudaDeviceProp::multiProcessorCount "multiProcessorCount" is the
 *   number of multiprocessors on the device;
 * - \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
 *   is 1 if there is a run time limit for kernels executed on the device, or
 *   0 if not.
 * - \ref ::cudaDeviceProp::integrated "integrated" is 1 if the device is an
 *   integrated (motherboard) GPU and 0 if it is a discrete (card) component.
 * - \ref ::cudaDeviceProp::canMapHostMemory "canMapHostMemory" is 1 if the
 *   device can map host memory into the CUDA address space for use with
 *   ::cudaHostAlloc()/::cudaHostGetDevicePointer(), or 0 if not;
 * - \ref ::cudaDeviceProp::computeMode "computeMode" is the compute mode
 *   that the device is currently in. Available modes are as follows:
 *   - cudaComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::cudaSetDevice() with this device.
 *   - cudaComputeModeExclusive: Compute-exclusive mode - Only one thread will
 *     be able to use ::cudaSetDevice() with this device.
 *   - cudaComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::cudaSetDevice() with this device.
 *   - cudaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::cudaSetDevice() with this device.
 *   <br> If ::cudaSetDevice() is called on an already occupied \p device with 
 *   computeMode ::cudaComputeModeExclusive, ::cudaErrorDeviceAlreadyInUse
 *   will be immediately returned indicating the device cannot be used.
 *   When an occupied exclusive mode device is chosen with ::cudaSetDevice,
 *   all subsequent non-device management runtime functions will return
 *   ::cudaErrorDevicesUnavailable.
 * - \ref ::cudaDeviceProp::maxTexture1D "maxTexture1D" is the maximum 1D
 *   texture size.
 * - \ref ::cudaDeviceProp::maxTexture1DMipmap "maxTexture1DMipmap" is the maximum
 *   1D mipmapped texture texture size.
 * - \ref ::cudaDeviceProp::maxTexture1DLinear "maxTexture1DLinear" is the maximum
 *   1D texture size for textures bound to linear memory.
 * - \ref ::cudaDeviceProp::maxTexture2D "maxTexture2D[2]" contains the maximum
 *   2D texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture2DMipmap "maxTexture2DMipmap[2]" contains the
 *   maximum 2D mipmapped texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture2DLinear "maxTexture2DLinear[3]" contains the 
 *   maximum 2D texture dimensions for 2D textures bound to pitch linear memory.
 * - \ref ::cudaDeviceProp::maxTexture2DGather "maxTexture2DGather[2]" contains the 
 *   maximum 2D texture dimensions if texture gather operations have to be performed.
 * - \ref ::cudaDeviceProp::maxTexture3D "maxTexture3D[3]" contains the maximum
 *   3D texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture3DAlt "maxTexture3DAlt[3]"
 *   contains the maximum alternate 3D texture dimensions.
 * - \ref ::cudaDeviceProp::maxTextureCubemap "maxTextureCubemap" is the 
 *   maximum cubemap texture width or height.
 * - \ref ::cudaDeviceProp::maxTexture1DLayered "maxTexture1DLayered[2]" contains
 *   the maximum 1D layered texture dimensions.
 * - \ref ::cudaDeviceProp::maxTexture2DLayered "maxTexture2DLayered[3]" contains
 *   the maximum 2D layered texture dimensions.
 * - \ref ::cudaDeviceProp::maxTextureCubemapLayered "maxTextureCubemapLayered[2]"
 *   contains the maximum cubemap layered texture dimensions.
 * - \ref ::cudaDeviceProp::maxSurface1D "maxSurface1D" is the maximum 1D
 *   surface size.
 * - \ref ::cudaDeviceProp::maxSurface2D "maxSurface2D[2]" contains the maximum
 *   2D surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurface3D "maxSurface3D[3]" contains the maximum
 *   3D surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurface1DLayered "maxSurface1DLayered[2]" contains
 *   the maximum 1D layered surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurface2DLayered "maxSurface2DLayered[3]" contains
 *   the maximum 2D layered surface dimensions.
 * - \ref ::cudaDeviceProp::maxSurfaceCubemap "maxSurfaceCubemap" is the maximum 
 *   cubemap surface width or height.
 * - \ref ::cudaDeviceProp::maxSurfaceCubemapLayered "maxSurfaceCubemapLayered[2]"
 *   contains the maximum cubemap layered surface dimensions.
 * - \ref ::cudaDeviceProp::surfaceAlignment "surfaceAlignment" specifies the
 *   alignment requirements for surfaces.
 * - \ref ::cudaDeviceProp::concurrentKernels "concurrentKernels" is 1 if the
 *   device supports executing multiple kernels within the same context
 *   simultaneously, or 0 if not. It is not guaranteed that multiple kernels
 *   will be resident on the device concurrently so this feature should not be
 *   relied upon for correctness;
 * - \ref ::cudaDeviceProp::ECCEnabled "ECCEnabled" is 1 if the device has ECC
 *   support turned on, or 0 if not.
 * - \ref ::cudaDeviceProp::pciBusID "pciBusID" is the PCI bus identifier of
 *   the device.
 * - \ref ::cudaDeviceProp::pciDeviceID "pciDeviceID" is the PCI device
 *   (sometimes called slot) identifier of the device.
 * - \ref ::cudaDeviceProp::pciDomainID "pciDomainID" is the PCI domain identifier
 *   of the device.
 * - \ref ::cudaDeviceProp::tccDriver "tccDriver" is 1 if the device is using a
 *   TCC driver or 0 if not.
 * - \ref ::cudaDeviceProp::asyncEngineCount "asyncEngineCount" is 1 when the
 *   device can concurrently copy memory between host and device while executing
 *   a kernel. It is 2 when the device can concurrently copy memory between host
 *   and device in both directions and execute a kernel at the same time. It is
 *   0 if neither of these is supported.
 * - \ref ::cudaDeviceProp::unifiedAddressing "unifiedAddressing" is 1 if the device 
 *   shares a unified address space with the host and 0 otherwise.
 * - \ref ::cudaDeviceProp::memoryClockRate "memoryClockRate" is the peak memory 
 *   clock frequency in kilohertz.
 * - \ref ::cudaDeviceProp::memoryBusWidth "memoryBusWidth" is the memory bus width  
 *   in bits.
 * - \ref ::cudaDeviceProp::l2CacheSize "l2CacheSize" is L2 cache size in bytes. 
 * - \ref ::cudaDeviceProp::persistingL2CacheMaxSize "persistingL2CacheMaxSize" is L2 cache's maximum persisting lines size in bytes.
 * - \ref ::cudaDeviceProp::maxThreadsPerMultiProcessor "maxThreadsPerMultiProcessor"  
 *   is the number of maximum resident threads per multiprocessor.
 * - \ref ::cudaDeviceProp::streamPrioritiesSupported "streamPrioritiesSupported"
 *   is 1 if the device supports stream priorities, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::globalL1CacheSupported "globalL1CacheSupported"
 *   is 1 if the device supports caching of globals in L1 cache, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::localL1CacheSupported "localL1CacheSupported"
 *   is 1 if the device supports caching of locals in L1 cache, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::sharedMemPerMultiprocessor "sharedMemPerMultiprocessor" is the
 *   maximum amount of shared memory available to a multiprocessor in bytes; this amount is
 *   shared by all thread blocks simultaneously resident on a multiprocessor;
 * - \ref ::cudaDeviceProp::regsPerMultiprocessor "regsPerMultiprocessor" is the maximum number
 *   of 32-bit registers available to a multiprocessor; this number is shared
 *   by all thread blocks simultaneously resident on a multiprocessor;
 * - \ref ::cudaDeviceProp::managedMemory "managedMemory"
 *   is 1 if the device supports allocating managed memory on this system, or 0 if it is not supported.
 * - \ref ::cudaDeviceProp::isMultiGpuBoard "isMultiGpuBoard"
 *   is 1 if the device is on a multi-GPU board (e.g. Gemini cards), and 0 if not;
 * - \ref ::cudaDeviceProp::multiGpuBoardGroupID "multiGpuBoardGroupID" is a unique identifier
 *   for a group of devices associated with the same board.
 *   Devices on the same multi-GPU board will share the same identifier;
 * - \ref ::cudaDeviceProp::singleToDoublePrecisionPerfRatio "singleToDoublePrecisionPerfRatio"  
 *   is the ratio of single precision performance (in floating-point operations per second)
 *   to double precision performance.
 * - \ref ::cudaDeviceProp::pageableMemoryAccess "pageableMemoryAccess" is 1 if the device supports
 *   coherently accessing pageable memory without calling cudaHostRegister on it, and 0 otherwise.
 * - \ref ::cudaDeviceProp::concurrentManagedAccess "concurrentManagedAccess" is 1 if the device can
 *   coherently access managed memory concurrently with the CPU, and 0 otherwise.
 * - \ref ::cudaDeviceProp::computePreemptionSupported "computePreemptionSupported" is 1 if the device
 *   supports Compute Preemption, and 0 otherwise.
 * - \ref ::cudaDeviceProp::canUseHostPointerForRegisteredMem "canUseHostPointerForRegisteredMem" is 1 if
 *   the device can access host registered memory at the same virtual address as the CPU, and 0 otherwise.
 * - \ref ::cudaDeviceProp::cooperativeLaunch "cooperativeLaunch" is 1 if the device supports launching
 *   cooperative kernels via ::cudaLaunchCooperativeKernel, and 0 otherwise.
 * - \ref ::cudaDeviceProp::cooperativeMultiDeviceLaunch "cooperativeMultiDeviceLaunch" is 1 if the device
 *   supports launching cooperative kernels via ::cudaLaunchCooperativeKernelMultiDevice, and 0 otherwise.
 * - \ref ::cudaDeviceProp::pageableMemoryAccessUsesHostPageTables "pageableMemoryAccessUsesHostPageTables" is 1 if the device accesses
 *   pageable memory via the host's page tables, and 0 otherwise.
 * - \ref ::cudaDeviceProp::directManagedMemAccessFromHost "directManagedMemAccessFromHost" is 1 if the host can directly access managed
 *   memory on the device without migration, and 0 otherwise.
 * - \ref ::cudaDeviceProp::maxBlocksPerMultiProcessor "maxBlocksPerMultiProcessor" is the maximum number of thread blocks
 *   that can reside on a multiprocessor.
 * - \ref ::cudaDeviceProp::accessPolicyMaxWindowSize "accessPolicyMaxWindowSize" is
 *   the maximum value of ::cudaAccessPolicyWindow::num_bytes.
 *
 * \param prop   - Properties for the specified device
 * \param device - Device number to get properties for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice, ::cudaChooseDevice,
 * ::cudaDeviceGetAttribute,
 * ::cuDeviceGetAttribute,
 * ::cuDeviceGetName
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);

/**
 * \brief Returns information about the device
 *
 * Returns in \p *value the integer value of the attribute \p attr on device
 * \p device. The supported attributes are:
 * - ::cudaDevAttrMaxThreadsPerBlock: Maximum number of threads per block
 * - ::cudaDevAttrMaxBlockDimX: Maximum x-dimension of a block
 * - ::cudaDevAttrMaxBlockDimY: Maximum y-dimension of a block
 * - ::cudaDevAttrMaxBlockDimZ: Maximum z-dimension of a block
 * - ::cudaDevAttrMaxGridDimX: Maximum x-dimension of a grid
 * - ::cudaDevAttrMaxGridDimY: Maximum y-dimension of a grid
 * - ::cudaDevAttrMaxGridDimZ: Maximum z-dimension of a grid
 * - ::cudaDevAttrMaxSharedMemoryPerBlock: Maximum amount of shared memory
 *   available to a thread block in bytes
 * - ::cudaDevAttrTotalConstantMemory: Memory available on device for
 *   __constant__ variables in a CUDA C kernel in bytes
 * - ::cudaDevAttrWarpSize: Warp size in threads
 * - ::cudaDevAttrMaxPitch: Maximum pitch in bytes allowed by the memory copy
 *   functions that involve memory regions allocated through ::cudaMallocPitch()
 * - ::cudaDevAttrMaxTexture1DWidth: Maximum 1D texture width
 * - ::cudaDevAttrMaxTexture1DLinearWidth: Maximum width for a 1D texture bound
 *   to linear memory
 * - ::cudaDevAttrMaxTexture1DMipmappedWidth: Maximum mipmapped 1D texture width
 * - ::cudaDevAttrMaxTexture2DWidth: Maximum 2D texture width
 * - ::cudaDevAttrMaxTexture2DHeight: Maximum 2D texture height
 * - ::cudaDevAttrMaxTexture2DLinearWidth: Maximum width for a 2D texture
 *   bound to linear memory
 * - ::cudaDevAttrMaxTexture2DLinearHeight: Maximum height for a 2D texture
 *   bound to linear memory
 * - ::cudaDevAttrMaxTexture2DLinearPitch: Maximum pitch in bytes for a 2D
 *   texture bound to linear memory
 * - ::cudaDevAttrMaxTexture2DMipmappedWidth: Maximum mipmapped 2D texture
 *   width
 * - ::cudaDevAttrMaxTexture2DMipmappedHeight: Maximum mipmapped 2D texture
 *   height
 * - ::cudaDevAttrMaxTexture3DWidth: Maximum 3D texture width
 * - ::cudaDevAttrMaxTexture3DHeight: Maximum 3D texture height
 * - ::cudaDevAttrMaxTexture3DDepth: Maximum 3D texture depth
 * - ::cudaDevAttrMaxTexture3DWidthAlt: Alternate maximum 3D texture width,
 *   0 if no alternate maximum 3D texture size is supported
 * - ::cudaDevAttrMaxTexture3DHeightAlt: Alternate maximum 3D texture height,
 *   0 if no alternate maximum 3D texture size is supported
 * - ::cudaDevAttrMaxTexture3DDepthAlt: Alternate maximum 3D texture depth,
 *   0 if no alternate maximum 3D texture size is supported
 * - ::cudaDevAttrMaxTextureCubemapWidth: Maximum cubemap texture width or
 *   height
 * - ::cudaDevAttrMaxTexture1DLayeredWidth: Maximum 1D layered texture width
 * - ::cudaDevAttrMaxTexture1DLayeredLayers: Maximum layers in a 1D layered
 *   texture
 * - ::cudaDevAttrMaxTexture2DLayeredWidth: Maximum 2D layered texture width
 * - ::cudaDevAttrMaxTexture2DLayeredHeight: Maximum 2D layered texture height
 * - ::cudaDevAttrMaxTexture2DLayeredLayers: Maximum layers in a 2D layered
 *   texture
 * - ::cudaDevAttrMaxTextureCubemapLayeredWidth: Maximum cubemap layered
 *   texture width or height
 * - ::cudaDevAttrMaxTextureCubemapLayeredLayers: Maximum layers in a cubemap
 *   layered texture
 * - ::cudaDevAttrMaxSurface1DWidth: Maximum 1D surface width
 * - ::cudaDevAttrMaxSurface2DWidth: Maximum 2D surface width
 * - ::cudaDevAttrMaxSurface2DHeight: Maximum 2D surface height
 * - ::cudaDevAttrMaxSurface3DWidth: Maximum 3D surface width
 * - ::cudaDevAttrMaxSurface3DHeight: Maximum 3D surface height
 * - ::cudaDevAttrMaxSurface3DDepth: Maximum 3D surface depth
 * - ::cudaDevAttrMaxSurface1DLayeredWidth: Maximum 1D layered surface width
 * - ::cudaDevAttrMaxSurface1DLayeredLayers: Maximum layers in a 1D layered
 *   surface
 * - ::cudaDevAttrMaxSurface2DLayeredWidth: Maximum 2D layered surface width
 * - ::cudaDevAttrMaxSurface2DLayeredHeight: Maximum 2D layered surface height
 * - ::cudaDevAttrMaxSurface2DLayeredLayers: Maximum layers in a 2D layered
 *   surface
 * - ::cudaDevAttrMaxSurfaceCubemapWidth: Maximum cubemap surface width
 * - ::cudaDevAttrMaxSurfaceCubemapLayeredWidth: Maximum cubemap layered
 *   surface width
 * - ::cudaDevAttrMaxSurfaceCubemapLayeredLayers: Maximum layers in a cubemap
 *   layered surface
 * - ::cudaDevAttrMaxRegistersPerBlock: Maximum number of 32-bit registers 
 *   available to a thread block
 * - ::cudaDevAttrClockRate: Peak clock frequency in kilohertz
 * - ::cudaDevAttrTextureAlignment: Alignment requirement; texture base
 *   addresses aligned to ::textureAlign bytes do not need an offset applied
 *   to texture fetches
 * - ::cudaDevAttrTexturePitchAlignment: Pitch alignment requirement for 2D
 *   texture references bound to pitched memory
 * - ::cudaDevAttrGpuOverlap: 1 if the device can concurrently copy memory
 *   between host and device while executing a kernel, or 0 if not
 * - ::cudaDevAttrMultiProcessorCount: Number of multiprocessors on the device
 * - ::cudaDevAttrKernelExecTimeout: 1 if there is a run time limit for kernels
 *   executed on the device, or 0 if not
 * - ::cudaDevAttrIntegrated: 1 if the device is integrated with the memory
 *   subsystem, or 0 if not
 * - ::cudaDevAttrCanMapHostMemory: 1 if the device can map host memory into
 *   the CUDA address space, or 0 if not
 * - ::cudaDevAttrComputeMode: Compute mode is the compute mode that the device
 *   is currently in. Available modes are as follows:
 *   - ::cudaComputeModeDefault: Default mode - Device is not restricted and
 *     multiple threads can use ::cudaSetDevice() with this device.
 *   - ::cudaComputeModeExclusive: Compute-exclusive mode - Only one thread will
 *     be able to use ::cudaSetDevice() with this device.
 *   - ::cudaComputeModeProhibited: Compute-prohibited mode - No threads can use
 *     ::cudaSetDevice() with this device.
 *   - ::cudaComputeModeExclusiveProcess: Compute-exclusive-process mode - Many 
 *     threads in one process will be able to use ::cudaSetDevice() with this
 *     device.
 * - ::cudaDevAttrConcurrentKernels: 1 if the device supports executing
 *   multiple kernels within the same context simultaneously, or 0 if
 *   not. It is not guaranteed that multiple kernels will be resident on the
 *   device concurrently so this feature should not be relied upon for
 *   correctness.
 * - ::cudaDevAttrEccEnabled: 1 if error correction is enabled on the device,
 *   0 if error correction is disabled or not supported by the device
 * - ::cudaDevAttrPciBusId: PCI bus identifier of the device
 * - ::cudaDevAttrPciDeviceId: PCI device (also known as slot) identifier of
 *   the device
 * - ::cudaDevAttrTccDriver: 1 if the device is using a TCC driver. TCC is only
 *   available on Tesla hardware running Windows Vista or later.
 * - ::cudaDevAttrMemoryClockRate: Peak memory clock frequency in kilohertz
 * - ::cudaDevAttrGlobalMemoryBusWidth: Global memory bus width in bits
 * - ::cudaDevAttrL2CacheSize: Size of L2 cache in bytes. 0 if the device
 *   doesn't have L2 cache.
 * - ::cudaDevAttrMaxThreadsPerMultiProcessor: Maximum resident threads per 
 *   multiprocessor
 * - ::cudaDevAttrUnifiedAddressing: 1 if the device shares a unified address
 *   space with the host, or 0 if not
 * - ::cudaDevAttrComputeCapabilityMajor: Major compute capability version
 *   number
 * - ::cudaDevAttrComputeCapabilityMinor: Minor compute capability version
 *   number
 * - ::cudaDevAttrStreamPrioritiesSupported: 1 if the device supports stream
 *   priorities, or 0 if not
 * - ::cudaDevAttrGlobalL1CacheSupported: 1 if device supports caching globals 
 *    in L1 cache, 0 if not
 * - ::cudaDevAttrLocalL1CacheSupported: 1 if device supports caching locals 
 *    in L1 cache, 0 if not
 * - ::cudaDevAttrMaxSharedMemoryPerMultiprocessor: Maximum amount of shared memory
 *   available to a multiprocessor in bytes; this amount is shared by all 
 *   thread blocks simultaneously resident on a multiprocessor
 * - ::cudaDevAttrMaxRegistersPerMultiprocessor: Maximum number of 32-bit registers 
 *   available to a multiprocessor; this number is shared by all thread blocks
 *   simultaneously resident on a multiprocessor
 * - ::cudaDevAttrManagedMemory: 1 if device supports allocating
 *   managed memory, 0 if not
 * - ::cudaDevAttrIsMultiGpuBoard: 1 if device is on a multi-GPU board, 0 if not
 * - ::cudaDevAttrMultiGpuBoardGroupID: Unique identifier for a group of devices on the
 *   same multi-GPU board
 * - ::cudaDevAttrHostNativeAtomicSupported: 1 if the link between the device and the
 *   host supports native atomic operations
 * - ::cudaDevAttrSingleToDoublePrecisionPerfRatio: Ratio of single precision performance
 *   (in floating-point operations per second) to double precision performance
 * - ::cudaDevAttrPageableMemoryAccess: 1 if the device supports coherently accessing
 *   pageable memory without calling cudaHostRegister on it, and 0 otherwise
 * - ::cudaDevAttrConcurrentManagedAccess: 1 if the device can coherently access managed
 *   memory concurrently with the CPU, and 0 otherwise
 * - ::cudaDevAttrComputePreemptionSupported: 1 if the device supports
 *   Compute Preemption, 0 if not
 * - ::cudaDevAttrCanUseHostPointerForRegisteredMem: 1 if the device can access host
 *   registered memory at the same virtual address as the CPU, and 0 otherwise
 * - ::cudaDevAttrCooperativeLaunch: 1 if the device supports launching cooperative kernels
 *   via ::cudaLaunchCooperativeKernel, and 0 otherwise
 * - ::cudaDevAttrCooperativeMultiDeviceLaunch: 1 if the device supports launching cooperative
 *   kernels via ::cudaLaunchCooperativeKernelMultiDevice, and 0 otherwise
 * - ::cudaDevAttrCanFlushRemoteWrites: 1 if the device supports flushing of outstanding 
 *   remote writes, and 0 otherwise
 * - ::cudaDevAttrHostRegisterSupported: 1 if the device supports host memory registration
 *   via ::cudaHostRegister, and 0 otherwise
 * - ::cudaDevAttrPageableMemoryAccessUsesHostPageTables: 1 if the device accesses pageable memory via the
 *   host's page tables, and 0 otherwise
 * - ::cudaDevAttrDirectManagedMemAccessFromHost: 1 if the host can directly access managed memory on the device
 *   without migration, and 0 otherwise
 * - ::cudaDevAttrMaxSharedMemoryPerBlockOptin: Maximum per block shared memory size on the device. This value can
 *   be opted into when using ::cudaFuncSetAttribute
 * - ::cudaDevAttrMaxBlocksPerMultiprocessor: Maximum number of thread blocks that can reside on a multiprocessor
 * - ::cudaDevAttrMaxPersistingL2CacheSize: Maximum L2 persisting lines capacity setting in bytes
 * - ::cudaDevAttrMaxAccessPolicyWindowSize: Maximum value of cudaAccessPolicyWindow::num_bytes
 * - ::cudaDevAttrReservedSharedMemoryPerBlock: Shared memory reserved by CUDA driver per block in bytes
 * - ::cudaDevAttrSparseCudaArraySupported: 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
 * - ::cudaDevAttrHostRegisterReadOnlySupported: Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly
 *   to register memory that must be mapped as read-only to the GPU
 * - ::cudaDevAttrMemoryPoolsSupported: 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, and 0 otherwise
 * - ::cudaDevAttrGPUDirectRDMASupported: 1 if the device supports GPUDirect RDMA APIs, and 0 otherwise
 * - ::cudaDevAttrGPUDirectRDMAFlushWritesOptions: bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum 
 * - ::cudaDevAttrGPUDirectRDMAWritesOrdering: see the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values
 * - ::cudaDevAttrMemoryPoolSupportedHandleTypes: Bitmask of handle types supported with mempool based IPC

 * - ::cudaDevAttrDeferredMappingCudaArraySupported : 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays.

 *
 * \param value  - Returned device attribute value
 * \param attr   - Device attribute to query
 * \param device - Device number to query 
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice, ::cudaChooseDevice,
 * ::cudaGetDeviceProperties,
 * ::cuDeviceGetAttribute
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);

/**
 * \brief Returns the default mempool of a device
 *
 * The default mempool of a device contains device memory from that device.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue
 * ::cudaErrorNotSupported
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuDeviceGetDefaultMemPool, ::cudaMallocAsync, ::cudaMemPoolTrimTo, ::cudaMemPoolGetAttribute, ::cudaDeviceSetMemPool, ::cudaMemPoolSetAttribute, ::cudaMemPoolSetAccess
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device);


/**
 * \brief Sets the current memory pool of a device
 *
 * The memory pool must be local to the specified device.
 * Unless a mempool is specified in the ::cudaMallocAsync call,
 * ::cudaMallocAsync allocates from the current mempool of the provided stream's device.
 * By default, a device's current memory pool is its default memory pool.
 *
 * \note Use ::cudaMallocFromPoolAsync to specify asynchronous allocations from a device different
 * than the one the stream runs on.
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * ::cudaErrorInvalidDevice
 * ::cudaErrorNotSupported
 * \notefnerr
 * \note_callback
 *
 * \sa ::cuDeviceSetDefaultMemPool, ::cudaDeviceGetMemPool, ::cudaDeviceGetDefaultMemPool, ::cudaMemPoolCreate, ::cudaMemPoolDestroy, ::cudaMallocFromPoolAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);

/**
 * \brief Gets the current mempool for a device
 *
 * Returns the last pool provided to ::cudaDeviceSetMemPool for this device
 * or the device's default memory pool if ::cudaDeviceSetMemPool has never been called.
 * By default the current mempool is the default mempool for a device,
 * otherwise the returned pool must have been set with ::cuDeviceSetMemPool or ::cudaDeviceSetMemPool.
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * ::cudaErrorNotSupported
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuDeviceGetMemPool, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceSetMemPool
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device);

/**
 * \brief Return NvSciSync attributes that this device can support.
 *
 * Returns in \p nvSciSyncAttrList, the properties of NvSciSync that
 * this CUDA device, \p dev can support. The returned \p nvSciSyncAttrList
 * can be used to create an NvSciSync that matches this device's capabilities.
 * 
 * If NvSciSyncAttrKey_RequiredPerm field in \p nvSciSyncAttrList is
 * already set this API will return ::cudaErrorInvalidValue.
 * 
 * The applications should set \p nvSciSyncAttrList to a valid 
 * NvSciSyncAttrList failing which this API will return
 * ::cudaErrorInvalidHandle.
 * 
 * The \p flags controls how applications intends to use
 * the NvSciSync created from the \p nvSciSyncAttrList. The valid flags are:
 * - ::cudaNvSciSyncAttrSignal, specifies that the applications intends to 
 * signal an NvSciSync on this CUDA device.
 * - ::cudaNvSciSyncAttrWait, specifies that the applications intends to 
 * wait on an NvSciSync on this CUDA device.
 *
 * At least one of these flags must be set, failing which the API
 * returns ::cudaErrorInvalidValue. Both the flags are orthogonal
 * to one another: a developer may set both these flags that allows to
 * set both wait and signal specific attributes in the same \p nvSciSyncAttrList.
 *
 * \param nvSciSyncAttrList     - Return NvSciSync attributes supported.
 * \param device                - Valid Cuda Device to get NvSciSync attributes for.
 * \param flags                 - flags describing NvSciSync usage.
 *
 * \return
 *
 * ::cudaSuccess,
 * ::cudaErrorDeviceUninitialized,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidHandle,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorNotSupported,
 * ::cudaErrorMemoryAllocation
 *
 * \sa
 * ::cudaImportExternalSemaphore,
 * ::cudaDestroyExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags);

/**
 * \brief Queries attributes of the link between two devices.
 *
 * Returns in \p *value the value of the requested attribute \p attrib of the
 * link between \p srcDevice and \p dstDevice. The supported attributes are:
 * - ::cudaDevP2PAttrPerformanceRank: A relative value indicating the
 *   performance of the link between two devices. Lower value means better
 *   performance (0 being the value used for most performant link).
 * - ::cudaDevP2PAttrAccessSupported: 1 if peer access is enabled.
 * - ::cudaDevP2PAttrNativeAtomicSupported: 1 if native atomic operations over
 *   the link are supported.
 * - ::cudaDevP2PAttrCudaArrayAccessSupported: 1 if accessing CUDA arrays over
 *   the link is supported.
 *
 * Returns ::cudaErrorInvalidDevice if \p srcDevice or \p dstDevice are not valid
 * or if they represent the same device.
 *
 * Returns ::cudaErrorInvalidValue if \p attrib is not valid or if \p value is
 * a null pointer.
 *
 * \param value         - Returned value of the requested attribute
 * \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
 * \param srcDevice     - The source device of the target link.
 * \param dstDevice     - The destination device of the target link.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaCtxEnablePeerAccess,
 * ::cudaCtxDisablePeerAccess,
 * ::cudaCtxCanAccessPeer,
 * ::cuDeviceGetP2PAttribute
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);

/**
 * \brief Select compute-device which best matches criteria
 *
 * Returns in \p *device the device which has properties that best match
 * \p *prop.
 *
 * \param device - Device with best match
 * \param prop   - Desired device properties
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice,
 * ::cudaGetDeviceProperties
 */
extern __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);

/**
 * \brief Set device to be used for GPU executions
 *
 * Sets \p device as the current device for the calling host thread.
 * Valid device id's are 0 to (::cudaGetDeviceCount() - 1).
 *
 * Any device memory subsequently allocated from this host thread
 * using ::cudaMalloc(), ::cudaMallocPitch() or ::cudaMallocArray()
 * will be physically resident on \p device.  Any host memory allocated
 * from this host thread using ::cudaMallocHost() or ::cudaHostAlloc() 
 * or ::cudaHostRegister() will have its lifetime associated  with
 * \p device.  Any streams or events created from this host thread will 
 * be associated with \p device.  Any kernels launched from this host
 * thread using the <<<>>> operator or ::cudaLaunchKernel() will be executed
 * on \p device.
 *
 * This call may be made from any host thread, to any device, and at 
 * any time.  This function will do no synchronization with the previous 
 * or new device, and should be considered a very low overhead call.
 * If the current context bound to the calling thread is not the primary context,
 * this call will bind the primary context to the calling thread and all the
 * subsequent memory allocations, stream and event creations, and kernel launches
 * will be associated with the primary context.
 *
 * \param device - Device on which the active host thread should execute the
 * device code.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorDeviceAlreadyInUse
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaGetDeviceProperties,
 * ::cudaChooseDevice,
 * ::cuCtxSetCurrent
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device);

/**
 * \brief Returns which device is currently being used
 *
 * Returns in \p *device the current device for the calling host thread.
 *
 * \param device - Returns the device on which the active host thread
 * executes the device code.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaSetDevice, ::cudaGetDeviceProperties,
 * ::cudaChooseDevice,
 * ::cuCtxGetCurrent
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device);

/**
 * \brief Set a list of devices that can be used for CUDA
 *
 * Sets a list of devices for CUDA execution in priority order using
 * \p device_arr. The parameter \p len specifies the number of elements in the
 * list.  CUDA will try devices from the list sequentially until it finds one
 * that works.  If this function is not called, or if it is called with a \p len
 * of 0, then CUDA will go back to its default behavior of trying devices
 * sequentially from a default list containing all of the available CUDA
 * devices in the system. If a specified device ID in the list does not exist,
 * this function will return ::cudaErrorInvalidDevice. If \p len is not 0 and
 * \p device_arr is NULL or if \p len exceeds the number of devices in
 * the system, then ::cudaErrorInvalidValue is returned.
 *
 * \param device_arr - List of devices to try
 * \param len        - Number of devices in specified list
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaSetDevice, ::cudaGetDeviceProperties,
 * ::cudaSetDeviceFlags,
 * ::cudaChooseDevice
 */
extern __host__ cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len);

/**
 * \brief Sets flags to be used for device executions
 * 
 * Records \p flags as the flags for the current device. If the current device
 * has been set and that device has already been initialized, the previous flags
 * are overwritten. If the current device has not been initialized, it is
 * initialized with the provided flags. If no device has been made current to
 * the calling thread, a default device is selected and initialized with the
 * provided flags.
 * 
 * The two LSBs of the \p flags parameter can be used to control how the CPU
 * thread interacts with the OS scheduler when waiting for results from the
 * device.
 *
 * - ::cudaDeviceScheduleAuto: The default value if the \p flags parameter is
 * zero, uses a heuristic based on the number of active CUDA contexts in the
 * process \p C and the number of logical processors in the system \p P. If
 * \p C \> \p P, then CUDA will yield to other OS threads when waiting for the
 * device, otherwise CUDA will not yield while waiting for results and
 * actively spin on the processor. Additionally, on Tegra devices,
 * ::cudaDeviceScheduleAuto uses a heuristic based on the power profile of
 * the platform and may choose ::cudaDeviceScheduleBlockingSync for low-powered
 * devices.
 * - ::cudaDeviceScheduleSpin: Instruct CUDA to actively spin when waiting for
 * results from the device. This can decrease latency when waiting for the
 * device, but may lower the performance of CPU threads if they are performing
 * work in parallel with the CUDA thread.
 * - ::cudaDeviceScheduleYield: Instruct CUDA to yield its thread when waiting
 * for results from the device. This can increase latency when waiting for the
 * device, but can increase the performance of CPU threads performing work in
 * parallel with the device.
 * - ::cudaDeviceScheduleBlockingSync: Instruct CUDA to block the CPU thread 
 * on a synchronization primitive when waiting for the device to finish work.
 * - ::cudaDeviceBlockingSync: Instruct CUDA to block the CPU thread on a 
 * synchronization primitive when waiting for the device to finish work. <br>
 * \ref deprecated "Deprecated:" This flag was deprecated as of CUDA 4.0 and
 * replaced with ::cudaDeviceScheduleBlockingSync.
 * - ::cudaDeviceMapHost: This flag enables allocating pinned
 * host memory that is accessible to the device. It is implicit for the
 * runtime but may be absent if a context is created using the driver API.
 * If this flag is not set, ::cudaHostGetDevicePointer() will always return
 * a failure code.
 * - ::cudaDeviceLmemResizeToMax: Instruct CUDA to not reduce local memory
 * after resizing local memory for a kernel. This can prevent thrashing by
 * local memory allocations when launching many kernels with high local
 * memory usage at the cost of potentially increased memory usage. <br>
 * \ref deprecated "Deprecated:" This flag is deprecated and the behavior enabled          
 * by this flag is now the default and cannot be disabled.
 *
 * \param flags - Parameters for device operation
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceFlags, ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaGetDeviceProperties,
 * ::cudaSetDevice, ::cudaSetValidDevices,
 * ::cudaChooseDevice,
 * ::cuDevicePrimaryCtxSetFlags
 */
extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags );

/**
 * \brief Gets the flags for the current device
 *
 * 
 * Returns in \p flags the flags for the current device. If there is a current
 * device for the calling thread, the flags for the device are returned. If
 * there is no current device, the flags for the first device are returned,
 * which may be the default flags.  Compare to the behavior of
 * ::cudaSetDeviceFlags.
 *
 * Typically, the flags returned should match the behavior that will be seen
 * if the calling thread uses a device after this call, without any change to
 * the flags or current device inbetween by this or another thread.  Note that
 * if the device is not initialized, it is possible for another thread to
 * change the flags for the current device before it is initialized.
 * Additionally, when using exclusive mode, if this thread has not requested a
 * specific device, it may use a device other than the first device, contrary
 * to the assumption made by this function.
 *
 * If a context has been created via the driver API and is current to the
 * calling thread, the flags for that context are always returned.
 *
 * Flags returned by this function may specifically include ::cudaDeviceMapHost
 * even though it is not accepted by ::cudaSetDeviceFlags because it is
 * implicit in runtime API flags.  The reason for this is that the current
 * context may have been created via the driver API in which case the flag is
 * not implicit and may be unset.
 *
 * \param flags - Pointer to store the device flags
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDevice, ::cudaGetDeviceProperties,
 * ::cudaSetDevice, ::cudaSetDeviceFlags,
 * ::cuCtxGetFlags,
 * ::cuDevicePrimaryCtxGetState
 */
extern __host__ cudaError_t CUDARTAPI cudaGetDeviceFlags( unsigned int *flags );
/** @} */ /* END CUDART_DEVICE */

/**
 * \defgroup CUDART_STREAM Stream Management
 *
 * ___MANBRIEF___ stream management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the stream management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream.
 *
 * \param pStream - Pointer to new stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaStreamCreateWithFlags,
 * ::cudaStreamGetPriority,
 * ::cudaStreamGetFlags,
 * ::cudaStreamQuery,
 * ::cudaStreamSynchronize,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamAddCallback,
 * ::cudaStreamDestroy,
 * ::cuStreamCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream);

/**
 * \brief Create an asynchronous stream
 *
 * Creates a new asynchronous stream.  The \p flags argument determines the 
 * behaviors of the stream.  Valid values for \p flags are
 * - ::cudaStreamDefault: Default stream creation flag.
 * - ::cudaStreamNonBlocking: Specifies that work running in the created 
 *   stream may run concurrently with work in stream 0 (the NULL stream), and that
 *   the created stream should perform no implicit synchronization with stream 0.
 *
 * \param pStream - Pointer to new stream identifier
 * \param flags   - Parameters for stream creation
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate,
 * ::cudaStreamCreateWithPriority,
 * ::cudaStreamGetFlags,
 * ::cudaStreamQuery,
 * ::cudaStreamSynchronize,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamAddCallback,
 * ::cudaStreamDestroy,
 * ::cuStreamCreate
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);

/**
 * \brief Create an asynchronous stream with the specified priority
 *
 * Creates a stream with the specified priority and returns a handle in \p pStream.
 * This API alters the scheduler priority of work in the stream. Work in a higher
 * priority stream may preempt work already executing in a low priority stream.
 *
 * \p priority follows a convention where lower numbers represent higher priorities.
 * '0' represents default priority. The range of meaningful numerical priorities can
 * be queried using ::cudaDeviceGetStreamPriorityRange. If the specified priority is
 * outside the numerical range returned by ::cudaDeviceGetStreamPriorityRange,
 * it will automatically be clamped to the lowest or the highest number in the range.
 *
 * \param pStream  - Pointer to new stream identifier
 * \param flags    - Flags for stream creation. See ::cudaStreamCreateWithFlags for a list of valid flags that can be passed
 * \param priority - Priority of the stream. Lower numbers represent higher priorities.
 *                   See ::cudaDeviceGetStreamPriorityRange for more information about
 *                   the meaningful stream priorities that can be passed.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \note Stream priorities are supported only on GPUs
 * with compute capability 3.5 or higher.
 *
 * \note In the current implementation, only compute kernels launched in
 * priority streams are affected by the stream's priority. Stream priorities have
 * no effect on host-to-device and device-to-host memory operations.
 *
 * \sa ::cudaStreamCreate,
 * ::cudaStreamCreateWithFlags,
 * ::cudaDeviceGetStreamPriorityRange,
 * ::cudaStreamGetPriority,
 * ::cudaStreamQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamAddCallback,
 * ::cudaStreamSynchronize,
 * ::cudaStreamDestroy,
 * ::cuStreamCreateWithPriority
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);

/**
 * \brief Query the priority of a stream
 *
 * Query the priority of a stream. The priority is returned in in \p priority.
 * Note that if the stream was created with a priority outside the meaningful
 * numerical range returned by ::cudaDeviceGetStreamPriorityRange,
 * this function returns the clamped priority.
 * See ::cudaStreamCreateWithPriority for details about priority clamping.
 *
 * \param hStream    - Handle to the stream to be queried
 * \param priority   - Pointer to a signed integer in which the stream's priority is returned
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaDeviceGetStreamPriorityRange,
 * ::cudaStreamGetFlags,
 * ::cuStreamGetPriority
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority);

/**
 * \brief Query the flags of a stream
 *
 * Query the flags of a stream. The flags are returned in \p flags.
 * See ::cudaStreamCreateWithFlags for a list of valid flags.
 *
 * \param hStream - Handle to the stream to be queried
 * \param flags   - Pointer to an unsigned integer in which the stream's flags are returned
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreateWithPriority,
 * ::cudaStreamCreateWithFlags,
 * ::cudaStreamGetPriority,
 * ::cuStreamGetFlags
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);

/**
 * \brief Resets all persisting lines in cache to normal status.
 *
 * Resets all persisting lines in cache to normal status.
 * Takes effect on function return.
 *
 * \return
 * ::cudaSuccess,
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache(void);

/**
 * \brief Copies attributes from source stream to destination stream.
 *
 * Copies attributes from source stream \p src to destination stream \p dst.
 * Both streams must have the same context.
 *
 * \param[out] dst Destination stream
 * \param[in] src Source stream
 * For attributes see ::cudaStreamAttrID
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotSupported
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);

 /**
 * \brief Queries stream attribute.
 *
 * Queries attribute \p attr from \p hStream and stores it in corresponding
 * member of \p value_out.
 *
 * \param[in] hStream
 * \param[in] attr
 * \param[out] value_out
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetAttribute(
        cudaStream_t hStream, enum cudaStreamAttrID attr,
        union cudaStreamAttrValue *value_out);

 /**
 * \brief Sets stream attribute.
 *
 * Sets attribute \p attr on \p hStream from corresponding attribute of
 * \p value. The updated attribute will be applied to subsequent work
 * submitted to the stream. It will not affect previously submitted work.
 *
 * \param[out] hStream
 * \param[in] attr
 * \param[in] value
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamSetAttribute(
        cudaStream_t hStream, enum cudaStreamAttrID attr,
        const union cudaStreamAttrValue *value);

 /**
 * \brief Destroys and cleans up an asynchronous stream
 *
 * Destroys and cleans up the asynchronous stream specified by \p stream.
 *
 * In case the device is still doing work in the stream \p stream
 * when ::cudaStreamDestroy() is called, the function will return immediately 
 * and the resources associated with \p stream will be released automatically 
 * once the device has completed all work in \p stream.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa ::cudaStreamCreate,
 * ::cudaStreamCreateWithFlags,
 * ::cudaStreamQuery,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamSynchronize,
 * ::cudaStreamAddCallback,
 * ::cuStreamDestroy
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);

/**
 * \brief Make a compute stream wait on an event
 *
 * Makes all future work submitted to \p stream wait for all work captured in
 * \p event.  See ::cudaEventRecord() for details on what is captured by an event.
 * The synchronization will be performed efficiently on the device when applicable.
 * \p event may be from a different device than \p stream.
 *
 * flags include:
 * - ::cudaEventWaitDefault: Default event creation flag.
 * - ::cudaEventWaitExternal: Event is captured in the graph as an external
 *   event node when performing stream capture.
 *
 * \param stream - Stream to wait
 * \param event  - Event to wait on
 * \param flags  - Parameters for the operation(See above)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy,
 * ::cuStreamWaitEvent
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags __dv(0));

/**
 * Type of stream callback functions.
 * \param stream The stream as passed to ::cudaStreamAddCallback, may be NULL.
 * \param status ::cudaSuccess or any persistent error on the stream.
 * \param userData User parameter provided at registration.
 */
typedef void (CUDART_CB *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);

/**
 * \brief Add a callback to a compute stream
 *
 * \note This function is slated for eventual deprecation and removal. If
 * you do not require the callback to execute in case of a device error,
 * consider using ::cudaLaunchHostFunc. Additionally, this function is not
 * supported with ::cudaStreamBeginCapture and ::cudaStreamEndCapture, unlike
 * ::cudaLaunchHostFunc.
 *
 * Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each 
 * cudaStreamAddCallback call, a callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 *
 * The callback may be passed ::cudaSuccess or an error code.  In the event
 * of a device error, all subsequently executed callbacks will receive an
 * appropriate ::cudaError_t.
 *
 * Callbacks must not make any CUDA API calls.  Attempting to use CUDA APIs
 * may result in ::cudaErrorNotPermitted.  Callbacks must not perform any
 * synchronization that may depend on outstanding device work or other callbacks
 * that are not mandated to run earlier.  Callbacks without a mandated order
 * (in independent streams) execute in undefined order and may be serialized.
 *
 * For the purposes of Unified Memory, callback execution makes a number of
 * guarantees:
 * <ul>
 *   <li>The callback stream is considered idle for the duration of the
 *   callback.  Thus, for example, a callback may always use memory attached
 *   to the callback stream.</li>
 *   <li>The start of execution of a callback has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the callback.  It thus synchronizes streams which have been "joined"
 *   prior to the callback.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding callbacks have executed.  Thus, for
 *   example, a callback might use global attached memory even if work has
 *   been added to another stream, if it has been properly ordered with an
 *   event.</li>
 *   <li>Completion of a callback does not cause a stream to become
 *   active except as described above.  The callback stream will remain idle
 *   if no device work follows the callback, and will remain idle across
 *   consecutive callbacks without device work in between.  Thus, for example,
 *   stream synchronization can be done by signaling from a callback at the
 *   end of the stream.</li>
 * </ul>
 *
 * \param stream   - Stream to add callback to
 * \param callback - The function to call once preceding stream operations are complete
 * \param userData - User specified data to be passed to the callback function
 * \param flags    - Reserved for future use, must be 0
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamSynchronize, ::cudaStreamWaitEvent, ::cudaStreamDestroy, ::cudaMallocManaged, ::cudaStreamAttachMemAsync,
 * ::cudaLaunchHostFunc, ::cuStreamAddCallback
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,
        cudaStreamCallback_t callback, void *userData, unsigned int flags);

/**
 * \brief Waits for stream tasks to complete
 *
 * Blocks until \p stream has completed all operations. If the
 * ::cudaDeviceScheduleBlockingSync flag was set for this device, 
 * the host thread will block until the stream is finished with 
 * all of its tasks.
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamQuery, ::cudaStreamWaitEvent, ::cudaStreamAddCallback, ::cudaStreamDestroy,
 * ::cuStreamSynchronize
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);

/**
 * \brief Queries an asynchronous stream for completion status
 *
 * Returns ::cudaSuccess if all operations in \p stream have
 * completed, or ::cudaErrorNotReady if not.
 *
 * For the purposes of Unified Memory, a return value of ::cudaSuccess
 * is equivalent to having called ::cudaStreamSynchronize().
 *
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidResourceHandle
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamWaitEvent, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy,
 * ::cuStreamQuery
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream);

/**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p stream to specify stream association of
 * \p length bytes of memory starting from \p devPtr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p devPtr must point to an one of the following types of memories:
 * - managed memory declared using the __managed__ keyword or allocated with
 *   ::cudaMallocManaged.
 * - a valid host-accessible region of system-allocated pageable memory. This
 *   type of memory may only be specified if the device associated with the
 *   stream reports a non-zero value for the device attribute
 *   ::cudaDevAttrPageableMemoryAccess.
 *
 * For managed allocations, \p length must be either zero or the entire
 * allocation's size. Both indicate that the entire allocation's stream
 * association is being changed. Currently, it is not possible to change stream
 * association for a portion of a managed allocation.
 *
 * For pageable allocations, \p length must be non-zero.
 *
 * The stream association is specified using \p flags which must be
 * one of ::cudaMemAttachGlobal, ::cudaMemAttachHost or ::cudaMemAttachSingle.
 * The default value for \p flags is ::cudaMemAttachSingle
 * If the ::cudaMemAttachGlobal flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::cudaMemAttachHost flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream on a device that
 * has a zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess.
 * If the ::cudaMemAttachSingle flag is specified and \p stream is associated with
 * a device that has a zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess,
 * the program makes a guarantee that it will only access the memory on the device
 * from \p stream. It is illegal to attach singly to the NULL stream, because the
 * NULL stream is a virtual global stream and not a specific stream. An error will
 * be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p stream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownership of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region. 
 *
 * It is a program's responsibility to order calls to ::cudaStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p stream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::cudaMallocManaged. For __managed__ variables, the default
 * association is always ::cudaMemAttachGlobal. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param stream  - Stream in which to enqueue the attach operation
 * \param devPtr  - Pointer to memory (must be a pointer to managed memory or
 *                  to a valid host-accessible region of system-allocated
 *                  memory)
 * \param length  - Length of memory (defaults to zero)
 * \param flags   - Must be one of ::cudaMemAttachGlobal, ::cudaMemAttachHost or ::cudaMemAttachSingle (defaults to ::cudaMemAttachSingle)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamWaitEvent, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy, ::cudaMallocManaged,
 * ::cuStreamAttachMemAsync
 */
#if defined(__cplusplus)
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags = cudaMemAttachSingle);
#else
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags);
#endif

/**
 * \brief Begins graph capture on a stream
 *
 * Begin graph capture on \p stream. When a stream is in capture mode, all operations
 * pushed into the stream will not be executed, but will instead be captured into
 * a graph, which will be returned via ::cudaStreamEndCapture. Capture may not be initiated
 * if \p stream is ::cudaStreamLegacy. Capture must be ended on the same stream in which
 * it was initiated, and it may only be initiated if the stream is not already in capture
 * mode. The capture mode may be queried via ::cudaStreamIsCapturing. A unique id
 * representing the capture sequence may be queried via ::cudaStreamGetCaptureInfo.
 *
 * If \p mode is not ::cudaStreamCaptureModeRelaxed, ::cudaStreamEndCapture must be
 * called on this stream from the same thread.
 *
 * \note Kernels captured using this API must not use texture and surface references.
 *       Reading or writing through any texture or surface reference is undefined
 *       behavior. This restriction does not apply to texture and surface objects.
 *
 * \param stream - Stream in which to initiate capture
 * \param mode    - Controls the interaction of this capture sequence with other API
 *                  calls that are potentially unsafe. For more details see
 *                  ::cudaThreadExchangeStreamCaptureMode.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa
 * ::cudaStreamCreate,
 * ::cudaStreamIsCapturing,
 * ::cudaStreamEndCapture,
 * ::cudaThreadExchangeStreamCaptureMode
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);

/**
 * \brief Swaps the stream capture interaction mode for a thread
 *
 * Sets the calling thread's stream capture interaction mode to the value contained
 * in \p *mode, and overwrites \p *mode with the previous mode for the thread. To
 * facilitate deterministic behavior across function or module boundaries, callers
 * are encouraged to use this API in a push-pop fashion: \code
     cudaStreamCaptureMode mode = desiredMode;
     cudaThreadExchangeStreamCaptureMode(&mode);
     ...
     cudaThreadExchangeStreamCaptureMode(&mode); // restore previous mode
 * \endcode
 *
 * During stream capture (see ::cudaStreamBeginCapture), some actions, such as a call
 * to ::cudaMalloc, may be unsafe. In the case of ::cudaMalloc, the operation is
 * not enqueued asynchronously to a stream, and is not observed by stream capture.
 * Therefore, if the sequence of operations captured via ::cudaStreamBeginCapture
 * depended on the allocation being replayed whenever the graph is launched, the
 * captured graph would be invalid.
 *
 * Therefore, stream capture places restrictions on API calls that can be made within
 * or concurrently to a ::cudaStreamBeginCapture-::cudaStreamEndCapture sequence. This
 * behavior can be controlled via this API and flags to ::cudaStreamBeginCapture.
 *
 * A thread's mode is one of the following:
 * - \p cudaStreamCaptureModeGlobal: This is the default mode. If the local thread has
 *   an ongoing capture sequence that was not initiated with
 *   \p cudaStreamCaptureModeRelaxed at \p cuStreamBeginCapture, or if any other thread
 *   has a concurrent capture sequence initiated with \p cudaStreamCaptureModeGlobal,
 *   this thread is prohibited from potentially unsafe API calls.
 * - \p cudaStreamCaptureModeThreadLocal: If the local thread has an ongoing capture
 *   sequence not initiated with \p cudaStreamCaptureModeRelaxed, it is prohibited
 *   from potentially unsafe API calls. Concurrent capture sequences in other threads
 *   are ignored.
 * - \p cudaStreamCaptureModeRelaxed: The local thread is not prohibited from potentially
 *   unsafe API calls. Note that the thread is still prohibited from API calls which
 *   necessarily conflict with stream capture, for example, attempting ::cudaEventQuery
 *   on an event that was last recorded inside a capture sequence.
 *
 * \param mode - Pointer to mode value to swap with the current mode
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \sa
 * ::cudaStreamBeginCapture
 */
extern __host__ cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode);

/**
 * \brief Ends capture on a stream, returning the captured graph
 *
 * End capture on \p stream, returning the captured graph via \p pGraph.
 * Capture must have been initiated on \p stream via a call to ::cudaStreamBeginCapture.
 * If capture was invalidated, due to a violation of the rules of stream capture, then
 * a NULL graph will be returned.
 *
 * If the \p mode argument to ::cudaStreamBeginCapture was not
 * ::cudaStreamCaptureModeRelaxed, this call must be from the same thread as
 * ::cudaStreamBeginCapture.
 *
 * \param stream - Stream to query
 * \param pGraph - The captured graph
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorStreamCaptureWrongThread
 * \notefnerr
 *
 * \sa
 * ::cudaStreamCreate,
 * ::cudaStreamBeginCapture,
 * ::cudaStreamIsCapturing
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);

/**
 * \brief Returns a stream's capture status
 *
 * Return the capture status of \p stream via \p pCaptureStatus. After a successful
 * call, \p *pCaptureStatus will contain one of the following:
 * - ::cudaStreamCaptureStatusNone: The stream is not capturing.
 * - ::cudaStreamCaptureStatusActive: The stream is capturing.
 * - ::cudaStreamCaptureStatusInvalidated: The stream was capturing but an error
 *   has invalidated the capture sequence. The capture sequence must be terminated
 *   with ::cudaStreamEndCapture on the stream where it was initiated in order to
 *   continue using \p stream.
 *
 * Note that, if this is called on ::cudaStreamLegacy (the "null stream") while
 * a blocking stream on the same device is capturing, it will return
 * ::cudaErrorStreamCaptureImplicit and \p *pCaptureStatus is unspecified
 * after the call. The blocking stream capture is not invalidated.
 *
 * When a blocking stream is capturing, the legacy stream is in an
 * unusable state until the blocking stream capture is terminated. The legacy
 * stream is not supported for stream capture, but attempted use would have an
 * implicit dependency on the capturing stream(s).
 *
 * \param stream         - Stream to query
 * \param pCaptureStatus - Returns the stream's capture status
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorStreamCaptureImplicit
 * \notefnerr
 *
 * \sa
 * ::cudaStreamCreate,
 * ::cudaStreamBeginCapture,
 * ::cudaStreamEndCapture
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);

/**
 * \brief Query capture status of a stream
 *
 * Note there is a later version of this API, ::cudaStreamGetCaptureInfo_v2. It will
 * supplant this version in 12.0, which is retained for minor version compatibility.
 *
 * Query the capture status of a stream and get a unique id representing
 * the capture sequence over the lifetime of the process.
 *
 * If called on ::cudaStreamLegacy (the "null stream") while a stream not created
 * with ::cudaStreamNonBlocking is capturing, returns ::cudaErrorStreamCaptureImplicit.
 *
 * A valid id is returned only if both of the following are true:
 * - the call returns ::cudaSuccess
 * - captureStatus is set to ::cudaStreamCaptureStatusActive
 *
 * \param stream         - Stream to query
 * \param pCaptureStatus - Returns the stream's capture status
 * \param pId            - Returns the unique id of the capture sequence
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorStreamCaptureImplicit
 * \notefnerr
 *
 * \sa
 * ::cudaStreamGetCaptureInfo_v2,
 * ::cudaStreamBeginCapture,
 * ::cudaStreamIsCapturing
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId);

/**
 * \brief Query a stream's capture state (11.3+)
 *
 * Query stream state related to stream capture.
 *
 * If called on ::cudaStreamLegacy (the "null stream") while a stream not created 
 * with ::cudaStreamNonBlocking is capturing, returns ::cudaErrorStreamCaptureImplicit.
 *
 * Valid data (other than capture status) is returned only if both of the following are true:
 * - the call returns cudaSuccess
 * - the returned capture status is ::cudaStreamCaptureStatusActive
 *
 * This version of cudaStreamGetCaptureInfo is introduced in CUDA 11.3 and will supplant the
 * previous version ::cudaStreamGetCaptureInfo in 12.0. Developers requiring compatibility
 * across minor versions to CUDA 11.0 (driver version 445) can do one of the following:
 * - Use the older version of the API, ::cudaStreamGetCaptureInfo
 * - Pass null for all of \p graph_out, \p dependencies_out, and \p numDependencies_out.
 *
 * \param stream - The stream to query
 * \param captureStatus_out - Location to return the capture status of the stream; required
 * \param id_out - Optional location to return an id for the capture sequence, which is
 *           unique over the lifetime of the process
 * \param graph_out - Optional location to return the graph being captured into. All
 *           operations other than destroy and node removal are permitted on the graph
 *           while the capture sequence is in progress. This API does not transfer
 *           ownership of the graph, which is transferred or destroyed at
 *           ::cudaStreamEndCapture. Note that the graph handle may be invalidated before
 *           end of capture for certain errors. Nodes that are or become
 *           unreachable from the original stream at ::cudaStreamEndCapture due to direct
 *           actions on the graph do not trigger ::cudaErrorStreamCaptureUnjoined.
 * \param dependencies_out - Optional location to store a pointer to an array of nodes.
 *           The next node to be captured in the stream will depend on this set of nodes,
 *           absent operations such as event wait which modify this set. The array pointer
 *           is valid until the next API call which operates on the stream or until end of
 *           capture. The node handles may be copied out and are valid until they or the
 *           graph is destroyed. The driver-owned array may also be passed directly to
 *           APIs that operate on the graph (not the stream) without copying.
 * \param numDependencies_out - Optional location to store the size of the array
 *           returned in dependencies_out.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorStreamCaptureImplicit
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cudaStreamGetCaptureInfo,
 * ::cudaStreamBeginCapture,
 * ::cudaStreamIsCapturing,
 * ::cudaStreamUpdateCaptureDependencies
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), cudaGraph_t *graph_out __dv(0), const cudaGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0));

/**
 * \brief Update the set of dependencies in a capturing stream (11.3+)
 *
 * Modifies the dependency set of a capturing stream. The dependency set is the set
 * of nodes that the next captured node in the stream will depend on.
 *
 * Valid flags are ::cudaStreamAddCaptureDependencies and
 * ::cudaStreamSetCaptureDependencies. These control whether the set passed to
 * the API is added to the existing set or replaces it. A flags value of 0 defaults
 * to ::cudaStreamAddCaptureDependencies.
 *
 * Nodes that are removed from the dependency set via this API do not result in
 * ::cudaErrorStreamCaptureUnjoined if they are unreachable from the stream at
 * ::cudaStreamEndCapture.
 *
 * Returns ::cudaErrorIllegalState if the stream is not capturing.
 *
 * This API is new in CUDA 11.3. Developers requiring compatibility across minor
 * versions of the CUDA driver to 11.0 should not use this API or provide a fallback.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorIllegalState
 * \notefnerr
 *
 * \sa
 * ::cudaStreamBeginCapture,
 * ::cudaStreamGetCaptureInfo,
 * ::cudaStreamGetCaptureInfo_v2
 */
extern __host__ cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags __dv(0));
/** @} */ /* END CUDART_STREAM */

/**
 * \defgroup CUDART_EVENT Event Management
 *
 * ___MANBRIEF___ event management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the event management functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Creates an event object
 *
 * Creates an event object for the current device using ::cudaEventDefault.
 *
 * \param event - Newly created event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*, unsigned int) "cudaEventCreate (C++ API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent,
 * ::cuEventCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event);

/**
 * \brief Creates an event object with the specified flags
 *
 * Creates an event object for the current device with the specified flags. Valid
 * flags include:
 * - ::cudaEventDefault: Default event creation flag.
 * - ::cudaEventBlockingSync: Specifies that event should use blocking
 *   synchronization. A host thread that uses ::cudaEventSynchronize() to wait
 *   on an event created with this flag will block until the event actually
 *   completes.
 * - ::cudaEventDisableTiming: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::cudaEventBlockingSync flag not specified will provide the best
 *   performance when used with ::cudaStreamWaitEvent() and ::cudaEventQuery().
 * - ::cudaEventInterprocess: Specifies that the created event may be used as an
 *   interprocess event by ::cudaIpcGetEventHandle(). ::cudaEventInterprocess must
 *   be specified along with ::cudaEventDisableTiming.
 *
 * \param event - Newly created event
 * \param flags - Flags for new event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent,
 * ::cuEventCreate
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);

/**
 * \brief Records an event
 *
 * Captures in \p event the contents of \p stream at the time of this call.
 * \p event and \p stream must be on the same CUDA context.
 * Calls such as ::cudaEventQuery() or ::cudaStreamWaitEvent() will then
 * examine or wait for completion of the work that was captured. Uses of
 * \p stream after this call do not modify \p event. See note on default
 * stream behavior for what is captured in the default case.
 *
 * ::cudaEventRecord() can be called multiple times on the same event and
 * will overwrite the previously captured state. Other APIs such as
 * ::cudaStreamWaitEvent() use the most recently captured state at the time
 * of the API call, and are not affected by later calls to
 * ::cudaEventRecord(). Before the first call to ::cudaEventRecord(), an
 * event represents an empty set of work, so for example ::cudaEventQuery()
 * would return ::cudaSuccess.
 *
 * \param event  - Event to record
 * \param stream - Stream in which to record event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent,
 * ::cudaEventRecordWithFlags,
 * ::cuEventRecord
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));

/**
 * \brief Records an event
 *
 * Captures in \p event the contents of \p stream at the time of this call.
 * \p event and \p stream must be on the same CUDA context.
 * Calls such as ::cudaEventQuery() or ::cudaStreamWaitEvent() will then
 * examine or wait for completion of the work that was captured. Uses of
 * \p stream after this call do not modify \p event. See note on default
 * stream behavior for what is captured in the default case.
 *
 * ::cudaEventRecordWithFlags() can be called multiple times on the same event and
 * will overwrite the previously captured state. Other APIs such as
 * ::cudaStreamWaitEvent() use the most recently captured state at the time
 * of the API call, and are not affected by later calls to
 * ::cudaEventRecordWithFlags(). Before the first call to ::cudaEventRecordWithFlags(), an
 * event represents an empty set of work, so for example ::cudaEventQuery()
 * would return ::cudaSuccess.
 *
 * flags include:
 * - ::cudaEventRecordDefault: Default event creation flag.
 * - ::cudaEventRecordExternal: Event is captured in the graph as an external
 *   event node when performing stream capture.
 *
 * \param event  - Event to record
 * \param stream - Stream in which to record event
 * \param flags  - Parameters for the operation(See above)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent,
 * ::cudaEventRecord,
 * ::cuEventRecord,
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream __dv(0), unsigned int flags __dv(0));
#endif

/**
 * \brief Queries an event's status
 *
 * Queries the status of all work currently captured by \p event. See
 * ::cudaEventRecord() for details on what is captured by an event.
 *
 * Returns ::cudaSuccess if all captured work has been completed, or
 * ::cudaErrorNotReady if any captured work is incomplete.
 *
 * For the purposes of Unified Memory, a return value of ::cudaSuccess
 * is equivalent to having called ::cudaEventSynchronize().
 *
 * \param event - Event to query
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cuEventQuery
 */
extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event);

/**
 * \brief Waits for an event to complete
 *
 * Waits until the completion of all work currently captured in \p event.
 * See ::cudaEventRecord() for details on what is captured by an event.
 *
 * Waiting for an event that was created with the ::cudaEventBlockingSync
 * flag will cause the calling CPU thread to block until the event has
 * been completed by the device.  If the ::cudaEventBlockingSync flag has
 * not been set, then the CPU thread will busy-wait until the event has
 * been completed by the device.
 *
 * \param event - Event to wait for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord,
 * ::cudaEventQuery, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cuEventSynchronize
 */
extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event);

/**
 * \brief Destroys an event object
 *
 * Destroys the event specified by \p event.
 *
 * An event may be destroyed before it is complete (i.e., while
 * ::cudaEventQuery() would return ::cudaErrorNotReady). In this case, the
 * call does not block on completion of the event, and any associated
 * resources will automatically be released asynchronously at completion.
 *
 * \param event - Event to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventRecord, ::cudaEventElapsedTime,
 * ::cuEventDestroy
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event);

/**
 * \brief Computes the elapsed time between events
 *
 * Computes the elapsed time between two events (in milliseconds with a
 * resolution of around 0.5 microseconds).
 *
 * If either event was last recorded in a non-NULL stream, the resulting time
 * may be greater than expected (even if both used the same stream handle). This
 * happens because the ::cudaEventRecord() operation takes place asynchronously
 * and there is no guarantee that the measured latency is actually just between
 * the two events. Any number of other different stream operations could execute
 * in between the two measured events, thus altering the timing in a significant
 * way.
 *
 * If ::cudaEventRecord() has not been called on either event, then
 * ::cudaErrorInvalidResourceHandle is returned. If ::cudaEventRecord() has been
 * called on both events but one or both of them has not yet been completed
 * (that is, ::cudaEventQuery() would return ::cudaErrorNotReady on at least one
 * of the events), ::cudaErrorNotReady is returned. If either event was created
 * with the ::cudaEventDisableTiming flag, then this function will return
 * ::cudaErrorInvalidResourceHandle.
 *
 * \param ms    - Time between \p start and \p end in ms
 * \param start - Starting event
 * \param end   - Ending event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventRecord,
 * ::cuEventElapsedTime
 */
extern __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/** @} */ /* END CUDART_EVENT */

/**
 * \defgroup CUDART_EXTRES_INTEROP External Resource Interoperability
 *
 * ___MANBRIEF___ External resource interoperability functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the external resource interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Imports an external memory object
 *
 * Imports an externally allocated memory object and returns
 * a handle to that in \p extMem_out.
 *
 * The properties of the handle being imported must be described in
 * \p memHandleDesc. The ::cudaExternalMemoryHandleDesc structure
 * is defined as follows:
 *
 * \code
        typedef struct cudaExternalMemoryHandleDesc_st {
            cudaExternalMemoryHandleType type;
            union {
                int fd;
                struct {
                    void *handle;
                    const void *name;
                } win32;
                const void *nvSciBufObject;
            } handle;
            unsigned long long size;
            unsigned int flags;
        } cudaExternalMemoryHandleDesc;
 * \endcode
 *
 * where ::cudaExternalMemoryHandleDesc::type specifies the type
 * of handle being imported. ::cudaExternalMemoryHandleType is
 * defined as:
 *
 * \code
        typedef enum cudaExternalMemoryHandleType_enum {
            cudaExternalMemoryHandleTypeOpaqueFd         = 1,
            cudaExternalMemoryHandleTypeOpaqueWin32      = 2,
            cudaExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
            cudaExternalMemoryHandleTypeD3D12Heap        = 4,
            cudaExternalMemoryHandleTypeD3D12Resource    = 5,
	        cudaExternalMemoryHandleTypeD3D11Resource    = 6,
		    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
            cudaExternalMemoryHandleTypeNvSciBuf         = 8
        } cudaExternalMemoryHandleType;
 * \endcode
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeOpaqueFd, then
 * ::cudaExternalMemoryHandleDesc::handle::fd must be a valid
 * file descriptor referencing a memory object. Ownership of
 * the file descriptor is transferred to the CUDA driver when the
 * handle is imported successfully. Performing any operations on the
 * file descriptor after it is imported results in undefined behavior.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeOpaqueWin32, then exactly one
 * of ::cudaExternalMemoryHandleDesc::handle::win32::handle and
 * ::cudaExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalMemoryHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a memory object. Ownership of this handle is
 * not transferred to CUDA after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::cudaExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a memory object.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt, then
 * ::cudaExternalMemoryHandleDesc::handle::win32::handle must
 * be non-NULL and
 * ::cudaExternalMemoryHandleDesc::handle::win32::name
 * must be NULL. The handle specified must be a globally shared KMT
 * handle. This handle does not hold a reference to the underlying
 * object, and thus will be invalid when all references to the
 * memory object are destroyed.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeD3D12Heap, then exactly one
 * of ::cudaExternalMemoryHandleDesc::handle::win32::handle and
 * ::cudaExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalMemoryHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Heap object. This handle holds a reference to the underlying
 * object. If ::cudaExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D12Heap object.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeD3D12Resource, then exactly one
 * of ::cudaExternalMemoryHandleDesc::handle::win32::handle and
 * ::cudaExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalMemoryHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Resource object. This handle holds a reference to the
 * underlying object. If
 * ::cudaExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D12Resource object.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeD3D11Resource,then exactly one
 * of ::cudaExternalMemoryHandleDesc::handle::win32::handle and
 * ::cudaExternalMemoryHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalMemoryHandleDesc::handle::win32::handle is    
 * not NULL, then it must represent a valid shared NT handle that is  
 * returned by  IDXGIResource1::CreateSharedHandle when referring to a 
 * ID3D11Resource object. If
 * ::cudaExternalMemoryHandleDesc::handle::win32::name
 * is not NULL, then it must point to a NULL-terminated array of
 * UTF-16 characters that refers to a ID3D11Resource object.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt, then
 * ::cudaExternalMemoryHandleDesc::handle::win32::handle must
 * be non-NULL and ::cudaExternalMemoryHandleDesc::handle::win32::name
 * must be NULL. The handle specified must be a valid shared KMT
 * handle that is returned by IDXGIResource::GetSharedHandle when
 * referring to a ID3D11Resource object.
 *
 * If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeNvSciBuf, then
 * ::cudaExternalMemoryHandleDesc::handle::nvSciBufObject must be NON-NULL
 * and reference a valid NvSciBuf object.
 * If the NvSciBuf object imported into CUDA is also mapped by other drivers, then the
 * application must use ::cudaWaitExternalSemaphoresAsync or ::cudaSignalExternalSemaphoresAsync
 * as approprriate barriers to maintain coherence between CUDA and the other drivers.
 * See ::cudaExternalSemaphoreWaitSkipNvSciBufMemSync and ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync 
 * for memory synchronization.
 *
 * The size of the memory object must be specified in
 * ::cudaExternalMemoryHandleDesc::size.
 *
 * Specifying the flag ::cudaExternalMemoryDedicated in
 * ::cudaExternalMemoryHandleDesc::flags indicates that the
 * resource is a dedicated resource. The definition of what a
 * dedicated resource is outside the scope of this extension.
 * This flag must be set if ::cudaExternalMemoryHandleDesc::type
 * is one of the following:
 * ::cudaExternalMemoryHandleTypeD3D12Resource
 * ::cudaExternalMemoryHandleTypeD3D11Resource
 * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
 *
 * \param extMem_out    - Returned handle to an external memory object
 * \param memHandleDesc - Memory import handle descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \note If the Vulkan memory imported into CUDA is mapped on the CPU then the
 * application must use vkInvalidateMappedMemoryRanges/vkFlushMappedMemoryRanges
 * as well as appropriate Vulkan pipeline barriers to maintain coherence between
 * CPU and GPU. For more information on these APIs, please refer to "Synchronization
 * and Cache Control" chapter from Vulkan specification.
 *
 *
 * \sa ::cudaDestroyExternalMemory,
 * ::cudaExternalMemoryGetMappedBuffer,
 * ::cudaExternalMemoryGetMappedMipmappedArray
 */
extern __host__ cudaError_t CUDARTAPI cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc);

/**
 * \brief Maps a buffer onto an imported memory object
 *
 * Maps a buffer onto an imported memory object and returns a device
 * pointer in \p devPtr.
 *
 * The properties of the buffer being mapped must be described in
 * \p bufferDesc. The ::cudaExternalMemoryBufferDesc structure is
 * defined as follows:
 *
 * \code
        typedef struct cudaExternalMemoryBufferDesc_st {
            unsigned long long offset;
            unsigned long long size;
            unsigned int flags;
        } cudaExternalMemoryBufferDesc;
 * \endcode
 *
 * where ::cudaExternalMemoryBufferDesc::offset is the offset in
 * the memory object where the buffer's base address is.
 * ::cudaExternalMemoryBufferDesc::size is the size of the buffer.
 * ::cudaExternalMemoryBufferDesc::flags must be zero.
 *
 * The offset and size have to be suitably aligned to match the
 * requirements of the external API. Mapping two buffers whose ranges
 * overlap may or may not result in the same virtual address being
 * returned for the overlapped portion. In such cases, the application
 * must ensure that all accesses to that region from the GPU are
 * volatile. Otherwise writes made via one address are not guaranteed
 * to be visible via the other address, even if they're issued by the
 * same thread. It is recommended that applications map the combined
 * range instead of mapping separate buffers and then apply the
 * appropriate offsets to the returned pointer to derive the
 * individual buffers.
 *
 * The returned pointer \p devPtr must be freed using ::cudaFree.
 *
 * \param devPtr     - Returned device pointer to buffer
 * \param extMem     - Handle to external memory object
 * \param bufferDesc - Buffer descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaImportExternalMemory,
 * ::cudaDestroyExternalMemory,
 * ::cudaExternalMemoryGetMappedMipmappedArray
 */
extern __host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc);

/**
 * \brief Maps a CUDA mipmapped array onto an external memory object
 *
 * Maps a CUDA mipmapped array onto an external object and returns a
 * handle to it in \p mipmap.
 *
 * The properties of the CUDA mipmapped array being mapped must be
 * described in \p mipmapDesc. The structure
 * ::cudaExternalMemoryMipmappedArrayDesc is defined as follows:
 *
 * \code
        typedef struct cudaExternalMemoryMipmappedArrayDesc_st {
            unsigned long long offset;
            cudaChannelFormatDesc formatDesc;
            cudaExtent extent;
            unsigned int flags;
            unsigned int numLevels;
        } cudaExternalMemoryMipmappedArrayDesc;
 * \endcode
 *
 * where ::cudaExternalMemoryMipmappedArrayDesc::offset is the
 * offset in the memory object where the base level of the mipmap
 * chain is.
 * ::cudaExternalMemoryMipmappedArrayDesc::formatDesc describes the
 * format of the data.
 * ::cudaExternalMemoryMipmappedArrayDesc::extent specifies the
 * dimensions of the base level of the mipmap chain.
 * ::cudaExternalMemoryMipmappedArrayDesc::flags are flags associated
 * with CUDA mipmapped arrays. For further details, please refer to
 * the documentation for ::cudaMalloc3DArray. Note that if the mipmapped
 * array is bound as a color target in the graphics API, then the flag
 * ::cudaArrayColorAttachment must be specified in 
 * ::cudaExternalMemoryMipmappedArrayDesc::flags.
 * ::cudaExternalMemoryMipmappedArrayDesc::numLevels specifies
 * the total number of levels in the mipmap chain.
 *
 * The returned CUDA mipmapped array must be freed using ::cudaFreeMipmappedArray.
 *
 * \param mipmap     - Returned CUDA mipmapped array
 * \param extMem     - Handle to external memory object
 * \param mipmapDesc - CUDA array descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaImportExternalMemory,
 * ::cudaDestroyExternalMemory,
 * ::cudaExternalMemoryGetMappedBuffer
 *
 * \note If ::cudaExternalMemoryHandleDesc::type is
 * ::cudaExternalMemoryHandleTypeNvSciBuf, then
 * ::cudaExternalMemoryMipmappedArrayDesc::numLevels must not be greater than 1.
 */
extern __host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc);

/**
 * \brief Destroys an external memory object.
 *
 * Destroys the specified external memory object. Any existing buffers
 * and CUDA mipmapped arrays mapped onto this object must no longer be
 * used and must be explicitly freed using ::cudaFree and
 * ::cudaFreeMipmappedArray respectively.
 *
 * \param extMem - External memory object to be destroyed
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa ::cudaImportExternalMemory,
 * ::cudaExternalMemoryGetMappedBuffer,
 * ::cudaExternalMemoryGetMappedMipmappedArray
 */
extern __host__ cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t extMem);

/**
 * \brief Imports an external semaphore
 *
 * Imports an externally allocated synchronization object and returns
 * a handle to that in \p extSem_out.
 *
 * The properties of the handle being imported must be described in
 * \p semHandleDesc. The ::cudaExternalSemaphoreHandleDesc is defined
 * as follows:
 *
 * \code
        typedef struct cudaExternalSemaphoreHandleDesc_st {
            cudaExternalSemaphoreHandleType type;
            union {
                int fd;
                struct {
                    void *handle;
                    const void *name;
                } win32;
                const void* NvSciSyncObj;
            } handle;
            unsigned int flags;
        } cudaExternalSemaphoreHandleDesc;
 * \endcode
 *
 * where ::cudaExternalSemaphoreHandleDesc::type specifies the type of
 * handle being imported. ::cudaExternalSemaphoreHandleType is defined
 * as:
 *
 * \code
        typedef enum cudaExternalSemaphoreHandleType_enum {
            cudaExternalSemaphoreHandleTypeOpaqueFd                = 1,
            cudaExternalSemaphoreHandleTypeOpaqueWin32             = 2,
            cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt          = 3,
            cudaExternalSemaphoreHandleTypeD3D12Fence              = 4,
            cudaExternalSemaphoreHandleTypeD3D11Fence              = 5,
            cudaExternalSemaphoreHandleTypeNvSciSync               = 6,
            cudaExternalSemaphoreHandleTypeKeyedMutex              = 7,
            cudaExternalSemaphoreHandleTypeKeyedMutexKmt           = 8,
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd     = 9,
            cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
        } cudaExternalSemaphoreHandleType;
 * \endcode
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeOpaqueFd, then
 * ::cudaExternalSemaphoreHandleDesc::handle::fd must be a valid file
 * descriptor referencing a synchronization object. Ownership of the
 * file descriptor is transferred to the CUDA driver when the handle
 * is imported successfully. Performing any operations on the file
 * descriptor after it is imported results in undefined behavior.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeOpaqueWin32, then exactly one of
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a synchronization object. Ownership of this handle is
 * not transferred to CUDA after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::cudaExternalSemaphoreHandleDesc::handle::win32::name is
 * not NULL, then it must name a valid synchronization object.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, then
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle must be
 * non-NULL and ::cudaExternalSemaphoreHandleDesc::handle::win32::name
 * must be NULL. The handle specified must be a globally shared KMT
 * handle. This handle does not hold a reference to the underlying
 * object, and thus will be invalid when all references to the
 * synchronization object are destroyed.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeD3D12Fence, then exactly one of
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D12Device::CreateSharedHandle when referring to a
 * ID3D12Fence object. This handle holds a reference to the underlying
 * object. If ::cudaExternalSemaphoreHandleDesc::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid ID3D12Fence object.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeD3D11Fence, then exactly one of
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * is returned by ID3D11Fence::CreateSharedHandle. If 
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::name
 * is not NULL, then it must name a valid synchronization object that
 * refers to a valid ID3D11Fence object.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeNvSciSync, then
 * ::cudaExternalSemaphoreHandleDesc::handle::nvSciSyncObj
 * represents a valid NvSciSyncObj.
 *
 * ::cudaExternalSemaphoreHandleTypeKeyedMutex, then exactly one of
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it represent a valid shared NT handle that
 * is returned by IDXGIResource1::CreateSharedHandle when referring to
 * a IDXGIKeyedMutex object.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt, then
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle must be
 * non-NULL and ::cudaExternalSemaphoreHandleDesc::handle::win32::name
 * must be NULL. The handle specified must represent a valid KMT
 * handle that is returned by IDXGIResource::GetSharedHandle when
 * referring to a IDXGIKeyedMutex object.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd, then
 * ::cudaExternalSemaphoreHandleDesc::handle::fd must be a valid file
 * descriptor referencing a synchronization object. Ownership of the
 * file descriptor is transferred to the CUDA driver when the handle
 * is imported successfully. Performing any operations on the file
 * descriptor after it is imported results in undefined behavior.
 *
 * If ::cudaExternalSemaphoreHandleDesc::type is
 * ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32, then exactly one of
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::handle and
 * ::cudaExternalSemaphoreHandleDesc::handle::win32::name must not be
 * NULL. If ::cudaExternalSemaphoreHandleDesc::handle::win32::handle
 * is not NULL, then it must represent a valid shared NT handle that
 * references a synchronization object. Ownership of this handle is
 * not transferred to CUDA after the import operation, so the
 * application must release the handle using the appropriate system
 * call. If ::cudaExternalSemaphoreHandleDesc::handle::win32::name is
 * not NULL, then it must name a valid synchronization object.
 *
 * \param extSem_out    - Returned handle to an external semaphore
 * \param semHandleDesc - Semaphore import handle descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDestroyExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc);

/**
 * \brief Signals a set of external semaphore objects
 *
 * Enqueues a signal operation on a set of externally allocated
 * semaphore object in the specified stream. The operations will be
 * executed when all prior operations in the stream complete.
 *
 * The exact semantics of signaling a semaphore depends on the type of
 * the object.
 *
 * If the semaphore object is any one of the following types:
 * ::cudaExternalSemaphoreHandleTypeOpaqueFd,
 * ::cudaExternalSemaphoreHandleTypeOpaqueWin32,
 * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
 * then signaling the semaphore will set it to the signaled state.
 *
 * If the semaphore object is any one of the following types:
 * ::cudaExternalSemaphoreHandleTypeD3D12Fence,
 * ::cudaExternalSemaphoreHandleTypeD3D11Fence,
 * ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
 * ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
 * then the semaphore will be set to the value specified in
 * ::cudaExternalSemaphoreSignalParams::params::fence::value.
 *
 * If the semaphore object is of the type ::cudaExternalSemaphoreHandleTypeNvSciSync
 * this API sets ::cudaExternalSemaphoreSignalParams::params::nvSciSync::fence to a
 * value that can be used by subsequent waiters of the same NvSciSync object to
 * order operations with those currently submitted in \p stream. Such an update
 * will overwrite previous contents of
 * ::cudaExternalSemaphoreSignalParams::params::nvSciSync::fence. By deefault,
 * signaling such an external semaphore object causes appropriate memory synchronization
 * operations to be performed over all the external memory objects that are imported as
 * ::cudaExternalMemoryHandleTypeNvSciBuf. This ensures that any subsequent accesses
 * made by other importers of the same set of NvSciBuf memory object(s) are coherent.
 * These operations can be skipped by specifying the flag
 * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync, which can be used as a
 * performance optimization when data coherency is not required. But specifying this
 * flag in scenarios where data coherency is required results in undefined behavior.
 * Also, for semaphore object of the type ::cudaExternalSemaphoreHandleTypeNvSciSync,
 * if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
 * ::cudaDeviceGetNvSciSyncAttributes to cudaNvSciSyncAttrSignal, this API will return
 * cudaErrorNotSupported.
 *
 * If the semaphore object is any one of the following types:
 * ::cudaExternalSemaphoreHandleTypeKeyedMutex,
 * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt,
 * then the keyed mutex will be released with the key specified in
 * ::cudaExternalSemaphoreSignalParams::params::keyedmutex::key.
 *
 * \param extSemArray - Set of external semaphores to be signaled
 * \param paramsArray - Array of semaphore parameters
 * \param numExtSems  - Number of semaphores to signal
 * \param stream     - Stream to enqueue the signal operations in
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaImportExternalSemaphore,
 * ::cudaDestroyExternalSemaphore,
 * ::cudaWaitExternalSemaphoresAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));

/**
 * \brief Waits on a set of external semaphore objects
 *
 * Enqueues a wait operation on a set of externally allocated
 * semaphore object in the specified stream. The operations will be
 * executed when all prior operations in the stream complete.
 *
 * The exact semantics of waiting on a semaphore depends on the type
 * of the object.
 *
 * If the semaphore object is any one of the following types:
 * ::cudaExternalSemaphoreHandleTypeOpaqueFd,
 * ::cudaExternalSemaphoreHandleTypeOpaqueWin32,
 * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
 * then waiting on the semaphore will wait until the semaphore reaches
 * the signaled state. The semaphore will then be reset to the
 * unsignaled state. Therefore for every signal operation, there can
 * only be one wait operation.
 *
 * If the semaphore object is any one of the following types:
 * ::cudaExternalSemaphoreHandleTypeD3D12Fence,
 * ::cudaExternalSemaphoreHandleTypeD3D11Fence,
 * ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
 * ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
 * then waiting on the semaphore will wait until the value of the
 * semaphore is greater than or equal to
 * ::cudaExternalSemaphoreWaitParams::params::fence::value.
 *
 * If the semaphore object is of the type ::cudaExternalSemaphoreHandleTypeNvSciSync
 * then, waiting on the semaphore will wait until the
 * ::cudaExternalSemaphoreSignalParams::params::nvSciSync::fence is signaled by the
 * signaler of the NvSciSyncObj that was associated with this semaphore object.
 * By default, waiting on such an external semaphore object causes appropriate
 * memory synchronization operations to be performed over all external memory objects
 * that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf. This ensures that
 * any subsequent accesses made by other importers of the same set of NvSciBuf memory
 * object(s) are coherent. These operations can be skipped by specifying the flag
 * ::cudaExternalSemaphoreWaitSkipNvSciBufMemSync, which can be used as a
 * performance optimization when data coherency is not required. But specifying this
 * flag in scenarios where data coherency is required results in undefined behavior.
 * Also, for semaphore object of the type ::cudaExternalSemaphoreHandleTypeNvSciSync,
 * if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in
 * ::cudaDeviceGetNvSciSyncAttributes to cudaNvSciSyncAttrWait, this API will return
 * cudaErrorNotSupported.
 *
 * If the semaphore object is any one of the following types:
 * ::cudaExternalSemaphoreHandleTypeKeyedMutex,
 * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt,
 * then the keyed mutex will be acquired when it is released with the key specified 
 * in ::cudaExternalSemaphoreSignalParams::params::keyedmutex::key or
 * until the timeout specified by
 * ::cudaExternalSemaphoreSignalParams::params::keyedmutex::timeoutMs
 * has lapsed. The timeout interval can either be a finite value
 * specified in milliseconds or an infinite value. In case an infinite
 * value is specified the timeout never elapses. The windows INFINITE
 * macro must be used to specify infinite timeout
 *
 * \param extSemArray - External semaphores to be waited on
 * \param paramsArray - Array of semaphore parameters
 * \param numExtSems  - Number of semaphores to wait on
 * \param stream      - Stream to enqueue the wait operations in
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * ::cudaErrorTimeout
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaImportExternalSemaphore,
 * ::cudaDestroyExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));

/**
 * \brief Destroys an external semaphore
 *
 * Destroys an external semaphore object and releases any references
 * to the underlying resource. Any outstanding signals or waits must
 * have completed before the semaphore is destroyed.
 *
 * \param extSem - External semaphore to be destroyed
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa ::cudaImportExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);

/** @} */ /* END CUDART_EXTRES_INTEROP */

/**
 * \defgroup CUDART_EXECUTION Execution Control
 *
 * ___MANBRIEF___ execution control functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the execution control functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Launches a device function
 *
 * The function invokes kernel \p func on \p gridDim (\p gridDim.x &times; \p gridDim.y
 * &times; \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x &times;
 * \p blockDim.y &times; \p blockDim.z) threads.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory
 * \param stream      - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorSharedObjectInitFailed,
 * ::cudaErrorInvalidPtx,
 * ::cudaErrorUnsupportedPtxVersion,
 * ::cudaErrorNoKernelImageForDevice,
 * ::cudaErrorJitCompilerNotFound,
 * ::cudaErrorJitCompilationDisabled
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * ::cuLaunchKernel
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);

/**
 * \brief Launches a device function where thread blocks can cooperate and synchronize as they execute
 *
 * The function invokes kernel \p func on \p gridDim (\p gridDim.x &times; \p gridDim.y
 * &times; \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x &times;
 * \p blockDim.y &times; \p blockDim.z) threads.
 *
 * The device on which this kernel is invoked must have a non-zero value for
 * the device attribute ::cudaDevAttrCooperativeLaunch.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
 *
 * The kernel cannot make use of CUDA dynamic parallelism.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory
 * \param stream      - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorCooperativeLaunchTooLarge,
 * ::cudaErrorSharedObjectInitFailed
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaLaunchCooperativeKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchCooperativeKernel (C++ API)",
 * ::cudaLaunchCooperativeKernelMultiDevice,
 * ::cuLaunchCooperativeKernel
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);

/**
 * \brief Launches device functions on multiple devices where thread blocks can cooperate and synchronize as they execute
 *
 * \deprecated This function is deprecated as of CUDA 11.3.
 *
 * Invokes kernels as specified in the \p launchParamsList array where each element
 * of the array specifies all the parameters required to perform a single kernel launch.
 * These kernels can cooperate and synchronize as they execute. The size of the array is
 * specified by \p numDevices.
 *
 * No two kernels can be launched on the same device. All the devices targeted by this
 * multi-device launch must be identical. All devices must have a non-zero value for the
 * device attribute ::cudaDevAttrCooperativeMultiDeviceLaunch.
 *
 * The same kernel must be launched on all devices. Note that any __device__ or __constant__
 * variables are independently instantiated on every device. It is the application's
 * responsiblity to ensure these variables are initialized and used appropriately.
 *
 * The size of the grids as specified in blocks, the size of the blocks themselves and the
 * amount of shared memory used by each thread block must also match across all launched kernels.
 *
 * The streams used to launch these kernels must have been created via either ::cudaStreamCreate
 * or ::cudaStreamCreateWithPriority or ::cudaStreamCreateWithPriority. The NULL stream or
 * ::cudaStreamLegacy or ::cudaStreamPerThread cannot be used.
 *
 * The total number of blocks launched per kernel cannot exceed the maximum number of blocks
 * per multiprocessor as returned by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::cudaDevAttrMultiProcessorCount. Since the
 * total number of blocks launched per device has to match across all devices, the maximum
 * number of blocks that can be launched per device will be limited by the device with the
 * least number of multiprocessors.
 *
 * The kernel cannot make use of CUDA dynamic parallelism.
 *
 * The ::cudaLaunchParams structure is defined as:
 * \code
        struct cudaLaunchParams
        {
            void *func;
            dim3 gridDim;
            dim3 blockDim;
            void **args;
            size_t sharedMem;
            cudaStream_t stream;
        };
 * \endcode
 * where:
 * - ::cudaLaunchParams::func specifies the kernel to be launched. This same functions must
 *   be launched on all devices. For templated functions, pass the function symbol as follows:
 *   func_name<template_arg_0,...,template_arg_N>
 * - ::cudaLaunchParams::gridDim specifies the width, height and depth of the grid in blocks.
 *   This must match across all kernels launched.
 * - ::cudaLaunchParams::blockDim is the width, height and depth of each thread block. This
 *   must match across all kernels launched.
 * - ::cudaLaunchParams::args specifies the arguments to the kernel. If the kernel has
 *   N parameters then ::cudaLaunchParams::args should point to array of N pointers. Each
 *   pointer, from <tt>::cudaLaunchParams::args[0]</tt> to <tt>::cudaLaunchParams::args[N - 1]</tt>,
 *   point to the region of memory from which the actual parameter will be copied.
 * - ::cudaLaunchParams::sharedMem is the dynamic shared-memory size per thread block in bytes.
 *   This must match across all kernels launched.
 * - ::cudaLaunchParams::stream is the handle to the stream to perform the launch in. This cannot
 *   be the NULL stream or ::cudaStreamLegacy or ::cudaStreamPerThread.
 *
 * By default, the kernel won't begin execution on any GPU until all prior work in all the specified
 * streams has completed. This behavior can be overridden by specifying the flag
 * ::cudaCooperativeLaunchMultiDeviceNoPreSync. When this flag is specified, each kernel
 * will only wait for prior work in the stream corresponding to that GPU to complete before it begins
 * execution.
 *
 * Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
 * execution until the kernels on all GPUs have completed. This behavior can be overridden by specifying
 * the flag ::cudaCooperativeLaunchMultiDeviceNoPostSync. When this flag is specified,
 * any subsequent work pushed in any of the specified streams will only wait for the kernel launched
 * on the GPU corresponding to that stream to complete before it begins execution.
 *
 * \param launchParamsList - List of launch parameters, one per device
 * \param numDevices       - Size of the \p launchParamsList array
 * \param flags            - Flags to control launch behavior
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorCooperativeLaunchTooLarge,
 * ::cudaErrorSharedObjectInitFailed
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaLaunchCooperativeKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchCooperativeKernel (C++ API)",
 * ::cudaLaunchCooperativeKernel,
 * ::cuLaunchCooperativeKernelMultiDevice
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  __dv(0));

/**
 * \brief Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache configuration
 * for the function specified via \p func. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute \p func.
 *
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. If the specified function does not exist,
 * then ::cudaErrorInvalidDeviceFunction is returned. For templated functions,
 * pass the function symbol as follows: func_name<template_arg_0,...,template_arg_N>
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 * - ::cudaFuncCachePreferEqual: prefer equal size L1 cache and shared memory
 *
 * \param func        - Device function symbol
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 * \note_init_rt
 * \note_callback
 *
 * \sa 
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * \ref ::cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C API)",
 * ::cudaThreadGetCacheConfig,
 * ::cudaThreadSetCacheConfig,
 * ::cuFuncSetCacheConfig
 */
extern __host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);

/**
 * \brief Sets the shared memory configuration for a device function
 *
 * On devices with configurable shared memory banks, this function will 
 * force all subsequent launches of the specified device function to have
 * the given shared memory bank size configuration. On any given launch of the
 * function, the shared memory configuration of the device will be temporarily
 * changed if needed to suit the function's preferred configuration. Changes in
 * shared memory configuration between subsequent launches of functions, 
 * may introduce a device side synchronization point.
 *
 * Any per-function setting of shared memory bank size set via 
 * ::cudaFuncSetSharedMemConfig will override the device wide setting set by
 * ::cudaDeviceSetSharedMemConfig.
 *
 * Changing the shared memory bank size will not increase shared memory usage
 * or affect occupancy of kernels, but may have major effects on performance. 
 * Larger bank sizes will allow for greater potential bandwidth to shared memory,
 * but will change what kinds of accesses to shared memory will result in bank 
 * conflicts.
 *
 * This function will do nothing on devices with fixed shared memory bank size.
 *
 * For templated functions, pass the function symbol as follows:
 * func_name<template_arg_0,...,template_arg_N>
 *
 * The supported bank configurations are:
 * - ::cudaSharedMemBankSizeDefault: use the device's shared memory configuration
 *   when launching this function.
 * - ::cudaSharedMemBankSizeFourByte: set shared memory bank width to be 
 *   four bytes natively when launching this function.
 * - ::cudaSharedMemBankSizeEightByte: set shared memory bank width to be eight 
 *   bytes natively when launching this function.
 *
 * \param func   - Device function symbol
 * \param config - Requested shared memory configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_string_api_deprecation2
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceSetSharedMemConfig,
 * ::cudaDeviceGetSharedMemConfig,
 * ::cudaDeviceSetCacheConfig,
 * ::cudaDeviceGetCacheConfig,
 * ::cudaFuncSetCacheConfig,
 * ::cuFuncSetSharedMemConfig
 */
extern __host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config);

/**
 * \brief Find out attributes for a given function
 *
 * This function obtains the attributes of a function specified via \p func.
 * \p func is a device function symbol and must be declared as a
 * \c __global__ function. The fetched attributes are placed in \p attr.
 * If the specified function does not exist, then
 * ::cudaErrorInvalidDeviceFunction is returned. For templated functions, pass
 * the function symbol as follows: func_name<template_arg_0,...,template_arg_N>
 *
 * Note that some function attributes such as
 * \ref ::cudaFuncAttributes::maxThreadsPerBlock "maxThreadsPerBlock"
 * may vary based on the device that is currently being used.
 *
 * \param attr - Return pointer to function's attributes
 * \param func - Device function symbol
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction
 * \notefnerr
 * \note_string_api_deprecation2
 * \note_init_rt
 * \note_callback
 *
 * \sa 
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, T*) "cudaFuncGetAttributes (C++ API)",
 * \ref ::cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C API)",
 * ::cuFuncGetAttribute
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);


/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p func.
 * The parameter \p func must be a pointer to a function that executes
 * on the device. The parameter specified by \p func must be declared as a \p __global__
 * function. The enumeration defined by \p attr is set to the value defined by \p value.
 * If the specified function does not exist, then ::cudaErrorInvalidDeviceFunction is returned.
 * If the specified attribute cannot be written, or if the value is incorrect, 
 * then ::cudaErrorInvalidValue is returned.
 *
 * Valid values for \p attr are:
 * - ::cudaFuncAttributeMaxDynamicSharedMemorySize - The requested maximum size in bytes of dynamically-allocated shared memory. The sum of this value and the function attribute ::sharedSizeBytes
 *   cannot exceed the device attribute ::cudaDevAttrMaxSharedMemoryPerBlockOptin. The maximal size of requestable dynamic shared memory may differ by GPU architecture.
 * - ::cudaFuncAttributePreferredSharedMemoryCarveout - On devices where the L1 cache and shared memory use the same hardware resources, 
 *   this sets the shared memory carveout preference, in percent of the total shared memory. See ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
 *   This is only a hint, and the driver can choose a different ratio if required to execute the function.
 *
 * \param func  - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);



/**
 * \brief Converts a double argument to be executed on a device
 *
 * \param d - Double to convert
 *
 * \deprecated This function is deprecated as of CUDA 7.5
 *
 * Converts the double value of \p d to an internal float representation if
 * the device does not support double arithmetic. If the device does natively
 * support doubles, then this function does nothing.
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForHost
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d);

/**
 * \brief Converts a double argument after execution on a device
 *
 * \deprecated This function is deprecated as of CUDA 7.5
 *
 * Converts the double value of \p d from a potentially internal float
 * representation if the device does not support double arithmetic. If the
 * device does natively support doubles, then this function does nothing.
 *
 * \param d - Double to convert
 *
 * \return
 * ::cudaSuccess
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice
 */
extern __CUDA_DEPRECATED  __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d);



/**
 * \brief Enqueues a host function call in a stream
 *
 * Enqueues a host function to run in a stream.  The function will be called
 * after currently enqueued work and will block work added after it.
 *
 * The host function must not make any CUDA API calls.  Attempting to use a
 * CUDA API may result in ::cudaErrorNotPermitted, but this is not required.
 * The host function must not perform any synchronization that may depend on
 * outstanding CUDA work not mandated to run earlier.  Host functions without a
 * mandated order (such as in independent streams) execute in undefined order
 * and may be serialized.
 *
 * For the purposes of Unified Memory, execution makes a number of guarantees:
 * <ul>
 *   <li>The stream is considered idle for the duration of the function's
 *   execution.  Thus, for example, the function may always use memory attached
 *   to the stream it was enqueued in.</li>
 *   <li>The start of execution of the function has the same effect as
 *   synchronizing an event recorded in the same stream immediately prior to
 *   the function.  It thus synchronizes streams which have been "joined"
 *   prior to the function.</li>
 *   <li>Adding device work to any stream does not have the effect of making
 *   the stream active until all preceding host functions and stream callbacks
 *   have executed.  Thus, for
 *   example, a function might use global attached memory even if work has
 *   been added to another stream, if the work has been ordered behind the
 *   function call with an event.</li>
 *   <li>Completion of the function does not cause a stream to become
 *   active except as described above.  The stream will remain idle
 *   if no device work follows the function, and will remain idle across
 *   consecutive host functions or stream callbacks without device work in
 *   between.  Thus, for example,
 *   stream synchronization can be done by signaling from a host function at the
 *   end of the stream.</li>
 * </ul>
 *
 * Note that, in constrast to ::cuStreamAddCallback, the function will not be
 * called in the event of an error in the CUDA context.
 *
 * \param hStream  - Stream to enqueue function call in
 * \param fn       - The function to call once preceding stream operations are complete
 * \param userData - User-specified data to be passed to the function
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate,
 * ::cudaStreamQuery,
 * ::cudaStreamSynchronize,
 * ::cudaStreamWaitEvent,
 * ::cudaStreamDestroy,
 * ::cudaMallocManaged,
 * ::cudaStreamAttachMemAsync,
 * ::cudaStreamAddCallback,
 * ::cuLaunchHostFunc
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);

/** @} */ /* END CUDART_EXECUTION */

/**
 * \defgroup CUDART_OCCUPANCY Occupancy
 *
 * ___MANBRIEF___ occupancy calculation functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the occupancy calculation functions of the CUDA runtime
 * application programming interface.
 *
 * Besides the occupancy calculator functions
 * (\ref ::cudaOccupancyMaxActiveBlocksPerMultiprocessor and \ref ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags),
 * there are also C++ only occupancy-based launch configuration functions documented in
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * See
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)"
 * \ref ::cudaOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "cudaOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 *
 * @{
 */

/**
 * \brief Returns occupancy for a device function
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calculated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * \ref ::cudaOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "cudaOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessor
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);

/**
 * \brief Returns dynamic shared memory available per block when launching \p numBlocks blocks on SM.
 *
 * Returns in \p *dynamicSmemSize the maximum size of dynamic shared memory to allow \p numBlocks blocks per SM. 
 *
 * \param dynamicSmemSize - Returned maximum dynamic shared memory 
 * \param func            - Kernel function for which occupancy is calculated
 * \param numBlocks       - Number of blocks to fit on SM 
 * \param blockSize       - Size of the block
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * ::cudaOccupancyAvailableDynamicSMemPerBlock
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize);

/**
 * \brief Returns occupancy for a device function with the specified flags
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * The \p flags parameter controls how special cases are handled. Valid flags include:
 *
 * - ::cudaOccupancyDefault: keeps the default behavior as
 *   ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *
 * - ::cudaOccupancyDisableCachingOverride: This flag suppresses the default behavior
 *   on platform where global caching affects occupancy. On such platforms, if caching
 *   is enabled, but per-block SM resource usage would result in zero occupancy, the
 *   occupancy calculator will calculate the occupancy as if caching is disabled.
 *   Setting this flag makes the occupancy calculator to return 0 in such cases.
 *   More information can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calculated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param flags           - Requested behavior for the occupancy calculator
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor,
 * \ref ::cudaOccupancyMaxPotentialBlockSize(int*, int*, T, size_t, int) "cudaOccupancyMaxPotentialBlockSize (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, T, size_t, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeWithFlags (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMem(int*, int*, T, UnaryFunction, int) "cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API)",
 * \ref ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int*, int*, T, UnaryFunction, int, unsigned int) "cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags (C++ API)",
 * \ref ::cudaOccupancyAvailableDynamicSMemPerBlock(size_t*, T, int, int) "cudaOccupancyAvailableDynamicSMemPerBlock (C++ API)",
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);

/** @} */ /* END CUDA_OCCUPANCY */

/**
 * \defgroup CUDART_MEMORY Memory Management
 *
 * ___MANBRIEF___ memory management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the memory management functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p size bytes of managed memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::cudaErrorNotSupported is returned. Support
 * for managed memory can be queried using the device attribute
 * ::cudaDevAttrManagedMemory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p size
 * is 0, ::cudaMallocManaged returns ::cudaErrorInvalidValue. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::cudaMemAttachGlobal or ::cudaMemAttachHost. The
 * default value for \p flags is ::cudaMemAttachGlobal.
 * If ::cudaMemAttachGlobal is specified, then this memory is accessible from
 * any stream on any device. If ::cudaMemAttachHost is specified, then the
 * allocation should not be accessed from devices that have a zero value for the
 * device attribute ::cudaDevAttrConcurrentManagedAccess; an explicit call to
 * ::cudaStreamAttachMemAsync will be required to enable access on such devices.
 *
 * If the association is later changed via ::cudaStreamAttachMemAsync to
 * a single stream, the default association, as specifed during ::cudaMallocManaged,
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::cudaMemAttachGlobal. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::cudaMallocManaged should be released with ::cudaFree.
 *
 * Device memory oversubscription is possible for GPUs that have a non-zero value for the
 * device attribute ::cudaDevAttrConcurrentManagedAccess. Managed memory on
 * such GPUs may be evicted from device memory to host memory at any time by the Unified
 * Memory driver in order to make room for other allocations.
 *
 * In a multi-GPU system where all GPUs have a non-zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess, managed memory may not be populated when this
 * API returns and instead may be populated on access. In such systems, managed memory can
 * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
 * maintain data locality and prevent excessive page faults to the extent possible. The application
 * can also guide the driver about memory usage patterns via ::cudaMemAdvise. The application
 * can also explicitly migrate memory to a desired processor's memory via
 * ::cudaMemPrefetchAsync.
 *
 * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess and all the GPUs have peer-to-peer support
 * with each other, the physical storage for managed memory is created on the GPU which is active
 * at the time ::cudaMallocManaged is called. All other GPUs will reference the data at reduced
 * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
 * memory among such GPUs.
 *
 * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
 * where the value of the device attribute ::cudaDevAttrConcurrentManagedAccess
 * is zero for at least one of those GPUs, the location chosen for physical storage of managed
 * memory is system-dependent.
 * - On Linux, the location chosen will be device memory as long as the current set of active
 * contexts are on devices that either have peer-to-peer support with each other or have a
 * non-zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess.
 * If there is an active context on a GPU that does not have a non-zero value for that device
 * attribute and it does not have peer-to-peer support with the other devices that have active
 * contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
 * Note that this means that managed memory that is located in device memory is migrated to
 * host memory if a new context is created on a GPU that doesn't have a non-zero value for
 * the device attribute and does not support peer-to-peer with at least one of the other devices
 * that has an active context. This in turn implies that context creation may fail if there is
 * insufficient host memory to migrate all managed allocations.
 * - On Windows, the physical storage is always created in 'zero-copy' or host memory.
 * All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
 * circumstances, use of the environment variable CUDA_VISIBLE_DEVICES is recommended to
 * restrict CUDA to only use those GPUs that have peer-to-peer support.
 * Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
 * value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all devices used in
 * that process that support managed memory have to be peer-to-peer compatible
 * with each other. The error ::cudaErrorInvalidDevice will be returned if a device
 * that supports managed memory is used and it is not peer-to-peer compatible with
 * any of the other managed memory supporting devices that were previously used in
 * that process, even if ::cudaDeviceReset has been called on those devices. These
 * environment variables are described in the CUDA programming guide under the
 * "CUDA environment variables" section.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 * \param flags  - Must be either ::cudaMemAttachGlobal or ::cudaMemAttachHost (defaults to ::cudaMemAttachGlobal)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * ::cudaMalloc3D, ::cudaMalloc3DArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc, ::cudaDeviceGetAttribute, ::cudaStreamAttachMemAsync,
 * ::cuMemAllocManaged
 */
#if defined(__cplusplus)
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
#else
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
#endif

/**
 * \brief Allocate memory on the device
 *
 * Allocates \p size bytes of linear memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. The allocated memory is
 * suitably aligned for any kind of variable. The memory is not cleared.
 * ::cudaMalloc() returns ::cudaErrorMemoryAllocation in case of failure.
 *
 * The device version of ::cudaFree cannot be used with a \p *devPtr
 * allocated using the host API, and vice versa.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * ::cudaMalloc3D, ::cudaMalloc3DArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::cuMemAlloc
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);

/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::cudaMemcpy*(). Since the memory can be accessed directly by the device,
 * it can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * memory with ::cudaMallocHost() may degrade system performance, since it
 * reduces the amount of memory available to the system for paging. As a
 * result, this function is best used sparingly to allocate staging areas for
 * data exchange between host and device.
 *
 * \param ptr  - Pointer to allocated host memory
 * \param size - Requested allocation size in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaMallocArray, ::cudaMalloc3D,
 * ::cudaMalloc3DArray, ::cudaHostAlloc, ::cudaFree, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t, unsigned int) "cudaMallocHost (C++ API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::cuMemAllocHost
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size);

/**
 * \brief Allocates pitched memory on the device
 *
 * Allocates at least \p width (in bytes) * \p height bytes of linear memory
 * on the device and returns in \p *devPtr a pointer to the allocated memory.
 * The function may pad the allocation to ensure that corresponding pointers
 * in any given row will continue to meet the alignment requirements for
 * coalescing as the address is updated from row to row. The pitch returned in
 * \p *pitch by ::cudaMallocPitch() is the width in bytes of the allocation.
 * The intended usage of \p pitch is as a separate parameter of the allocation,
 * used to compute addresses within the 2D array. Given the row and column of
 * an array element of type \p T, the address is computed as:
 * \code
    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
   \endcode
 *
 * For allocations of 2D arrays, it is recommended that programmers consider
 * performing pitch allocations using ::cudaMallocPitch(). Due to pitch
 * alignment restrictions in the hardware, this is especially true if the
 * application will be performing 2D memory copies between different regions
 * of device memory (whether linear memory or CUDA arrays).
 *
 * \param devPtr - Pointer to allocated pitched device memory
 * \param pitch  - Pitch for allocation
 * \param width  - Requested pitched allocation width (in bytes)
 * \param height - Requested pitched allocation height
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaMalloc3D, ::cudaMalloc3DArray,
 * ::cudaHostAlloc,
 * ::cuMemAllocPitch
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a CUDA array according to the ::cudaChannelFormatDesc structure
 * \p desc and returns a handle to the new CUDA array in \p *array.
 *
 * The ::cudaChannelFormatDesc is defined as:
 * \code
    struct cudaChannelFormatDesc {
        int x, y, z, w;
    enum cudaChannelFormatKind f;
    };
    \endcode
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::cudaArraySurfaceLoadStore: Allocates an array that can be read from or written to using a surface reference
 * - ::cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the array.
 * - ::cudaArraySparse: Allocates a CUDA array without physical backing memory. The subregions within this sparse array
 *   can later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. 
 *   The physical backing memory must be allocated via ::cuMemCreate.

 * - ::cudaArrayDeferredMapping: Allocates a CUDA array without physical backing memory. The entire array can 
 *   later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. 
 *   The physical backing memory must be allocated via ::cuMemCreate.

 *
 * \p width and \p height must meet certain size requirements. See ::cudaMalloc3DArray() for more details.
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param width  - Requested array allocation width
 * \param height - Requested array allocation height
 * \param flags  - Requested properties of allocated array
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaMalloc3D, ::cudaMalloc3DArray,
 * ::cudaHostAlloc,
 * ::cuArrayCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));

/**
 * \brief Frees memory on the device
 *
 * Frees the memory space pointed to by \p devPtr, which must have been
 * returned by a previous call to ::cudaMalloc() or ::cudaMallocPitch().
 * Otherwise, or if ::cudaFree(\p devPtr) has already been called before,
 * an error is returned. If \p devPtr is 0, no operation is performed.
 * ::cudaFree() returns ::cudaErrorValue in case of failure.
 *
 * The device version of ::cudaFree cannot be used with a \p *devPtr
 * allocated using the host API, and vice versa.
 *
 * \param devPtr - Device pointer to memory to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaMallocArray, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaMalloc3D, ::cudaMalloc3DArray,
 * ::cudaHostAlloc,
 * ::cuMemFree
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);

/**
 * \brief Frees page-locked memory
 *
 * Frees the memory space pointed to by \p hostPtr, which must have been
 * returned by a previous call to ::cudaMallocHost() or ::cudaHostAlloc().
 *
 * \param ptr - Pointer to memory to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaMalloc3D, ::cudaMalloc3DArray, ::cudaHostAlloc,
 * ::cuMemFreeHost
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr);

/**
 * \brief Frees an array on the device
 *
 * Frees the CUDA array \p array, which must have been returned by a
 * previous call to ::cudaMallocArray(). If \p devPtr is 0,
 * no operation is performed.
 *
 * \param array - Pointer to array to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::cuArrayDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array);

/**
 * \brief Frees a mipmapped array on the device
 *
 * Frees the CUDA mipmapped array \p mipmappedArray, which must have been 
 * returned by a previous call to ::cudaMallocMipmappedArray(). If \p devPtr
 * is 0, no operation is performed.
 *
 * \param mipmappedArray - Pointer to mipmapped array to free
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc, ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::cuMipmappedArrayDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);


/**
 * \brief Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::cudaMemcpy(). Since the memory can be accessed directly by the device, it
 * can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between host
 * and device.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaHostAllocDefault: This flag's value is defined to be 0 and causes
 * ::cudaHostAlloc() to emulate ::cudaMallocHost().
 * - ::cudaHostAllocPortable: The memory returned by this call will be
 * considered as pinned memory by all CUDA contexts, not just the one that
 * performed the allocation.
 * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
 * The device pointer to the memory may be obtained by calling
 * ::cudaHostGetDevicePointer().
 * - ::cudaHostAllocWriteCombined: Allocates the memory as write-combined (WC).
 * WC memory can be transferred across the PCI Express bus more quickly on some
 * system configurations, but cannot be read efficiently by most CPUs.  WC
 * memory is a good option for buffers that will be written by the CPU and read
 * by the device via mapped pinned memory or host->device transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * In order for the ::cudaHostAllocMapped flag to have any effect, the CUDA context
 * must support the ::cudaDeviceMapHost flag, which can be checked via
 * ::cudaGetDeviceFlags(). The ::cudaDeviceMapHost flag is implicitly set for
 * contexts created via the runtime API.
 *
 * The ::cudaHostAllocMapped flag may be specified on CUDA contexts for devices
 * that do not support mapped pinned memory. The failure is deferred to
 * ::cudaHostGetDevicePointer() because the memory may be mapped into other
 * CUDA contexts via the ::cudaHostAllocPortable flag.
 *
 * Memory allocated by this function must be freed with ::cudaFreeHost().
 *
 * \param pHost - Device pointer to allocated memory
 * \param size  - Requested allocation size in bytes
 * \param flags - Requested properties of allocated memory
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaSetDeviceFlags,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost,
 * ::cudaGetDeviceFlags,
 * ::cuMemHostAlloc
 */
extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags);

/**
 * \brief Registers an existing host memory range for use by CUDA
 *
 * Page-locks the memory range specified by \p ptr and \p size and maps it
 * for the device(s) as specified by \p flags. This memory range also is added
 * to the same tracking mechanism as ::cudaHostAlloc() to automatically accelerate
 * calls to functions such as ::cudaMemcpy(). Since the memory can be accessed 
 * directly by the device, it can be read or written with much higher bandwidth 
 * than pageable memory that has not been registered.  Page-locking excessive
 * amounts of memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to register staging areas for data exchange between
 * host and device.
 *
 * ::cudaHostRegister is supported only on I/O coherent devices that have a non-zero
 * value for the device attribute ::cudaDevAttrHostRegisterSupported.
 *
 * The \p flags parameter enables different options to be specified that
 * affect the allocation, as follows.
 *
 * - ::cudaHostRegisterDefault: On a system with unified virtual addressing,
 *   the memory will be both mapped and portable.  On a system with no unified
 *   virtual addressing, the memory will be neither mapped nor portable.
 *
 * - ::cudaHostRegisterPortable: The memory returned by this call will be
 *   considered as pinned memory by all CUDA contexts, not just the one that
 *   performed the allocation.
 *
 * - ::cudaHostRegisterMapped: Maps the allocation into the CUDA address
 *   space. The device pointer to the memory may be obtained by calling
 *   ::cudaHostGetDevicePointer().
 *
 * - ::cudaHostRegisterIoMemory: The passed memory pointer is treated as
 *   pointing to some memory-mapped I/O space, e.g. belonging to a
 *   third-party PCIe device, and it will marked as non cache-coherent and
 *   contiguous.
 *
 * - ::cudaHostRegisterReadOnly: The passed memory pointer is treated as
 *   pointing to memory that is considered read-only by the device.  On
 *   platforms without ::cudaDevAttrPageableMemoryAccessUsesHostPageTables, this
 *   flag is required in order to register memory mapped to the CPU as
 *   read-only.  Support for the use of this flag can be queried from the device
 *   attribute cudaDeviceAttrReadOnlyHostRegisterSupported.  Using this flag with
 *   a current context associated with a device that does not have this attribute
 *   set will cause ::cudaHostRegister to error with cudaErrorNotSupported.
 *
 * All of these flags are orthogonal to one another: a developer may page-lock
 * memory that is portable or mapped with no restrictions.
 *
 * The CUDA context must have been created with the ::cudaMapHost flag in
 * order for the ::cudaHostRegisterMapped flag to have any effect.
 *
 * The ::cudaHostRegisterMapped flag may be specified on CUDA contexts for
 * devices that do not support mapped pinned memory. The failure is deferred
 * to ::cudaHostGetDevicePointer() because the memory may be mapped into
 * other CUDA contexts via the ::cudaHostRegisterPortable flag.
 *
 * For devices that have a non-zero value for the device attribute
 * ::cudaDevAttrCanUseHostPointerForRegisteredMem, the memory
 * can also be accessed from the device using the host pointer \p ptr.
 * The device pointer returned by ::cudaHostGetDevicePointer() may or may not
 * match the original host pointer \p ptr and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::cudaHostGetDevicePointer()
 * will match the original pointer \p ptr. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::cudaHostGetDevicePointer() will not match the original host pointer \p ptr,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * The memory page-locked by this function must be unregistered with ::cudaHostUnregister().
 *
 * \param ptr   - Host pointer to memory to page-lock
 * \param size  - Size in bytes of the address range to page-lock in bytes
 * \param flags - Flags for allocation request
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorHostMemoryAlreadyRegistered,
 * ::cudaErrorNotSupported
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaHostUnregister, ::cudaHostGetFlags, ::cudaHostGetDevicePointer,
 * ::cuMemHostRegister
 */
extern __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags);

/**
 * \brief Unregisters a memory range that was registered with cudaHostRegister
 *
 * Unmaps the memory range whose base address is specified by \p ptr, and makes
 * it pageable again.
 *
 * The base address must be the same one specified to ::cudaHostRegister().
 *
 * \param ptr - Host pointer to memory to unregister
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorHostMemoryNotRegistered
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaHostUnregister,
 * ::cuMemHostUnregister
 */
extern __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr);

/**
 * \brief Passes back device pointer of mapped host memory allocated by
 * cudaHostAlloc or registered by cudaHostRegister
 *
 * Passes back the device pointer corresponding to the mapped, pinned host
 * buffer allocated by ::cudaHostAlloc() or registered by ::cudaHostRegister().
 *
 * ::cudaHostGetDevicePointer() will fail if the ::cudaDeviceMapHost flag was
 * not specified before deferred context creation occurred, or if called on a
 * device that does not support mapped, pinned memory.
 *
 * For devices that have a non-zero value for the device attribute
 * ::cudaDevAttrCanUseHostPointerForRegisteredMem, the memory
 * can also be accessed from the device using the host pointer \p pHost.
 * The device pointer returned by ::cudaHostGetDevicePointer() may or may not
 * match the original host pointer \p pHost and depends on the devices visible to the
 * application. If all devices visible to the application have a non-zero value for the
 * device attribute, the device pointer returned by ::cudaHostGetDevicePointer()
 * will match the original pointer \p pHost. If any device visible to the application
 * has a zero value for the device attribute, the device pointer returned by
 * ::cudaHostGetDevicePointer() will not match the original host pointer \p pHost,
 * but it will be suitable for use on all devices provided Unified Virtual Addressing
 * is enabled. In such systems, it is valid to access the memory using either pointer
 * on devices that have a non-zero value for the device attribute. Note however that
 * such devices should access the memory using only of the two pointers and not both.
 *
 * \p flags provides for future releases.  For now, it must be set to 0.
 *
 * \param pDevice - Returned device pointer for mapped memory
 * \param pHost   - Requested host pointer mapping
 * \param flags   - Flags for extensions (must be 0 for now)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaSetDeviceFlags, ::cudaHostAlloc,
 * ::cuMemHostGetDevicePointer
 */
extern __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);

/**
 * \brief Passes back flags used to allocate pinned host memory allocated by
 * cudaHostAlloc
 *
 * ::cudaHostGetFlags() will fail if the input pointer does not
 * reside in an address range allocated by ::cudaHostAlloc().
 *
 * \param pFlags - Returned flags word
 * \param pHost - Host pointer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaHostAlloc,
 * ::cuMemHostGetFlags
 */
extern __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost);

/**
 * \brief Allocates logical 1D, 2D, or 3D memory objects on the device
 *
 * Allocates at least \p width * \p height * \p depth bytes of linear memory
 * on the device and returns a ::cudaPitchedPtr in which \p ptr is a pointer
 * to the allocated memory. The function may pad the allocation to ensure
 * hardware alignment requirements are met. The pitch returned in the \p pitch
 * field of \p pitchedDevPtr is the width in bytes of the allocation.
 *
 * The returned ::cudaPitchedPtr contains additional fields \p xsize and
 * \p ysize, the logical width and height of the allocation, which are
 * equivalent to the \p width and \p height \p extent parameters provided by
 * the programmer during allocation.
 *
 * For allocations of 2D and 3D objects, it is highly recommended that
 * programmers perform allocations using ::cudaMalloc3D() or
 * ::cudaMallocPitch(). Due to alignment restrictions in the hardware, this is
 * especially true if the application will be performing memory copies
 * involving 2D or 3D objects (whether linear memory or CUDA arrays).
 *
 * \param pitchedDevPtr  - Pointer to allocated pitched device memory
 * \param extent         - Requested allocation size (\p width field in bytes)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMemcpy3D, ::cudaMemset3D,
 * ::cudaMalloc3DArray, ::cudaMallocArray, ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc, ::make_cudaPitchedPtr, ::make_cudaExtent,
 * ::cuMemAllocPitch
 */
extern __host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);

/**
 * \brief Allocate an array on the device
 *
 * Allocates a CUDA array according to the ::cudaChannelFormatDesc structure
 * \p desc and returns a handle to the new CUDA array in \p *array.
 *
 * The ::cudaChannelFormatDesc is defined as:
 * \code
    struct cudaChannelFormatDesc {
        int x, y, z, w;
        enum cudaChannelFormatKind f;
    };
    \endcode
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * ::cudaMalloc3DArray() can allocate the following:
 *
 * - A 1D array is allocated if the height and depth extents are both zero.
 * - A 2D array is allocated if only the depth extent is zero.
 * - A 3D array is allocated if all three extents are non-zero.
 * - A 1D layered CUDA array is allocated if only the height extent is zero and
 * the cudaArrayLayered flag is set. Each layer is a 1D array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered CUDA array is allocated if all three extents are non-zero and 
 * the cudaArrayLayered flag is set. Each layer is a 2D array. The number of layers is 
 * determined by the depth extent.
 * - A cubemap CUDA array is allocated if all three extents are non-zero and the
 * cudaArrayCubemap flag is set. Width must be equal to height, and depth must be six. A cubemap is
 * a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube. 
 * The order of the six layers in memory is the same as that listed in ::cudaGraphicsCubeFace.
 * - A cubemap layered CUDA array is allocated if all three extents are non-zero, and both,
 * cudaArrayCubemap and cudaArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists 
 * of a collection of cubemaps. The first six layers represent the first cubemap, the next six layers form 
 * the second cubemap, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
 * - ::cudaArrayLayered: Allocates a layered CUDA array, with the depth extent indicating the number of layers
 * - ::cudaArrayCubemap: Allocates a cubemap CUDA array. Width must be equal to height, and depth must be six.
 *   If the cudaArrayLayered flag is also set, depth must be a multiple of six.
 * - ::cudaArraySurfaceLoadStore: Allocates a CUDA array that could be read from or written to using a surface
 *   reference.
 * - ::cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the CUDA 
 *   array. Texture gather can only be performed on 2D CUDA arrays.
 * - ::cudaArraySparse: Allocates a CUDA array without physical backing memory. The subregions within this sparse array 
 *   can later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. This flag can only be used for 
 *   creating 2D, 3D or 2D layered sparse CUDA arrays. The physical backing memory must be allocated via ::cuMemCreate.

 * - ::cudaArrayDeferredMapping: Allocates a CUDA array without physical backing memory. The entire array can
 *   later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. The physical backing memory must be allocated
 *   via ::cuMemCreate.

 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * Note that 2D CUDA arrays have different size requirements if the ::cudaArrayTextureGather flag is set. In that
 * case, the valid range for (width, height, depth) is ((1,maxTexture2DGather[0]), (1,maxTexture2DGather[1]), 0).
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>CUDA array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with cudaArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1D), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2D[0]), (1,maxTexture2D[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Cubemap</entry>
 * <entry>{ (1,maxTextureCubemap), (1,maxTextureCubemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceCubemap), (1,maxSurfaceCubemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Cubemap Layered</entry>
 * <entry>{ (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
 * (1,maxTextureCubemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
 * (1,maxSurfaceCubemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param array  - Pointer to allocated array in device memory
 * \param desc   - Requested channel format
 * \param extent - Requested allocation size (\p width field in elements)
 * \param flags  - Flags for extensions
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc, ::cudaMallocPitch, ::cudaFree,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::make_cudaExtent,
 * ::cuArray3DCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0));

/**
 * \brief Allocate a mipmapped array on the device
 *
 * Allocates a CUDA mipmapped array according to the ::cudaChannelFormatDesc structure
 * \p desc and returns a handle to the new CUDA mipmapped array in \p *mipmappedArray.
 * \p numLevels specifies the number of mipmap levels to be allocated. This value is
 * clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 *
 * The ::cudaChannelFormatDesc is defined as:
 * \code
    struct cudaChannelFormatDesc {
        int x, y, z, w;
        enum cudaChannelFormatKind f;
    };
    \endcode
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * ::cudaMallocMipmappedArray() can allocate the following:
 *
 * - A 1D mipmapped array is allocated if the height and depth extents are both zero.
 * - A 2D mipmapped array is allocated if only the depth extent is zero.
 * - A 3D mipmapped array is allocated if all three extents are non-zero.
 * - A 1D layered CUDA mipmapped array is allocated if only the height extent is zero and
 * the cudaArrayLayered flag is set. Each layer is a 1D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and 
 * the cudaArrayLayered flag is set. Each layer is a 2D mipmapped array. The number of layers is 
 * determined by the depth extent.
 * - A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the
 * cudaArrayCubemap flag is set. Width must be equal to height, and depth must be six.
 * The order of the six layers in memory is the same as that listed in ::cudaGraphicsCubeFace.
 * - A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both,
 * cudaArrayCubemap and cudaArrayLayered flags are set. Width must be equal to height, and depth must be 
 * a multiple of six. A cubemap layered CUDA mipmapped array is a special type of 2D layered CUDA mipmapped
 * array that consists of a collection of cubemap mipmapped arrays. The first six layers represent the 
 * first cubemap mipmapped array, the next six layers form the second cubemap mipmapped array, and so on.
 *
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaArrayDefault: This flag's value is defined to be 0 and provides default mipmapped array allocation
 * - ::cudaArrayLayered: Allocates a layered CUDA mipmapped array, with the depth extent indicating the number of layers
 * - ::cudaArrayCubemap: Allocates a cubemap CUDA mipmapped array. Width must be equal to height, and depth must be six.
 *   If the cudaArrayLayered flag is also set, depth must be a multiple of six.
 * - ::cudaArraySurfaceLoadStore: This flag indicates that individual mipmap levels of the CUDA mipmapped array 
 *   will be read from or written to using a surface reference.
 * - ::cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the CUDA 
 *   array. Texture gather can only be performed on 2D CUDA mipmapped arrays, and the gather operations are
 *   performed only on the most detailed mipmap level.
 * - ::cudaArraySparse: Allocates a CUDA mipmapped array without physical backing memory. The subregions within this sparse array
 *   can later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. This flag can only be used for creating 
 *   2D, 3D or 2D layered sparse CUDA mipmapped arrays. The physical backing memory must be allocated via ::cuMemCreate.

 * - ::cudaArrayDeferredMapping: Allocates a CUDA mipmapped array without physical backing memory. The entire array can
 *   later be mapped onto a physical memory allocation by calling ::cuMemMapArrayAsync. The physical backing memory must be allocated
 *   via ::cuMemCreate.

 *
 * The width, height and depth extents must meet certain size requirements as listed in the following table.
 * All values are specified in elements.
 *
 * \xmlonly
 * <table outputclass="xmlonly">
 * <tgroup cols="3" colsep="1" rowsep="1">
 * <colspec colname="c1" colwidth="1.0*"/>
 * <colspec colname="c2" colwidth="3.0*"/>
 * <colspec colname="c3" colwidth="3.0*"/>
 * <thead>
 * <row>
 * <entry>CUDA array type</entry>
 * <entry>Valid extents that must always be met {(width range in elements),
 * (height range), (depth range)}</entry>
 * <entry>Valid extents with cudaArraySurfaceLoadStore set {(width range in
 * elements), (height range), (depth range)}</entry>
 * </row>
 * </thead>
 * <tbody>
 * <row>
 * <entry>1D</entry>
 * <entry>{ (1,maxTexture1DMipmap), 0, 0 }</entry>
 * <entry>{ (1,maxSurface1D), 0, 0 }</entry>
 * </row>
 * <row>
 * <entry>2D</entry>
 * <entry>{ (1,maxTexture2DMipmap[0]), (1,maxTexture2DMipmap[1]), 0 }</entry>
 * <entry>{ (1,maxSurface2D[0]), (1,maxSurface2D[1]), 0 }</entry>
 * </row>
 * <row>
 * <entry>3D</entry>
 * <entry>{ (1,maxTexture3D[0]), (1,maxTexture3D[1]), (1,maxTexture3D[2]) }
 * OR { (1,maxTexture3DAlt[0]), (1,maxTexture3DAlt[1]),
 * (1,maxTexture3DAlt[2]) }</entry>
 * <entry>{ (1,maxSurface3D[0]), (1,maxSurface3D[1]), (1,maxSurface3D[2]) }</entry>
 * </row>
 * <row>
 * <entry>1D Layered</entry>
 * <entry>{ (1,maxTexture1DLayered[0]), 0, (1,maxTexture1DLayered[1]) }</entry>
 * <entry>{ (1,maxSurface1DLayered[0]), 0, (1,maxSurface1DLayered[1]) }</entry>
 * </row>
 * <row>
 * <entry>2D Layered</entry>
 * <entry>{ (1,maxTexture2DLayered[0]), (1,maxTexture2DLayered[1]),
 * (1,maxTexture2DLayered[2]) }</entry>
 * <entry>{ (1,maxSurface2DLayered[0]), (1,maxSurface2DLayered[1]),
 * (1,maxSurface2DLayered[2]) }</entry>
 * </row>
 * <row>
 * <entry>Cubemap</entry>
 * <entry>{ (1,maxTextureCubemap), (1,maxTextureCubemap), 6 }</entry>
 * <entry>{ (1,maxSurfaceCubemap), (1,maxSurfaceCubemap), 6 }</entry>
 * </row>
 * <row>
 * <entry>Cubemap Layered</entry>
 * <entry>{ (1,maxTextureCubemapLayered[0]), (1,maxTextureCubemapLayered[0]),
 * (1,maxTextureCubemapLayered[1]) }</entry>
 * <entry>{ (1,maxSurfaceCubemapLayered[0]), (1,maxSurfaceCubemapLayered[0]),
 * (1,maxSurfaceCubemapLayered[1]) }</entry>
 * </row>
 * </tbody>
 * </tgroup>
 * </table>
 * \endxmlonly
 *
 * \param mipmappedArray  - Pointer to allocated mipmapped array in device memory
 * \param desc            - Requested channel format
 * \param extent          - Requested allocation size (\p width field in elements)
 * \param numLevels       - Number of mipmap levels to allocate
 * \param flags           - Flags for extensions
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc, ::cudaMallocPitch, ::cudaFree,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::make_cudaExtent,
 * ::cuMipmappedArrayCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags __dv(0));

/**
 * \brief Gets a mipmap level of a CUDA mipmapped array
 *
 * Returns in \p *levelArray a CUDA array that represents a single mipmap level
 * of the CUDA mipmapped array \p mipmappedArray.
 *
 * If \p level is greater than the maximum number of levels in this mipmapped array,
 * ::cudaErrorInvalidValue is returned.
 *
 * If \p mipmappedArray is NULL,
 * ::cudaErrorInvalidResourceHandle is returned.
 *
 * \param levelArray     - Returned mipmap level CUDA array
 * \param mipmappedArray - CUDA mipmapped array
 * \param level          - Mipmap level
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc, ::cudaMallocPitch, ::cudaFree,
 * ::cudaFreeArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc,
 * ::make_cudaExtent,
 * ::cuMipmappedArrayGetLevel
 */
extern __host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);

/**
 * \brief Copies data between 3D objects
 *
\code
struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);

struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
};
struct cudaPos make_cudaPos(size_t x, size_t y, size_t z);

struct cudaMemcpy3DParms {
  cudaArray_t           srcArray;
  struct cudaPos        srcPos;
  struct cudaPitchedPtr srcPtr;
  cudaArray_t           dstArray;
  struct cudaPos        dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent     extent;
  enum cudaMemcpyKind   kind;
};
\endcode
 *
 * ::cudaMemcpy3D() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a CUDA
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::cudaMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
cudaMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::cudaMemcpy3D() must specify one of \p srcArray or
 * \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::cudaMemcpy3D() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a CUDA array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no CUDA array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * For ::cudaMemcpyHostToHost or ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost
 * passed as kind and cudaArray type passed as source or destination, if the kind
 * implies cudaArray type to be present on the host, ::cudaMemcpy3D() will
 * disregard that implication and silently correct the kind based on the fact that
 * cudaArray type can only be present on the device.
 *
 * If the source and destination are both arrays, ::cudaMemcpy3D() will return
 * an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must entirely contain the region defined by \p srcPos
 * and \p extent. The destination object must entirely contain the region
 * defined by \p dstPos and \p extent.
 *
 * ::cudaMemcpy3D() returns an error if the pitch of \p srcPtr or \p dstPtr
 * exceeds the maximum allowed. The pitch of a ::cudaPitchedPtr allocated
 * with ::cudaMalloc3D() will always be valid.
 *
 * \param p - 3D memory copy parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc3DArray, ::cudaMemset3D, ::cudaMemcpy3DAsync,
 * ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::make_cudaExtent, ::make_cudaPos,
 * ::cuMemcpy3D
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p);

/**
 * \brief Copies memory between devices
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::cudaMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * Note that this function is synchronous with respect to the host only if
 * the source or destination of the transfer is host memory.  Note also 
 * that this copy is serialized with respect to all pending and future 
 * asynchronous work in to the current device, the copy's source device,
 * and the copy's destination device (use ::cudaMemcpy3DPeerAsync to avoid 
 * this synchronization).
 *
 * \param p - Parameters for the memory copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync, ::cudaMemcpyPeerAsync,
 * ::cudaMemcpy3DPeerAsync,
 * ::cuMemcpy3DPeer
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);

/**
 * \brief Copies data between 3D objects
 *
\code
struct cudaExtent {
  size_t width;
  size_t height;
  size_t depth;
};
struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);

struct cudaPos {
  size_t x;
  size_t y;
  size_t z;
};
struct cudaPos make_cudaPos(size_t x, size_t y, size_t z);

struct cudaMemcpy3DParms {
  cudaArray_t           srcArray;
  struct cudaPos        srcPos;
  struct cudaPitchedPtr srcPtr;
  cudaArray_t           dstArray;
  struct cudaPos        dstPos;
  struct cudaPitchedPtr dstPtr;
  struct cudaExtent     extent;
  enum cudaMemcpyKind   kind;
};
\endcode
 *
 * ::cudaMemcpy3DAsync() copies data betwen two 3D objects. The source and
 * destination objects may be in either host memory, device memory, or a CUDA
 * array. The source, destination, extent, and kind of copy performed is
 * specified by the ::cudaMemcpy3DParms struct which should be initialized to
 * zero before use:
\code
cudaMemcpy3DParms myParms = {0};
\endcode
 *
 * The struct passed to ::cudaMemcpy3DAsync() must specify one of \p srcArray
 * or \p srcPtr and one of \p dstArray or \p dstPtr. Passing more than one
 * non-zero source or destination will cause ::cudaMemcpy3DAsync() to return an
 * error.
 *
 * The \p srcPos and \p dstPos fields are optional offsets into the source and
 * destination objects and are defined in units of each object's elements. The
 * element for a host or device pointer is assumed to be <b>unsigned char</b>.
 * For CUDA arrays, positions must be in the range [0, 2048) for any
 * dimension.
 *
 * The \p extent field defines the dimensions of the transferred area in
 * elements. If a CUDA array is participating in the copy, the extent is
 * defined in terms of that array's elements. If no CUDA array is
 * participating in the copy then the extents are defined in elements of
 * <b>unsigned char</b>.
 *
 * The \p kind field defines the direction of the copy. It must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * For ::cudaMemcpyHostToHost or ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost
 * passed as kind and cudaArray type passed as source or destination, if the kind
 * implies cudaArray type to be present on the host, ::cudaMemcpy3DAsync() will
 * disregard that implication and silently correct the kind based on the fact that
 * cudaArray type can only be present on the device.
 *
 * If the source and destination are both arrays, ::cudaMemcpy3DAsync() will
 * return an error if they do not have the same element size.
 *
 * The source and destination object may not overlap. If overlapping source
 * and destination objects are specified, undefined behavior will result.
 *
 * The source object must lie entirely within the region defined by \p srcPos
 * and \p extent. The destination object must lie entirely within the region
 * defined by \p dstPos and \p extent.
 *
 * ::cudaMemcpy3DAsync() returns an error if the pitch of \p srcPtr or
 * \p dstPtr exceeds the maximum allowed. The pitch of a
 * ::cudaPitchedPtr allocated with ::cudaMalloc3D() will always be valid.
 *
 * ::cudaMemcpy3DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param p      - 3D memory copy parameters
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMalloc3D, ::cudaMalloc3DArray, ::cudaMemset3D, ::cudaMemcpy3D,
 * ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, :::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::make_cudaExtent, ::make_cudaPos,
 * ::cuMemcpy3DAsync
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));

/**
 * \brief Copies memory between devices asynchronously.
 *
 * Perform a 3D memory copy according to the parameters specified in
 * \p p.  See the definition of the ::cudaMemcpy3DPeerParms structure
 * for documentation of its parameters.
 *
 * \param p      - Parameters for the memory copy
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync, ::cudaMemcpyPeerAsync,
 * ::cudaMemcpy3DPeerAsync,
 * ::cuMemcpy3DPeerAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream __dv(0));

/**
 * \brief Gets free and total device memory
 *
 * Returns in \p *total the total amount of memory available to the the current context.
 * Returns in \p *free the amount of memory on the device that is free according to the OS.
 * CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free.
 * In a multi-tenet situation, free estimate returned is prone to race condition where
 * a new allocation/free done by a different process or a different thread in the same
 * process between the time when free memory was estimated and reported, will result in
 * deviation in free value reported and actual free memory.
 *
 * The integrated GPU on Tegra shares memory with CPU and other component
 * of the SoC. The free and total values returned by the API excludes
 * the SWAP memory space maintained by the OS on some platforms.
 * The OS may move some of the memory pages into swap area as the GPU or
 * CPU allocate or access memory. See Tegra app note on how to calculate
 * total and free memory on Tegra.
 *
 * \param free  - Returned free memory in bytes
 * \param total - Returned total memory in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuMemGetInfo
 */
extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total);

/**
 * \brief Gets info about the specified cudaArray
 * 
 * Returns in \p *desc, \p *extent and \p *flags respectively, the type, shape 
 * and flags of \p array.
 *
 * Any of \p *desc, \p *extent and \p *flags may be specified as NULL.
 *
 * \param desc   - Returned array type
 * \param extent - Returned array shape. 2D arrays will have depth of zero
 * \param flags  - Returned array flags
 * \param array  - The ::cudaArray to get info for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuArrayGetDescriptor,
 * ::cuArray3DGetDescriptor
 */
extern __host__ cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array);

/**
 * \brief Gets a CUDA array plane from a CUDA array
 *
 * Returns in \p pPlaneArray a CUDA array that represents a single format plane
 * of the CUDA array \p hArray.
 *
 * If \p planeIdx is greater than the maximum number of planes in this array or if the array does
 * not have a multi-planar format e.g: ::cudaChannelFormatKindNV12, then ::cudaErrorInvalidValue is returned.
 *
 * Note that if the \p hArray has format ::cudaChannelFormatKindNV12, then passing in 0 for \p planeIdx returns
 * a CUDA array of the same size as \p hArray but with one 8-bit channel and ::cudaChannelFormatKindUnsigned as its format kind.
 * If 1 is passed for \p planeIdx, then the returned CUDA array has half the height and width
 * of \p hArray with two 8-bit channels and ::cudaChannelFormatKindUnsigned as its format kind.
 *
 * \param pPlaneArray   - Returned CUDA array referenced by the \p planeIdx
 * \param hArray        - CUDA array
 * \param planeIdx      - Plane index
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::cuArrayGetPlane
 */
extern __host__ cudaError_t CUDARTAPI cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx);


/**
 * \brief Returns the memory requirements of a CUDA array
 *
 * Returns the memory requirements of a CUDA array in \p memoryRequirements
 * If the CUDA array is not allocated with flag ::cudaArrayDeferredMapping
 * ::cudaErrorInvalidValue will be returned.
 *
 * The returned value in ::cudaArrayMemoryRequirements::size
 * represents the total size of the CUDA array.
 * The returned value in ::cudaArrayMemoryRequirements::alignment
 * represents the alignment necessary for mapping the CUDA array.
 *
 * \return
 * ::cudaSuccess
 * ::cudaErrorInvalidValue
 *
 * \param[out] memoryRequirements - Pointer to ::cudaArrayMemoryRequirements
 * \param[in] array - CUDA array to get the memory requirements of
 * \param[in] device - Device to get the memory requirements for
 * \sa ::cudaMipmappedArrayGetMemoryRequirements
 */
extern __host__ cudaError_t CUDARTAPI cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements  *memoryRequirements, cudaArray_t array, int device);

/**
 * \brief Returns the memory requirements of a CUDA mipmapped array
 *
 * Returns the memory requirements of a CUDA mipmapped array in \p memoryRequirements
 * If the CUDA mipmapped array is not allocated with flag ::cudaArrayDeferredMapping
 * ::cudaErrorInvalidValue will be returned.
 *
 * The returned value in ::cudaArrayMemoryRequirements::size
 * represents the total size of the CUDA mipmapped array.
 * The returned value in ::cudaArrayMemoryRequirements::alignment
 * represents the alignment necessary for mapping the CUDA mipmapped
 * array.
 *
 * \return
 * ::cudaSuccess
 * ::cudaErrorInvalidValue
 *
 * \param[out] memoryRequirements - Pointer to ::cudaArrayMemoryRequirements
 * \param[in] mipmap - CUDA mipmapped array to get the memory requirements of
 * \param[in] device - Device to get the memory requirements for
 * \sa ::cudaArrayGetMemoryRequirements
 */
extern __host__ cudaError_t CUDARTAPI cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements *memoryRequirements, cudaMipmappedArray_t mipmap, int device);


/**
 * \brief Returns the layout properties of a sparse CUDA array
 *
 * Returns the layout properties of a sparse CUDA array in \p sparseProperties.
 * If the CUDA array is not allocated with flag ::cudaArraySparse
 * ::cudaErrorInvalidValue will be returned.
 *
 * If the returned value in ::cudaArraySparseProperties::flags contains ::cudaArraySparsePropertiesSingleMipTail,
 * then ::cudaArraySparseProperties::miptailSize represents the total size of the array. Otherwise, it will be zero.
 * Also, the returned value in ::cudaArraySparseProperties::miptailFirstLevel is always zero.
 * Note that the \p array must have been allocated using ::cudaMallocArray or ::cudaMalloc3DArray. For CUDA arrays obtained
 * using ::cudaMipmappedArrayGetLevel, ::cudaErrorInvalidValue will be returned. Instead, ::cudaMipmappedArrayGetSparseProperties
 * must be used to obtain the sparse properties of the entire CUDA mipmapped array to which \p array belongs to.
 *
 * \return
 * ::cudaSuccess
 * ::cudaErrorInvalidValue
 *
 * \param[out] sparseProperties - Pointer to return the ::cudaArraySparseProperties
 * \param[in] array             - The CUDA array to get the sparse properties of 
 *
 * \sa
 * ::cudaMipmappedArrayGetSparseProperties,
 * ::cuMemMapArrayAsync
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array);
#endif

/**
 * \brief Returns the layout properties of a sparse CUDA mipmapped array
 *
 * Returns the sparse array layout properties in \p sparseProperties.
 * If the CUDA mipmapped array is not allocated with flag ::cudaArraySparse
 * ::cudaErrorInvalidValue will be returned.
 *
 * For non-layered CUDA mipmapped arrays, ::cudaArraySparseProperties::miptailSize returns the
 * size of the mip tail region. The mip tail region includes all mip levels whose width, height or depth
 * is less than that of the tile.
 * For layered CUDA mipmapped arrays, if ::cudaArraySparseProperties::flags contains ::cudaArraySparsePropertiesSingleMipTail,
 * then ::cudaArraySparseProperties::miptailSize specifies the size of the mip tail of all layers combined.
 * Otherwise, ::cudaArraySparseProperties::miptailSize specifies mip tail size per layer.
 * The returned value of ::cudaArraySparseProperties::miptailFirstLevel is valid only if ::cudaArraySparseProperties::miptailSize is non-zero.
 *
 * \return
 * ::cudaSuccess
 * ::cudaErrorInvalidValue
 *
 * \param[out] sparseProperties - Pointer to return ::cudaArraySparseProperties
 * \param[in] mipmap            - The CUDA mipmapped array to get the sparse properties of
 *
 * \sa
 * ::cudaArrayGetSparseProperties,
 * ::cuMemMapArrayAsync
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap);
#endif

/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind specifies the direction
 * of the copy, and must be one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Calling
 * ::cudaMemcpy() with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * \param dst   - Destination memory address
 * \param src   - Source memory address
 * \param count - Size in bytes to copy
 * \param kind  - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \note_sync
 * \note_memcpy
 *
 * \sa ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyDtoH,
 * ::cuMemcpyHtoD,
 * ::cuMemcpyDtoD,
 * ::cuMemcpy
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

/**
 * \brief Copies memory between two devices
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host, but 
 * serialized with respect all pending and future asynchronous work in to the 
 * current device, \p srcDevice, and \p dstDevice (use ::cudaMemcpyPeerAsync 
 * to avoid this synchronization).
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyAsync, ::cudaMemcpyPeerAsync,
 * ::cudaMemcpy3DPeerAsync,
 * ::cuMemcpyPeer
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. \p dpitch and
 * \p spitch are the widths in memory in bytes of the 2D arrays pointed to by
 * \p dst and \p src, including any padding added to the end of each row. The
 * memory areas may not overlap. \p width must not exceed either \p dpitch or
 * \p spitch. Calling ::cudaMemcpy2D() with \p dst and \p src pointers that do
 * not match the direction of the copy results in an undefined behavior.
 * ::cudaMemcpy2D() returns an error if \p dpitch or \p spitch exceeds
 * the maximum allowed.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the CUDA array \p dst starting at
 * \p hOffset rows and \p wOffset bytes from the upper left corner,
 * where \p kind specifies the direction of the copy, and must be one
 * of ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the CUDA array \p dst. \p width must
 * not exceed \p spitch. ::cudaMemcpy2DToArray() returns an error if \p spitch
 * exceeds the maximum allowed.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the CUDA
 * array \p src starting at \p hOffset rows and \p wOffset bytes from the
 * upper left corner to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. \p dpitch is the
 * width in memory in bytes of the 2D array pointed to by \p dst, including any
 * padding added to the end of each row. \p wOffset + \p width must not exceed
 * the width of the CUDA array \p src. \p width must not exceed \p dpitch.
 * ::cudaMemcpy2DFromArray() returns an error if \p dpitch exceeds the maximum
 * allowed.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the CUDA
 * array \p src starting at \p hOffsetSrc rows and \p wOffsetSrc bytes from the
 * upper left corner to the CUDA array \p dst starting at \p hOffsetDst rows
 * and \p wOffsetDst bytes from the upper left corner, where \p kind
 * specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p wOffsetDst + \p width must not exceed the width of the CUDA array \p dst.
 * \p wOffsetSrc + \p width must not exceed the width of the CUDA array \p src.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset (columns in bytes)
 * \param hOffsetDst - Destination starting Y offset (rows)
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset (columns in bytes)
 * \param hOffsetSrc - Source starting Y offset (rows)
 * \param width      - Width of matrix transfer (columns in bytes)
 * \param height     - Height of matrix transfer (rows)
 * \param kind       - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2D,
 * ::cuMemcpy2DUnaligned
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::cudaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray,  ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy,
 * ::cuMemcpyHtoD,
 * ::cuMemcpyDtoD
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::cudaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy,
 * ::cuMemcpyDtoH,
 * ::cuMemcpyDtoD
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));


/**
 * \brief Copies data between host and device
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * memory area pointed to by \p dst, where \p kind specifies the
 * direction of the copy, and must be one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * 
 * The memory areas may not overlap. Calling ::cudaMemcpyAsync() with \p dst and
 * \p src pointers that do not match the direction of the copy results in an
 * undefined behavior.
 *
 * ::cudaMemcpyAsync() is asynchronous with respect to the host, so the call
 * may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and the \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param dst    - Destination memory address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyAsync,
 * ::cuMemcpyDtoHAsync,
 * ::cuMemcpyHtoDAsync,
 * ::cuMemcpyDtoDAsync
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/**
 * \brief Copies memory between two devices asynchronously.
 *
 * Copies memory from one device to memory on another device.  \p dst is the 
 * base device pointer of the destination memory and \p dstDevice is the 
 * destination device.  \p src is the base device pointer of the source memory 
 * and \p srcDevice is the source device.  \p count specifies the number of bytes 
 * to copy.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param dst       - Destination device pointer
 * \param dstDevice - Destination device
 * \param src       - Source device pointer
 * \param srcDevice - Source device
 * \param count     - Size of memory copy in bytes
 * \param stream    - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync,
 * ::cudaMemcpy3DPeerAsync,
 * ::cuMemcpyPeerAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p dpitch and \p spitch are the widths in memory in bytes of the 2D arrays
 * pointed to by \p dst and \p src, including any padding added to the end of
 * each row. The memory areas may not overlap. \p width must not exceed either
 * \p dpitch or \p spitch.
 *
 * Calling ::cudaMemcpy2DAsync() with \p dst and \p src pointers that do not
 * match the direction of the copy results in an undefined behavior.
 * ::cudaMemcpy2DAsync() returns an error if \p dpitch or \p spitch is greater
 * than the maximum allowed.
 *
 * ::cudaMemcpy2DAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param dst    - Destination memory address
 * \param dpitch - Pitch of destination memory
 * \param src    - Source memory address
 * \param spitch - Pitch of source memory
 * \param width  - Width of matrix transfer (columns in bytes)
 * \param height - Height of matrix transfer (rows)
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the memory
 * area pointed to by \p src to the CUDA array \p dst starting at \p hOffset
 * rows and \p wOffset bytes from the upper left corner, where \p kind specifies
 * the direction of the copy, and must be one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p spitch is the width in memory in bytes of the 2D array pointed to by
 * \p src, including any padding added to the end of each row. \p wOffset +
 * \p width must not exceed the width of the CUDA array \p dst. \p width must
 * not exceed \p spitch. ::cudaMemcpy2DToArrayAsync() returns an error if
 * \p spitch exceeds the maximum allowed.
 *
 * ::cudaMemcpy2DToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and
 * \p stream is non-zero, the copy may overlap with operations in other
 * streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 *
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * Copies a matrix (\p height rows of \p width bytes each) from the CUDA
 * array \p src starting at \p hOffset rows and \p wOffset bytes from the
 * upper left corner to the memory area pointed to by \p dst,
 * where \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 * \p dpitch is the width in memory in bytes of the 2D
 * array pointed to by \p dst, including any padding added to the end of each
 * row. \p wOffset + \p width must not exceed the width of the CUDA array
 * \p src. \p width must not exceed \p dpitch. ::cudaMemcpy2DFromArrayAsync()
 * returns an error if \p dpitch exceeds the maximum allowed.
 *
 * ::cudaMemcpy2DFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream is
 * non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidPitchValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 * \note_memcpy
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 *
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpy2DAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/**
 * \brief Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area pointed to by \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::cudaMemcpyToSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice and \p stream is non-zero, the copy
 * may overlap with operations in other streams.
 *
 * \param symbol - Device symbol address
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyAsync,
 * ::cuMemcpyHtoDAsync,
 * ::cuMemcpyDtoDAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/**
 * \brief Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that resides in
 * global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::cudaMemcpyFromSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyDeviceToHost and \p stream is non-zero, the copy may overlap
 * with operations in other streams.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync,
 * ::cuMemcpyAsync,
 * ::cuMemcpyDtoHAsync,
 * ::cuMemcpyDtoDAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));


/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuMemsetD8,
 * ::cuMemsetD16,
 * ::cuMemsetD32
 */
extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::cudaMallocPitch().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p devPtr refers to pinned host memory.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory(Unused if \p height is 1)
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemset, ::cudaMemset3D, ::cudaMemsetAsync,
 * ::cudaMemset2DAsync, ::cudaMemset3DAsync,
 * ::cuMemsetD2D8,
 * ::cuMemsetD2D16,
 * ::cuMemsetD2D32
 */
extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 * The \p pitch field of \p pitchedDevPtr is ignored when \p height and \p depth 
 * are both equal to 1. 
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::cudaMalloc3D().
 *
 * Note that this function is asynchronous with respect to the host unless
 * \p pitchedDevPtr refers to pinned host memory.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemset, ::cudaMemset2D,
 * ::cudaMemsetAsync, ::cudaMemset2DAsync, ::cudaMemset3DAsync,
 * ::cudaMalloc3D, ::make_cudaPitchedPtr,
 * ::make_cudaExtent
 */
extern __host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);

/**
 * \brief Initializes or sets device memory to a value
 *
 * Fills the first \p count bytes of the memory area pointed to by \p devPtr
 * with the constant byte value \p value.
 *
 * ::cudaMemsetAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param devPtr - Pointer to device memory
 * \param value  - Value to set for each byte of specified memory
 * \param count  - Size in bytes to set
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemset, ::cudaMemset2D, ::cudaMemset3D,
 * ::cudaMemset2DAsync, ::cudaMemset3DAsync,
 * ::cuMemsetD8Async,
 * ::cuMemsetD16Async,
 * ::cuMemsetD32Async
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0));

/**
 * \brief Initializes or sets device memory to a value
 *
 * Sets to the specified value \p value a matrix (\p height rows of \p width
 * bytes each) pointed to by \p dstPtr. \p pitch is the width in bytes of the
 * 2D array pointed to by \p dstPtr, including any padding added to the end
 * of each row. This function performs fastest when the pitch is one that has
 * been passed back by ::cudaMallocPitch().
 *
 * ::cudaMemset2DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param devPtr - Pointer to 2D device memory
 * \param pitch  - Pitch in bytes of 2D device memory(Unused if \p height is 1)
 * \param value  - Value to set for each byte of specified memory
 * \param width  - Width of matrix set (columns in bytes)
 * \param height - Height of matrix set (rows)
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemset, ::cudaMemset2D, ::cudaMemset3D,
 * ::cudaMemsetAsync, ::cudaMemset3DAsync,
 * ::cuMemsetD2D8Async,
 * ::cuMemsetD2D16Async,
 * ::cuMemsetD2D32Async
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));

/**
 * \brief Initializes or sets device memory to a value
 *
 * Initializes each element of a 3D array to the specified value \p value.
 * The object to initialize is defined by \p pitchedDevPtr. The \p pitch field
 * of \p pitchedDevPtr is the width in memory in bytes of the 3D array pointed
 * to by \p pitchedDevPtr, including any padding added to the end of each row.
 * The \p xsize field specifies the logical width of each row in bytes, while
 * the \p ysize field specifies the height of each 2D slice in rows.
 * The \p pitch field of \p pitchedDevPtr is ignored when \p height and \p depth 
 * are both equal to 1. 
 *
 * The extents of the initialized region are specified as a \p width in bytes,
 * a \p height in rows, and a \p depth in slices.
 *
 * Extents with \p width greater than or equal to the \p xsize of
 * \p pitchedDevPtr may perform significantly faster than extents narrower
 * than the \p xsize. Secondarily, extents with \p height equal to the
 * \p ysize of \p pitchedDevPtr will perform faster than when the \p height is
 * shorter than the \p ysize.
 *
 * This function performs fastest when the \p pitchedDevPtr has been allocated
 * by ::cudaMalloc3D().
 *
 * ::cudaMemset3DAsync() is asynchronous with respect to the host, so
 * the call may return before the memset is complete. The operation can optionally
 * be associated to a stream by passing a non-zero \p stream argument.
 * If \p stream is non-zero, the operation may overlap with operations in other streams.
 *
 * The device version of this function only handles device to device copies and
 * cannot be given local or shared pointers.
 *
 * \param pitchedDevPtr - Pointer to pitched device memory
 * \param value         - Value to set for each byte of specified memory
 * \param extent        - Size parameters for where to set device memory (\p width field in bytes)
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_memset
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemset, ::cudaMemset2D, ::cudaMemset3D,
 * ::cudaMemsetAsync, ::cudaMemset2DAsync,
 * ::cudaMalloc3D, ::make_cudaPitchedPtr,
 * ::make_cudaExtent
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0));

/**
 * \brief Finds the address associated with a CUDA symbol
 *
 * Returns in \p *devPtr the address of symbol \p symbol on the device.
 * \p symbol is a variable that resides in global or constant memory space.
 * If \p symbol cannot be found, or if \p symbol is not declared in the
 * global or constant memory space, \p *devPtr is unchanged and the error
 * ::cudaErrorInvalidSymbol is returned.
 *
 * \param devPtr - Return device pointer associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaGetSymbolAddress(void**, const T&) "cudaGetSymbolAddress (C++ API)",
 * \ref ::cudaGetSymbolSize(size_t*, const void*) "cudaGetSymbolSize (C API)",
 * ::cuModuleGetGlobal
 */
extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol);

/**
 * \brief Finds the size of the object associated with a CUDA symbol
 *
 * Returns in \p *size the size of symbol \p symbol. \p symbol is a variable that
 * resides in global or constant memory space. If \p symbol cannot be found, or
 * if \p symbol is not declared in global or constant memory space, \p *size is
 * unchanged and the error ::cudaErrorInvalidSymbol is returned.
 *
 * \param size   - Size of object associated with symbol
 * \param symbol - Device symbol address
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaGetSymbolAddress(void**, const void*) "cudaGetSymbolAddress (C API)",
 * \ref ::cudaGetSymbolSize(size_t*, const T&) "cudaGetSymbolSize (C++ API)",
 * ::cuModuleGetGlobal
 */
extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol);

/**
 * \brief Prefetches memory to the specified destination device
 *
 * Prefetches memory to the specified destination device.  \p devPtr is the 
 * base device pointer of the memory to be prefetched and \p dstDevice is the 
 * destination device. \p count specifies the number of bytes to copy. \p stream
 * is the stream in which the operation is enqueued. The memory range must refer
 * to managed memory allocated via ::cudaMallocManaged or declared via __managed__ variables.
 *
 * Passing in cudaCpuDeviceId for \p dstDevice will prefetch the data to host memory. If
 * \p dstDevice is a GPU, then the device attribute ::cudaDevAttrConcurrentManagedAccess
 * must be non-zero. Additionally, \p stream must be associated with a device that has a
 * non-zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess.
 *
 * The start address and end address of the memory range will be rounded down and rounded up
 * respectively to be aligned to CPU page size before the prefetch operation is enqueued
 * in the stream.
 *
 * If no physical memory has been allocated for this region, then this memory region
 * will be populated and mapped on the destination device. If there's insufficient
 * memory to prefetch the desired region, the Unified Memory driver may evict pages from other
 * ::cudaMallocManaged allocations to host memory in order to make room. Device memory
 * allocated using ::cudaMalloc or ::cudaMallocArray will not be evicted.
 *
 * By default, any mappings to the previous location of the migrated pages are removed and
 * mappings for the new location are only setup on \p dstDevice. The exact behavior however
 * also depends on the settings applied to this memory range via ::cudaMemAdvise as described
 * below:
 *
 * If ::cudaMemAdviseSetReadMostly was set on any subset of this memory range,
 * then that subset will create a read-only copy of the pages on \p dstDevice.
 *
 * If ::cudaMemAdviseSetPreferredLocation was called on any subset of this memory
 * range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
 * preferred location of any pages in the memory range.
 *
 * If ::cudaMemAdviseSetAccessedBy was called on any subset of this memory range,
 * then mappings to those pages from all the appropriate processors are updated to
 * refer to the new location if establishing such a mapping is possible. Otherwise,
 * those mappings are cleared.
 *
 * Note that this API is not required for functionality and only serves to improve performance
 * by allowing the application to migrate data to a suitable location before it is accessed.
 * Memory accesses to this range are always coherent and are allowed even when the data is
 * actively being migrated.
 *
 * Note that this function is asynchronous with respect to the host and all work
 * on other devices.
 *
 * \param devPtr    - Pointer to be prefetched
 * \param count     - Size in bytes
 * \param dstDevice - Destination device to prefetch to
 * \param stream    - Stream to enqueue prefetch operation
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync,
 * ::cudaMemcpy3DPeerAsync, ::cudaMemAdvise,
 * ::cuMemPrefetchAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream __dv(0));

/**
 * \brief Advise about the usage of a given memory range
 *
 * Advise the Unified Memory subsystem about the usage pattern for the memory range
 * starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
 * range will be rounded down and rounded up respectively to be aligned to CPU page size before the
 * advice is applied. The memory range must refer to managed memory allocated via ::cudaMallocManaged
 * or declared via __managed__ variables. The memory range could also refer to system-allocated pageable
 * memory provided it represents a valid, host-accessible region of memory and all additional constraints
 * imposed by \p advice as outlined below are also satisfied. Specifying an invalid system-allocated pageable
 * memory range results in an error being returned.
 *
 * The \p advice parameter can take the following values:
 * - ::cudaMemAdviseSetReadMostly: This implies that the data is mostly going to be read
 * from and only occasionally written to. Any read accesses from any processor to this region will create a
 * read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::cudaMemPrefetchAsync
 * is called on this region, it will create a read-only copy of the data on the destination processor.
 * If any processor writes to this region, all copies of the corresponding page will be invalidated
 * except for the one where the write occurred. The \p device argument is ignored for this advice.
 * Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
 * that has a non-zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess.
 * Also, if a context is created on a device that does not have the device attribute
 * ::cudaDevAttrConcurrentManagedAccess set, then read-duplication will not occur until
 * all such contexts are destroyed.
 * If the memory region refers to valid system-allocated pageable memory, then the accessing device must
 * have a non-zero value for the device attribute ::cudaDevAttrPageableMemoryAccess for a read-only
 * copy to be created on that device. Note however that if the accessing device also has a non-zero value for the
 * device attribute ::cudaDevAttrPageableMemoryAccessUsesHostPageTables, then setting this advice
 * will not create a read-only copy when that device accesses this memory region.
 *
 * - ::cudaMemAdviceUnsetReadMostly: Undoes the effect of ::cudaMemAdviceReadMostly and also prevents the
 * Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
 * copies of the data will be collapsed into a single copy. The location for the collapsed
 * copy will be the preferred location if the page has a preferred location and one of the read-duplicated
 * copies was resident at that location. Otherwise, the location chosen is arbitrary.
 *
 * - ::cudaMemAdviseSetPreferredLocation: This advice sets the preferred location for the
 * data to be the memory belonging to \p device. Passing in cudaCpuDeviceId for \p device sets the
 * preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
 * device attribute ::cudaDevAttrConcurrentManagedAccess. Setting the preferred location
 * does not cause data to migrate to that location immediately. Instead, it guides the migration policy
 * when a fault occurs on that memory region. If the data is already in its preferred location and the
 * faulting processor can establish a mapping without requiring the data to be migrated, then
 * data migration will be avoided. On the other hand, if the data is not in its preferred location
 * or if a direct mapping cannot be established, then it will be migrated to the processor accessing
 * it. It is important to note that setting the preferred location does not prevent data prefetching
 * done using ::cudaMemPrefetchAsync.
 * Having a preferred location can override the page thrash detection and resolution logic in the Unified
 * Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
 * memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
 * if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
 * If ::cudaMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice, unless read accesses from
 * \p device will not result in a read-only copy being created on that device as outlined in description for
 * the advice ::cudaMemAdviseSetReadMostly.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::cudaDevAttrPageableMemoryAccess. Additionally, if \p device has
 * a non-zero value for the device attribute ::cudaDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect. Note however that this behavior may change in the future.
 *
 * - ::cudaMemAdviseUnsetPreferredLocation: Undoes the effect of ::cudaMemAdviseSetPreferredLocation
 * and changes the preferred location to none.
 *
 * - ::cudaMemAdviseSetAccessedBy: This advice implies that the data will be accessed by \p device.
 * Passing in ::cudaCpuDeviceId for \p device will set the advice for the CPU. If \p device is a GPU, then
 * the device attribute ::cudaDevAttrConcurrentManagedAccess must be non-zero.
 * This advice does not cause data migration and has no impact on the location of the data per se. Instead,
 * it causes the data to always be mapped in the specified processor's page tables, as long as the
 * location of the data permits a mapping to be established. If the data gets migrated for any reason,
 * the mappings are updated accordingly.
 * This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
 * Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
 * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
 * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
 * migration may be too high. But preventing faults can still help improve performance, and so having
 * a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
 * to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
 * ::cudaMemAdviceSetAccessedBy flag set for this data will now have its mapping updated to point to the
 * page in host memory.
 * If ::cudaMemAdviseSetReadMostly is also set on this memory region or any subset of it, then the
 * policies associated with that advice will override the policies of this advice. Additionally, if the
 * preferred location of this memory region or any subset of it is also \p device, then the policies
 * associated with ::cudaMemAdviseSetPreferredLocation will override the policies of this advice.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::cudaDevAttrPageableMemoryAccess. Additionally, if \p device has
 * a non-zero value for the device attribute ::cudaDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect.
 *
 * - ::cudaMemAdviseUnsetAccessedBy: Undoes the effect of ::cudaMemAdviseSetAccessedBy. Any mappings to
 * the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
 * If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
 * value for the device attribute ::cudaDevAttrPageableMemoryAccess. Additionally, if \p device has
 * a non-zero value for the device attribute ::cudaDevAttrPageableMemoryAccessUsesHostPageTables,
 * then this call has no effect.
 *
 * \param devPtr - Pointer to memory to set the advice for
 * \param count  - Size in bytes of the memory range
 * \param advice - Advice to be applied for the specified memory range
 * \param device - Device to apply the advice for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpyPeer, ::cudaMemcpyAsync,
 * ::cudaMemcpy3DPeerAsync, ::cudaMemPrefetchAsync,
 * ::cuMemAdvise
 */
extern __host__ cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device);

/**
* \brief Query an attribute of a given memory range
*
* Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
* memory range must refer to managed memory allocated via ::cudaMallocManaged or declared via
* __managed__ variables.
*
* The \p attribute parameter can take the following values:
* - ::cudaMemRangeAttributeReadMostly: If this attribute is specified, \p data will be interpreted
* as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
* memory range have read-duplication enabled, or 0 otherwise.
* - ::cudaMemRangeAttributePreferredLocation: If this attribute is specified, \p data will be
* interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
* id if all pages in the memory range have that GPU as their preferred location, or it will be cudaCpuDeviceId
* if all pages in the memory range have the CPU as their preferred location, or it will be cudaInvalidDeviceId
* if either all the pages don't have the same preferred location or some of the pages don't have a
* preferred location at all. Note that the actual location of the pages in the memory range at the time of
* the query may be different from the preferred location.
* - ::cudaMemRangeAttributeAccessedBy: If this attribute is specified, \p data will be interpreted
* as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
* will be a list of device ids that had ::cudaMemAdviceSetAccessedBy set for that entire memory range.
* If any device does not have that advice set for the entire memory range, that device will not be included.
* If \p data is larger than the number of devices that have that advice set for that memory range,
* cudaInvalidDeviceId will be returned in all the extra space provided. For ex., if \p dataSize is 12
* (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
* { 0, cudaInvalidDeviceId, cudaInvalidDeviceId }. If \p data is smaller than the number of devices that have
* that advice set, then only as many devices will be returned as can fit in the array. There is no
* guarantee on which specific devices will be returned, however.
* - ::cudaMemRangeAttributeLastPrefetchLocation: If this attribute is specified, \p data will be
* interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
* to which all pages in the memory range were prefetched explicitly via ::cudaMemPrefetchAsync. This will either be
* a GPU id or cudaCpuDeviceId depending on whether the last location for prefetch was a GPU or the CPU
* respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
* prefetched to the same location, cudaInvalidDeviceId will be returned. Note that this simply returns the
* last location that the applicaton requested to prefetch the memory range to. It gives no indication as to
* whether the prefetch operation to that location has completed or even begun.
*
* \param data      - A pointers to a memory location where the result
*                    of each attribute query will be written to.
* \param dataSize  - Array containing the size of data
* \param attribute - The attribute to query
* \param devPtr    - Start of the range to query
* \param count     - Size of the range to query
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemRangeGetAttributes, ::cudaMemPrefetchAsync,
 * ::cudaMemAdvise,
 * ::cuMemRangeGetAttribute
 */
extern __host__ cudaError_t CUDARTAPI cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count);

/**
 * \brief Query attributes of a given memory range.
 *
 * Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
 * memory range must refer to managed memory allocated via ::cudaMallocManaged or declared via
 * __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
 * entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
 * The results of the query will be stored in \p data.
 *
 * The list of supported attributes are given below. Please refer to ::cudaMemRangeGetAttribute for
 * attribute descriptions and restrictions.
 *
 * - ::cudaMemRangeAttributeReadMostly
 * - ::cudaMemRangeAttributePreferredLocation
 * - ::cudaMemRangeAttributeAccessedBy
 * - ::cudaMemRangeAttributeLastPrefetchLocation
 *
 * \param data          - A two-dimensional array containing pointers to memory
 *                        locations where the result of each attribute query will be written to.
 * \param dataSizes     - Array containing the sizes of each result
 * \param attributes    - An array of attributes to query
 *                        (numAttributes and the number of attributes in this array should match)
 * \param numAttributes - Number of attributes to query
 * \param devPtr        - Start of the range to query
 * \param count         - Size of the range to query
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemRangeGetAttribute, ::cudaMemAdvise,
 * ::cudaMemPrefetchAsync,
 * ::cuMemRangeGetAttributes
 */
extern __host__ cudaError_t CUDARTAPI cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);

/** @} */ /* END CUDART_MEMORY */

/**
 * \defgroup CUDART_MEMORY_DEPRECATED Memory Management [DEPRECATED]
 *
 * ___MANBRIEF___ deprecated memory management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes deprecated memory management functions of the CUDA runtime
 * application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * CUDA array \p dst starting at \p hOffset rows and \p wOffset bytes from
 * the upper left corner, where \p kind specifies the direction
 * of the copy, and must be one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyHtoA,
 * ::cuMemcpyDtoA
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the CUDA array \p src starting at \p hOffset rows
 * and \p wOffset bytes from the upper left corner to the memory area pointed to
 * by \p dst, where \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_sync
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyAtoH,
 * ::cuMemcpyAtoD
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the CUDA array \p src starting at \p hOffsetSrc
 * rows and \p wOffsetSrc bytes from the upper left corner to the CUDA array
 * \p dst starting at \p hOffsetDst rows and \p wOffsetDst bytes from the upper
 * left corner, where \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param dst        - Destination memory address
 * \param wOffsetDst - Destination starting X offset (columns in bytes)
 * \param hOffsetDst - Destination starting Y offset (rows)
 * \param src        - Source memory address
 * \param wOffsetSrc - Source starting X offset (columns in bytes)
 * \param hOffsetSrc - Source starting Y offset (rows)
 * \param count      - Size in bytes to copy
 * \param kind       - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyAtoA
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the memory area pointed to by \p src to the
 * CUDA array \p dst starting at \p hOffset rows and \p wOffset bytes from
 * the upper left corner, where \p kind specifies the
 * direction of the copy, and must be one of ::cudaMemcpyHostToHost,
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::cudaMemcpyToArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param wOffset - Destination starting X offset (columns in bytes)
 * \param hOffset - Destination starting Y offset (rows)
 * \param src     - Source memory address
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpyFromArrayAsync, ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyHtoAAsync,
 * ::cuMemcpy2DAsync
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/**
 * \brief Copies data between host and device
 *
 * \deprecated
 *
 * Copies \p count bytes from the CUDA array \p src starting at \p hOffset rows
 * and \p wOffset bytes from the upper left corner to the memory area pointed to
 * by \p dst, where \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * ::cudaMemcpyFromArrayAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If \p
 * kind is ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToHost and \p stream
 * is non-zero, the copy may overlap with operations in other streams.
 *
 * \param dst     - Destination memory address
 * \param src     - Source memory address
 * \param wOffset - Source starting X offset (columns in bytes)
 * \param hOffset - Source starting Y offset (rows)
 * \param count   - Size in bytes to copy
 * \param kind    - Type of transfer
 * \param stream  - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidMemcpyDirection
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D, ::cudaMemcpyToArray,
 * ::cudaMemcpy2DToArray, ::cudaMemcpyFromArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpyArrayToArray, ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpyToArrayAsync, ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync,
 * ::cuMemcpyAtoHAsync,
 * ::cuMemcpy2DAsync
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));

/** @} */ /* END CUDART_MEMORY_DEPRECATED */

/**
 * \defgroup CUDART_MEMORY_POOLS Stream Ordered Memory Allocator 
 *
 * ___MANBRIEF___ Functions for performing allocation and free operations in stream order.
 *                Functions for controlling the behavior of the underlying allocator.
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 * 
 *
 * @{
 *
 * \section CUDART_MEMORY_POOLS_overview overview
 *
 * The asynchronous allocator allows the user to allocate and free in stream order.
 * All asynchronous accesses of the allocation must happen between
 * the stream executions of the allocation and the free. If the memory is accessed
 * outside of the promised stream order, a use before allocation / use after free error
 * will cause undefined behavior.
 *
 * The allocator is free to reallocate the memory as long as it can guarantee
 * that compliant memory accesses will not overlap temporally.
 * The allocator may refer to internal stream ordering as well as inter-stream dependencies
 * (such as CUDA events and null stream dependencies) when establishing the temporal guarantee.
 * The allocator may also insert inter-stream dependencies to establish the temporal guarantee.
 *
 * \section CUDART_MEMORY_POOLS_support Supported Platforms
 *
 * Whether or not a device supports the integrated stream ordered memory allocator
 * may be queried by calling ::cudaDeviceGetAttribute() with the device attribute
 * ::cudaDevAttrMemoryPoolsSupported.
 */

/**
 * \brief Allocates memory with stream ordered semantics
 *
 * Inserts an allocation operation into \p hStream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the memory pool associated with the stream's device.
 *
 * \note The default memory pool of a device contains device memory from that device.
 * \note Basic stream ordering allows future work submitted into the same stream to use the allocation.
 *       Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
 *       operation completes before work submitted in a separate stream runs.
 * \note During stream capture, this function results in the creation of an allocation node.  In this case,
 *       the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 *       are used to set the node's creation parameters.
 *
 * \param[out] devPtr  - Returned device pointer
 * \param[in] size     - Number of bytes to allocate
 * \param[in] hStream  - The stream establishing the stream ordering contract and the memory pool to allocate from
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported,
 * ::cudaErrorOutOfMemory,
 * \notefnerr
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuMemAllocAsync,
 * \ref ::cudaMallocAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream)  "cudaMallocAsync (C++ API)", 
 * ::cudaMallocFromPoolAsync, ::cudaFreeAsync, ::cudaDeviceSetMemPool, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceGetMemPool, ::cudaMemPoolSetAccess, ::cudaMemPoolSetAttribute, ::cudaMemPoolGetAttribute
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);

/**
 * \brief Frees memory with stream ordered semantics
 *
 * Inserts a free operation into \p hStream.
 * The allocation must not be accessed after stream execution reaches the free.
 * After this API returns, accessing the memory from any subsequent work launched on the GPU
 * or querying its pointer attributes results in undefined behavior.
 *
 * \note During stream capture, this function results in the creation of a free node and
 *       must therefore be passed the address of a graph allocation.
 *
 * \param dptr - memory to free
 * \param hStream - The stream establishing the stream ordering promise
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported
 * \notefnerr
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cuMemFreeAsync, ::cudaMallocAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaFreeAsync(void *devPtr, cudaStream_t hStream);

/**
 * \brief Tries to release memory back to the OS
 *
 * Releases memory back to the OS until the pool contains fewer than minBytesToKeep
 * reserved bytes, or there is no more memory that the allocator can safely release.
 * The allocator cannot release OS allocations that back outstanding asynchronous allocations.
 * The OS allocations may happen at different granularity from the user allocations.
 *
 * \note: Allocations that have not been freed count as outstanding.
 * \note: Allocations that have been asynchronously freed but whose completion has
 *        not been observed on the host (eg. by a synchronize) can count as outstanding.
 *
 * \param[in] pool           - The memory pool to trim
 * \param[in] minBytesToKeep - If the pool has less than minBytesToKeep reserved,
 * the TrimTo operation is a no-op.  Otherwise the pool will be guaranteed to have
 * at least minBytesToKeep bytes reserved after the operation.
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_callback
 *
 * \sa ::cuMemPoolTrimTo, ::cudaMallocAsync, ::cudaFreeAsync, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceGetMemPool, ::cudaMemPoolCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);

/**
 * \brief Sets attributes of a memory pool
 *
 * Supported attributes are:
 * - ::cudaMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                    Amount of reserved memory in bytes to hold onto before trying
 *                    to release memory back to the OS. When more than the release
 *                    threshold bytes of memory are held by the memory pool, the
 *                    allocator will try to release memory back to the OS on the
 *                    next call to stream, event or context synchronize. (default 0)
 * - ::cudaMemPoolReuseFollowEventDependencies: (value type = int)
 *                    Allow ::cudaMallocAsync to use memory asynchronously freed
 *                    in another stream as long as a stream ordering dependency
 *                    of the allocating stream on the free action exists.
 *                    Cuda events and null stream interactions can create the required
 *                    stream ordered dependencies. (default enabled)
 * - ::cudaMemPoolReuseAllowOpportunistic: (value type = int)
 *                    Allow reuse of already completed frees when there is no dependency
 *                    between the free and allocation. (default enabled)
 * - ::cudaMemPoolReuseAllowInternalDependencies: (value type = int)
 *                    Allow ::cudaMallocAsync to insert new stream dependencies
 *                    in order to establish the stream ordering required to reuse
 *                    a piece of memory released by ::cudaFreeAsync (default enabled).
 * - ::cudaMemPoolAttrReservedMemHigh: (value type = cuuint64_t)
 *                    Reset the high watermark that tracks the amount of backing memory that was
 *                    allocated for the memory pool. It is illegal to set this attribute to a non-zero value.
 * - ::cudaMemPoolAttrUsedMemHigh: (value type = cuuint64_t)
 *                    Reset the high watermark that tracks the amount of used memory that was
 *                    allocated for the memory pool. It is illegal to set this attribute to a non-zero value.
 *
 * \param[in] pool  - The memory pool to modify
 * \param[in] attr  - The attribute to modify
 * \param[in] value - Pointer to the value to assign
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_callback
 *
 * \sa ::cuMemPoolSetAttribute, ::cudaMallocAsync, ::cudaFreeAsync, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceGetMemPool, ::cudaMemPoolCreate

 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value );

/**
 * \brief Gets attributes of a memory pool
 *
 * Supported attributes are:
 * - ::cudaMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                    Amount of reserved memory in bytes to hold onto before trying
 *                    to release memory back to the OS. When more than the release
 *                    threshold bytes of memory are held by the memory pool, the
 *                    allocator will try to release memory back to the OS on the
 *                    next call to stream, event or context synchronize. (default 0)
 * - ::cudaMemPoolReuseFollowEventDependencies: (value type = int)
 *                    Allow ::cudaMallocAsync to use memory asynchronously freed
 *                    in another stream as long as a stream ordering dependency
 *                    of the allocating stream on the free action exists.
 *                    Cuda events and null stream interactions can create the required
 *                    stream ordered dependencies. (default enabled)
 * - ::cudaMemPoolReuseAllowOpportunistic: (value type = int)
 *                    Allow reuse of already completed frees when there is no dependency
 *                    between the free and allocation. (default enabled)
 * - ::cudaMemPoolReuseAllowInternalDependencies: (value type = int)
 *                    Allow ::cudaMallocAsync to insert new stream dependencies
 *                    in order to establish the stream ordering required to reuse
 *                    a piece of memory released by ::cudaFreeAsync (default enabled).
 * - ::cudaMemPoolAttrReservedMemCurrent: (value type = cuuint64_t)
 *                    Amount of backing memory currently allocated for the mempool.
 * - ::cudaMemPoolAttrReservedMemHigh: (value type = cuuint64_t)
 *                    High watermark of backing memory allocated for the mempool since
 *                    the last time it was reset.
 * - ::cudaMemPoolAttrUsedMemCurrent: (value type = cuuint64_t)
 *                    Amount of memory from the pool that is currently in use by the application.
 * - ::cudaMemPoolAttrUsedMemHigh: (value type = cuuint64_t)
 *                    High watermark of the amount of memory from the pool that was in use by the
 *                    application since the last time it was reset.
 *
 * \param[in] pool  - The memory pool to get attributes of 
 * \param[in] attr  - The attribute to get
 * \param[in] value - Retrieved value 
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_callback
 *
 * \sa ::cuMemPoolGetAttribute, ::cudaMallocAsync, ::cudaFreeAsync, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceGetMemPool, ::cudaMemPoolCreate

 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value );

/**
 * \brief Controls visibility of pools between devices
 *
 * \param[in] pool  - The pool being modified
 * \param[in] map   - Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
 * \param[in] count - Number of descriptors in the map array.
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa ::cuMemPoolSetAccess, ::cudaMemPoolGetAccess, ::cudaMallocAsync, cudaFreeAsync
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count);

/**
 * \brief Returns the accessibility of a pool from a device
 *
 * Returns the accessibility of the pool's memory from the specified location.
 *
 * \param[out] flags   - the accessibility of the pool from the specified location
 * \param[in] memPool  - the pool being queried
 * \param[in] location - the location accessing the pool
 *
 * \sa ::cuMemPoolGetAccess, ::cudaMemPoolSetAccess
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location);

/**
 * \brief Creates a memory pool
 *
 * Creates a CUDA memory pool and returns the handle in \p pool.  The \p poolProps determines
 * the properties of the pool such as the backing device and IPC capabilities.
 *
 * By default, the pool's memory will be accessible from the device it is allocated on.
 *
 * \note Specifying cudaMemHandleTypeNone creates a memory pool that will not support IPC.
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported
 *
 * \sa ::cuMemPoolCreate, ::cudaDeviceSetMemPool, ::cudaMallocFromPoolAsync, ::cudaMemPoolExportToShareableHandle, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceGetMemPool

 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps);

/**
 * \brief Destroys the specified memory pool 
 *
 * If any pointers obtained from this pool haven't been freed or
 * the pool has free operations that haven't completed
 * when ::cudaMemPoolDestroy is invoked, the function will return immediately and the
 * resources associated with the pool will be released automatically
 * once there are no more outstanding allocations.
 *
 * Destroying the current mempool of a device sets the default mempool of
 * that device as the current mempool for that device.
 *
 * \note A device's default memory pool cannot be destroyed.
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa cuMemPoolDestroy, ::cudaFreeAsync, ::cudaDeviceSetMemPool, ::cudaDeviceGetDefaultMemPool, ::cudaDeviceGetMemPool, ::cudaMemPoolCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t memPool);

/**
 * \brief Allocates memory from a specified pool with stream ordered semantics.
 *
 * Inserts an allocation operation into \p hStream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the specified memory pool.
 *
 * \note
 *    -  The specified memory pool may be from a device different than that of the specified \p hStream.
 *
 *    -  Basic stream ordering allows future work submitted into the same stream to use the allocation.
 *       Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
 *       operation completes before work submitted in a separate stream runs.
 *
 * \note During stream capture, this function results in the creation of an allocation node.  In this case,
 *       the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 *       are used to set the node's creation parameters.
 *
 * \param[out] ptr     - Returned device pointer
 * \param[in] bytesize - Number of bytes to allocate
 * \param[in] memPool  - The pool to allocate from
 * \param[in] stream   - The stream establishing the stream ordering semantic
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported,
 * ::cudaErrorOutOfMemory
 *
 * \sa ::cuMemAllocFromPoolAsync,
 * \ref ::cudaMallocAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream)  "cudaMallocAsync (C++ API)", 
 * ::cudaMallocAsync, ::cudaFreeAsync, ::cudaDeviceGetDefaultMemPool, ::cudaMemPoolCreate, ::cudaMemPoolSetAccess, ::cudaMemPoolSetAttribute
 */
extern __host__ cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);

/**
 * \brief Exports a memory pool to the requested handle type.
 *
 * Given an IPC capable mempool, create an OS handle to share the pool with another process.
 * A recipient process can convert the shareable handle into a mempool with ::cudaMemPoolImportFromShareableHandle.
 * Individual pointers can then be shared with the ::cudaMemPoolExportPointer and ::cudaMemPoolImportPointer APIs.
 * The implementation of what the shareable handle is and how it can be transferred is defined by the requested
 * handle type.
 *
 * \note: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than cudaMemHandleTypeNone.
 *
 * \param[out] handle_out  - pointer to the location in which to store the requested handle 
 * \param[in] pool         - pool to export
 * \param[in] handleType   - the type of handle to create
 * \param[in] flags        - must be 0
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorOutOfMemory
 *
 * \sa ::cuMemPoolExportToShareableHandle, ::cudaMemPoolImportFromShareableHandle, ::cudaMemPoolExportPointer, ::cudaMemPoolImportPointer
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void                            *shareableHandle,
    cudaMemPool_t                    memPool,
    enum cudaMemAllocationHandleType handleType,
    unsigned int                     flags);

/**
 * \brief imports a memory pool from a shared handle.
 *
 * Specific allocations can be imported from the imported pool with ::cudaMemPoolImportPointer.
 *
 * \note Imported memory pools do not support creating new allocations.
 *       As such imported memory pools may not be used in ::cudaDeviceSetMemPool
 *       or ::cudaMallocFromPoolAsync calls.
 *
 * \param[out] pool_out    - Returned memory pool
 * \param[in] handle       - OS handle of the pool to open
 * \param[in] handleType   - The type of handle being imported
 * \param[in] flags        - must be 0
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorOutOfMemory
 *
 * \sa ::cuMemPoolImportFromShareableHandle, ::cudaMemPoolExportToShareableHandle, ::cudaMemPoolExportPointer, ::cudaMemPoolImportPointer
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t                   *memPool,
    void                            *shareableHandle,
    enum cudaMemAllocationHandleType handleType,
    unsigned int                     flags);

/**
 * \brief Export data to share a memory pool allocation between processes.
 *
 * Constructs \p shareData_out for sharing a specific allocation from an already shared memory pool.
 * The recipient process can import the allocation with the ::cudaMemPoolImportPointer api.
 * The data is not a handle and may be shared through any IPC mechanism.
 *
 * \param[out] shareData_out - Returned export data
 * \param[in] ptr            - pointer to memory being exported
 *
 * \returns
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorOutOfMemory
 *
 * \sa ::cuMemPoolExportPointer, ::cudaMemPoolExportToShareableHandle, ::cudaMemPoolImportFromShareableHandle, ::cudaMemPoolImportPointer
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr);

/**
 * \brief Import a memory pool allocation from another process.
 *
 * Returns in \p ptr_out a pointer to the imported memory.
 * The imported memory must not be accessed before the allocation operation completes
 * in the exporting process. The imported memory must be freed from all importing processes before
 * being freed in the exporting process. The pointer may be freed with cudaFree
 * or cudaFreeAsync.  If ::cudaFreeAsync is used, the free must be completed
 * on the importing process before the free operation on the exporting process.
 *
 * \note The ::cudaFreeAsync api may be used in the exporting process before
 *       the ::cudaFreeAsync operation completes in its stream as long as the
 *       ::cudaFreeAsync in the exporting process specifies a stream with
 *       a stream dependency on the importing process's ::cudaFreeAsync.
 *
 * \param[out] ptr_out  - pointer to imported memory
 * \param[in] pool      - pool from which to import
 * \param[in] shareData - data specifying the memory to import
 *
 * \returns
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_OUT_OF_MEMORY
 *
 * \sa ::cuMemPoolImportPointer, ::cudaMemPoolExportToShareableHandle, ::cudaMemPoolImportFromShareableHandle, ::cudaMemPoolExportPointer
 */
extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData);

/** @} */ /* END CUDART_MEMORY_POOLS */

/**
 * \defgroup CUDART_UNIFIED Unified Addressing
 *
 * ___MANBRIEF___ unified addressing functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the unified addressing functions of the CUDA 
 * runtime application programming interface.
 *
 * @{
 *
 * \section CUDART_UNIFIED_overview Overview
 *
 * CUDA devices can share a unified address space with the host.  
 * For these devices there is no distinction between a device
 * pointer and a host pointer -- the same pointer value may be 
 * used to access memory from the host program and from a kernel 
 * running on the device (with exceptions enumerated below).
 *
 * \section CUDART_UNIFIED_support Supported Platforms
 * 
 * Whether or not a device supports unified addressing may be 
 * queried by calling ::cudaGetDeviceProperties() with the device 
 * property ::cudaDeviceProp::unifiedAddressing.
 *
 * Unified addressing is automatically enabled in 64-bit processes .
 *
 * \section CUDART_UNIFIED_lookup Looking Up Information from Pointer Values
 *
 * It is possible to look up information about the memory which backs a 
 * pointer value.  For instance, one may want to know if a pointer points
 * to host or device memory.  As another example, in the case of device 
 * memory, one may want to know on which CUDA device the memory 
 * resides.  These properties may be queried using the function 
 * ::cudaPointerGetAttributes()
 *
 * Since pointers are unique, it is not necessary to specify information
 * about the pointers specified to ::cudaMemcpy() and other copy functions.  
 * The copy direction ::cudaMemcpyDefault may be used to specify that the 
 * CUDA runtime should infer the location of the pointer from its value.
 *
 * \section CUDART_UNIFIED_automaphost Automatic Mapping of Host Allocated Host Memory
 *
 * All host memory allocated through all devices using ::cudaMallocHost() and
 * ::cudaHostAlloc() is always directly accessible from all devices that 
 * support unified addressing.  This is the case regardless of whether or 
 * not the flags ::cudaHostAllocPortable and ::cudaHostAllocMapped are 
 * specified.
 *
 * The pointer value through which allocated host memory may be accessed 
 * in kernels on all devices that support unified addressing is the same 
 * as the pointer value through which that memory is accessed on the host.
 * It is not necessary to call ::cudaHostGetDevicePointer() to get the device 
 * pointer for these allocations.  
 *
 * Note that this is not the case for memory allocated using the flag
 * ::cudaHostAllocWriteCombined, as discussed below.
 *
 * \section CUDART_UNIFIED_autopeerregister Direct Access of Peer Memory
 
 * Upon enabling direct access from a device that supports unified addressing 
 * to another peer device that supports unified addressing using 
 * ::cudaDeviceEnablePeerAccess() all memory allocated in the peer device using 
 * ::cudaMalloc() and ::cudaMallocPitch() will immediately be accessible 
 * by the current device.  The device pointer value through 
 * which any peer's memory may be accessed in the current device 
 * is the same pointer value through which that memory may be 
 * accessed from the peer device. 
 *
 * \section CUDART_UNIFIED_exceptions Exceptions, Disjoint Addressing
 * 
 * Not all memory may be accessed on devices through the same pointer
 * value through which they are accessed on the host.  These exceptions
 * are host memory registered using ::cudaHostRegister() and host memory
 * allocated using the flag ::cudaHostAllocWriteCombined.  For these 
 * exceptions, there exists a distinct host and device address for the
 * memory.  The device address is guaranteed to not overlap any valid host
 * pointer range and is guaranteed to have the same value across all devices
 * that support unified addressing.  
 * 
 * This device address may be queried using ::cudaHostGetDevicePointer() 
 * when a device using unified addressing is current.  Either the host 
 * or the unified device pointer value may be used to refer to this memory 
 * in ::cudaMemcpy() and similar functions using the ::cudaMemcpyDefault 
 * memory direction.
 *
 */

/**
 * \brief Returns attributes about a specified pointer
 *
 * Returns in \p *attributes the attributes of the pointer \p ptr.
 * If pointer was not allocated in, mapped by or registered with context
 * supporting unified addressing ::cudaErrorInvalidValue is returned.
 *
 * \note In CUDA 11.0 forward passing host pointer will return ::cudaMemoryTypeUnregistered
 * in ::cudaPointerAttributes::type and call will return ::cudaSuccess.
 *
 * The ::cudaPointerAttributes structure is defined as:
 * \code
    struct cudaPointerAttributes {
        enum cudaMemoryType type;
        int device;
        void *devicePointer;
        void *hostPointer;
    }
    \endcode
 * In this structure, the individual fields mean
 *
 * - \ref ::cudaPointerAttributes::type identifies type of memory. It can be
 *    ::cudaMemoryTypeUnregistered for unregistered host memory,
 *    ::cudaMemoryTypeHost for registered host memory, ::cudaMemoryTypeDevice for device
 *    memory or  ::cudaMemoryTypeManaged for managed memory.
 *
 * - \ref ::cudaPointerAttributes::device "device" is the device against which
 *   \p ptr was allocated.  If \p ptr has memory type ::cudaMemoryTypeDevice
 *   then this identifies the device on which the memory referred to by \p ptr
 *   physically resides.  If \p ptr has memory type ::cudaMemoryTypeHost then this
 *   identifies the device which was current when the allocation was made
 *   (and if that device is deinitialized then this allocation will vanish
 *   with that device's state).
 *
 * - \ref ::cudaPointerAttributes::devicePointer "devicePointer" is
 *   the device pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the current device.
 *   If the memory referred to by \p ptr cannot be accessed directly by the 
 *   current device then this is NULL.  
 *
 * - \ref ::cudaPointerAttributes::hostPointer "hostPointer" is
 *   the host pointer alias through which the memory referred to by \p ptr
 *   may be accessed on the host.
 *   If the memory referred to by \p ptr cannot be accessed directly by the
 *   host then this is NULL.
 *
 * \param attributes - Attributes for the specified pointer
 * \param ptr        - Pointer to get attributes for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaGetDeviceCount, ::cudaGetDevice, ::cudaSetDevice,
 * ::cudaChooseDevice,
 * ::cuPointerGetAttributes
 */
extern __host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);

/** @} */ /* END CUDART_UNIFIED */

/**
 * \defgroup CUDART_PEER Peer Device Memory Access
 *
 * ___MANBRIEF___ peer device memory access functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the peer device memory access functions of the CUDA runtime
 * application programming interface.
 *
 * @{
 */

/**
 * \brief Queries if a device may directly access a peer device's memory.
 *
 * Returns in \p *canAccessPeer a value of 1 if device \p device is capable of
 * directly accessing memory from \p peerDevice and 0 otherwise.  If direct
 * access of \p peerDevice from \p device is possible, then access may be
 * enabled by calling ::cudaDeviceEnablePeerAccess().
 *
 * \param canAccessPeer - Returned access capability
 * \param device        - Device from which allocations on \p peerDevice are to
 *                        be directly accessed.
 * \param peerDevice    - Device on which the allocations to be directly accessed 
 *                        by \p device reside.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceEnablePeerAccess,
 * ::cudaDeviceDisablePeerAccess,
 * ::cuDeviceCanAccessPeer
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);

/**
 * \brief Enables direct access to memory allocations on a peer device.
 *
 * On success, all allocations from \p peerDevice will immediately be accessible by
 * the current device.  They will remain accessible until access is explicitly
 * disabled using ::cudaDeviceDisablePeerAccess() or either device is reset using
 * ::cudaDeviceReset().
 *
 * Note that access granted by this call is unidirectional and that in order to access
 * memory on the current device from \p peerDevice, a separate symmetric call 
 * to ::cudaDeviceEnablePeerAccess() is required.
 *
 * Note that there are both device-wide and system-wide limitations per system
 * configuration, as noted in the CUDA Programming Guide under the section
 * "Peer-to-Peer Memory Access".
 *
 * Returns ::cudaErrorInvalidDevice if ::cudaDeviceCanAccessPeer() indicates
 * that the current device cannot directly access memory from \p peerDevice.
 *
 * Returns ::cudaErrorPeerAccessAlreadyEnabled if direct access of
 * \p peerDevice from the current device has already been enabled.
 *
 * Returns ::cudaErrorInvalidValue if \p flags is not 0.
 *
 * \param peerDevice  - Peer device to enable direct access to from the current device
 * \param flags       - Reserved for future use and must be set to 0
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorPeerAccessAlreadyEnabled,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceCanAccessPeer,
 * ::cudaDeviceDisablePeerAccess,
 * ::cuCtxEnablePeerAccess
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);

/**
 * \brief Disables direct access to memory allocations on a peer device.
 *
 * Returns ::cudaErrorPeerAccessNotEnabled if direct access to memory on
 * \p peerDevice has not yet been enabled from the current device.
 *
 * \param peerDevice - Peer device to disable direct access to
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorPeerAccessNotEnabled,
 * ::cudaErrorInvalidDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaDeviceCanAccessPeer,
 * ::cudaDeviceEnablePeerAccess,
 * ::cuCtxDisablePeerAccess
 */
extern __host__ cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice);

/** @} */ /* END CUDART_PEER */

/** \defgroup CUDART_OPENGL OpenGL Interoperability */

/** \defgroup CUDART_OPENGL_DEPRECATED OpenGL Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D9 Direct3D 9 Interoperability */

/** \defgroup CUDART_D3D9_DEPRECATED Direct3D 9 Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D10 Direct3D 10 Interoperability */

/** \defgroup CUDART_D3D10_DEPRECATED Direct3D 10 Interoperability [DEPRECATED] */

/** \defgroup CUDART_D3D11 Direct3D 11 Interoperability */

/** \defgroup CUDART_D3D11_DEPRECATED Direct3D 11 Interoperability [DEPRECATED] */

/** \defgroup CUDART_VDPAU VDPAU Interoperability */

/** \defgroup CUDART_EGL EGL Interoperability */

/**
 * \defgroup CUDART_INTEROP Graphics Interoperability
 *
 * ___MANBRIEF___ graphics interoperability functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graphics interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Unregisters a graphics resource for access by CUDA
 *
 * Unregisters the graphics resource \p resource so it is not accessible by
 * CUDA unless registered again.
 *
 * If \p resource is invalid then ::cudaErrorInvalidResourceHandle is
 * returned.
 *
 * \param resource - Resource to unregister
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::cudaGraphicsD3D9RegisterResource,
 * ::cudaGraphicsD3D10RegisterResource,
 * ::cudaGraphicsD3D11RegisterResource,
 * ::cudaGraphicsGLRegisterBuffer,
 * ::cudaGraphicsGLRegisterImage,
 * ::cuGraphicsUnregisterResource
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);

/**
 * \brief Set usage flags for mapping a graphics resource
 *
 * Set \p flags for mapping the graphics resource \p resource.
 *
 * Changes to \p flags will take effect the next time \p resource is mapped.
 * The \p flags argument may be any of the following:
 * - ::cudaGraphicsMapFlagsNone: Specifies no hints about how \p resource will
 *     be used. It is therefore assumed that CUDA may read from or write to \p resource.
 * - ::cudaGraphicsMapFlagsReadOnly: Specifies that CUDA will not write to \p resource.
 * - ::cudaGraphicsMapFlagsWriteDiscard: Specifies CUDA will not read from \p resource and will
 *   write over the entire contents of \p resource, so none of the data
 *   previously stored in \p resource will be preserved.
 *
 * If \p resource is presently mapped for access by CUDA then ::cudaErrorUnknown is returned.
 * If \p flags is not one of the above values then ::cudaErrorInvalidValue is returned.
 *
 * \param resource - Registered resource to set flags for
 * \param flags    - Parameters for resource mapping
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphicsMapResources,
 * ::cuGraphicsResourceSetMapFlags
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);

/**
 * \brief Map graphics resources for access by CUDA
 *
 * Maps the \p count graphics resources in \p resources for access by CUDA.
 *
 * The resources in \p resources may be accessed by CUDA until they
 * are unmapped. The graphics API from which \p resources were registered
 * should not access any resources while they are mapped by CUDA. If an
 * application does so, the results are undefined.
 *
 * This function provides the synchronization guarantee that any graphics calls
 * issued before ::cudaGraphicsMapResources() will complete before any subsequent CUDA
 * work issued in \p stream begins.
 *
 * If \p resources contains any duplicate entries then ::cudaErrorInvalidResourceHandle
 * is returned. If any of \p resources are presently mapped for access by
 * CUDA then ::cudaErrorUnknown is returned.
 *
 * \param count     - Number of resources to map
 * \param resources - Resources to map for CUDA
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cudaGraphicsUnmapResources,
 * ::cuGraphicsMapResources
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0));

/**
 * \brief Unmap graphics resources.
 *
 * Unmaps the \p count graphics resources in \p resources.
 *
 * Once unmapped, the resources in \p resources may not be accessed by CUDA
 * until they are mapped again.
 *
 * This function provides the synchronization guarantee that any CUDA work issued
 * in \p stream before ::cudaGraphicsUnmapResources() will complete before any
 * subsequently issued graphics work begins.
 *
 * If \p resources contains any duplicate entries then ::cudaErrorInvalidResourceHandle
 * is returned. If any of \p resources are not presently mapped for access by
 * CUDA then ::cudaErrorUnknown is returned.
 *
 * \param count     - Number of resources to unmap
 * \param resources - Resources to unmap
 * \param stream    - Stream for synchronization
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphicsMapResources,
 * ::cuGraphicsUnmapResources
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0));

/**
 * \brief Get an device pointer through which to access a mapped graphics resource.
 *
 * Returns in \p *devPtr a pointer through which the mapped graphics resource
 * \p resource may be accessed.
 * Returns in \p *size the size of the memory in bytes which may be accessed from that pointer.
 * The value set in \p devPtr may change every time that \p resource is mapped.
 *
 * If \p resource is not a buffer then it cannot be accessed via a pointer and
 * ::cudaErrorUnknown is returned.
 * If \p resource is not mapped then ::cudaErrorUnknown is returned.
 * *
 * \param devPtr     - Returned pointer through which \p resource may be accessed
 * \param size       - Returned size of the buffer accessible starting at \p *devPtr
 * \param resource   - Mapped resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphicsMapResources,
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cuGraphicsResourceGetMappedPointer
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);

/**
 * \brief Get an array through which to access a subresource of a mapped graphics resource.
 *
 * Returns in \p *array an array through which the subresource of the mapped
 * graphics resource \p resource which corresponds to array index \p arrayIndex
 * and mipmap level \p mipLevel may be accessed.  The value set in \p array may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::cudaErrorUnknown is returned.
 * If \p arrayIndex is not a valid array index for \p resource then
 * ::cudaErrorInvalidValue is returned.
 * If \p mipLevel is not a valid mipmap level for \p resource then
 * ::cudaErrorInvalidValue is returned.
 * If \p resource is not mapped then ::cudaErrorUnknown is returned.
 *
 * \param array       - Returned array through which a subresource of \p resource may be accessed
 * \param resource    - Mapped resource to access
 * \param arrayIndex  - Array index for array textures or cubemap face
 *                      index as defined by ::cudaGraphicsCubeFace for
 *                      cubemap textures for the subresource to access
 * \param mipLevel    - Mipmap level for the subresource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGraphicsSubResourceGetMappedArray
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);

/**
 * \brief Get a mipmapped array through which to access a mapped graphics resource.
 *
 * Returns in \p *mipmappedArray a mipmapped array through which the mapped
 * graphics resource \p resource may be accessed. The value set in \p mipmappedArray may
 * change every time that \p resource is mapped.
 *
 * If \p resource is not a texture then it cannot be accessed via an array and
 * ::cudaErrorUnknown is returned.
 * If \p resource is not mapped then ::cudaErrorUnknown is returned.
 *
 * \param mipmappedArray - Returned mipmapped array through which \p resource may be accessed
 * \param resource       - Mapped resource to access
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorUnknown
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGraphicsResourceGetMappedMipmappedArray
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource);

/** @} */ /* END CUDART_INTEROP */

/**
 * \defgroup CUDART_TEXTURE Texture Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ texture reference management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture reference management functions
 * of the CUDA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Binds a memory area to a texture
 *
 * \deprecated
 *
 * Binds \p size bytes of the memory area pointed to by \p devPtr to the
 * texture reference \p texref. \p desc describes how the memory is interpreted
 * when fetching values from the texture. Any memory previously bound to
 * \p texref is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture()"
 * returns in \p *offset a byte offset that
 * must be applied to texture fetches in order to read from the desired memory.
 * This offset must be divided by the texel size and passed to kernels that
 * read from the texture so they can be applied to the ::tex1Dfetch() function.
 * If the device memory pointer was returned from ::cudaMalloc(), the offset is
 * guaranteed to be 0 and NULL may be passed as the \p offset parameter.
 *
 * The total number of elements (or texels) in the linear address range
 * cannot exceed ::cudaDeviceProp::maxTexture1DLinear[0].
 * The number of elements is computed as (\p size / elementSize),
 * where elementSize is determined from \p desc.
 *
 * \param offset - Offset in bytes
 * \param texref - Texture to bind
 * \param devPtr - Memory area on device
 * \param desc   - Channel format
 * \param size   - Size of the memory area pointed to by devPtr
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t) "cudaBindTexture (C++ API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)",
 * ::cuTexRefSetAddress,
 * ::cuTexRefSetAddressMode,
 * ::cuTexRefSetFormat,
 * ::cuTexRefSetFlags,
 * ::cuTexRefSetBorderColor
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));

/**
 * \brief Binds a 2D memory area to a texture
 *
 * \deprecated
 *
 * Binds the 2D memory area pointed to by \p devPtr to the
 * texture reference \p texref. The size of the area is constrained by
 * \p width in texel units, \p height in texel units, and \p pitch in byte
 * units. \p desc describes how the memory is interpreted when fetching values
 * from the texture. Any memory previously bound to \p texref is unbound.
 *
 * Since the hardware enforces an alignment requirement on texture base
 * addresses, ::cudaBindTexture2D() returns in \p *offset a byte offset that
 * must be applied to texture fetches in order to read from the desired memory.
 * This offset must be divided by the texel size and passed to kernels that
 * read from the texture so they can be applied to the ::tex2D() function.
 * If the device memory pointer was returned from ::cudaMalloc(), the offset is
 * guaranteed to be 0 and NULL may be passed as the \p offset parameter.
 *
 * \p width and \p height, which are specified in elements (or texels), cannot
 * exceed ::cudaDeviceProp::maxTexture2DLinear[0] and ::cudaDeviceProp::maxTexture2DLinear[1]
 * respectively. \p pitch, which is specified in bytes, cannot exceed
 * ::cudaDeviceProp::maxTexture2DLinear[2].
 *
 * The driver returns ::cudaErrorInvalidValue if \p pitch is not a multiple of
 * ::cudaDeviceProp::texturePitchAlignment.
 *
 * \param offset - Offset in bytes
 * \param texref - Texture reference to bind
 * \param devPtr - 2D memory area on device
 * \param desc   - Channel format
 * \param width  - Width in texel units
 * \param height - Height in texel units
 * \param pitch  - Pitch in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t, size_t, size_t) "cudaBindTexture2D (C++ API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct texture<T, dim, readMode>&, const void*, size_t, size_t, size_t) "cudaBindTexture2D (C++ API, inherited channel descriptor)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)",
 * ::cuTexRefSetAddress2D,
 * ::cuTexRefSetFormat,
 * ::cuTexRefSetFlags,
 * ::cuTexRefSetAddressMode,
 * ::cuTexRefSetBorderColor
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch);

/**
 * \brief Binds an array to a texture
 *
 * \deprecated
 *
 * Binds the CUDA array \p array to the texture reference \p texref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the texture. Any CUDA array previously bound to \p texref is unbound.
 *
 * \param texref - Texture to bind
 * \param array  - Memory array on device
 * \param desc   - Channel format
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindTextureToArray (C++ API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)",
 * ::cuTexRefSetArray,
 * ::cuTexRefSetFormat,
 * ::cuTexRefSetFlags,
 * ::cuTexRefSetAddressMode,
 * ::cuTexRefSetFilterMode,
 * ::cuTexRefSetBorderColor,
 * ::cuTexRefSetMaxAnisotropy
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);

/**
 * \brief Binds a mipmapped array to a texture
 *
 * \deprecated
 *
 * Binds the CUDA mipmapped array \p mipmappedArray to the texture reference \p texref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the texture. Any CUDA mipmapped array previously bound to \p texref is unbound.
 *
 * \param texref         - Texture to bind
 * \param mipmappedArray - Memory mipmapped array on device
 * \param desc           - Channel format
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindTextureToArray (C++ API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)",
 * ::cuTexRefSetMipmappedArray,
 * ::cuTexRefSetMipmapFilterMode,
 * ::cuTexRefSetMipmapLevelClamp,
 * ::cuTexRefSetMipmapLevelBias,
 * ::cuTexRefSetFormat,
 * ::cuTexRefSetFlags,
 * ::cuTexRefSetAddressMode,
 * ::cuTexRefSetBorderColor,
 * ::cuTexRefSetMaxAnisotropy
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc);

/**
 * \brief Unbinds a texture
 *
 * \deprecated
 *
 * Unbinds the texture bound to \p texref. If \p texref is not currently bound, no operation is performed.
 *
 * \param texref - Texture to unbind
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct texture< T, dim, readMode>&) "cudaUnbindTexture (C++ API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)"
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref);

/**
 * \brief Get the alignment offset of a texture
 *
 * \deprecated
 *
 * Returns in \p *offset the offset that was returned when texture reference
 * \p texref was bound.
 *
 * \param offset - Offset of texture reference in bytes
 * \param texref - Texture to get offset of
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidTexture,
 * ::cudaErrorInvalidTextureBinding
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct texture< T, dim, readMode>&) "cudaGetTextureAlignmentOffset (C++ API)"
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);

/**
 * \brief Get the texture reference associated with a symbol
 *
 * \deprecated
 *
 * Returns in \p *texref the structure associated to the texture reference
 * defined by symbol \p symbol.
 *
 * \param texref - Texture reference associated with symbol
 * \param symbol - Texture to get reference for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidTexture
 * \notefnerr
 * \note_string_api_deprecation_50
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaGetChannelDesc,
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct textureReference*) "cudaGetTextureAlignmentOffset (C API)",
 * \ref ::cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t) "cudaBindTexture (C API)",
 * \ref ::cudaBindTexture2D(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t, size_t, size_t) "cudaBindTexture2D (C API)",
 * \ref ::cudaBindTextureToArray(const struct textureReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindTextureToArray (C API)",
 * \ref ::cudaUnbindTexture(const struct textureReference*) "cudaUnbindTexture (C API)",
 * ::cuModuleGetTexRef
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const void *symbol);

/** @} */ /* END CUDART_TEXTURE */

/**
 * \defgroup CUDART_SURFACE Surface Reference Management [DEPRECATED]
 *
 * ___MANBRIEF___ surface reference management functions of the CUDA runtime
 * API (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level surface reference management functions
 * of the CUDA runtime application programming interface.
 *
 * Some functions have overloaded C++ API template versions documented separately in the
 * \ref CUDART_HIGHLEVEL "C++ API Routines" module.
 *
 * @{
 */

/**
 * \brief Binds an array to a surface
 *
 * \deprecated
 *
 * Binds the CUDA array \p array to the surface reference \p surfref.
 * \p desc describes how the memory is interpreted when fetching values from
 * the surface. Any CUDA array previously bound to \p surfref is unbound.
 *
 * \param surfref - Surface to bind
 * \param array  - Memory array on device
 * \param desc   - Channel format
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSurface
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaBindSurfaceToArray(const struct surface< T, dim>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindSurfaceToArray (C++ API)",
 * \ref ::cudaBindSurfaceToArray(const struct surface< T, dim>&, cudaArray_const_t) "cudaBindSurfaceToArray (C++ API, inherited channel descriptor)",
 * ::cudaGetSurfaceReference,
 * ::cuSurfRefSetArray
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindSurfaceToArray(const struct surfaceReference *surfref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);

/**
 * \brief Get the surface reference associated with a symbol
 *
 * \deprecated
 *
 * Returns in \p *surfref the structure associated to the surface reference
 * defined by symbol \p symbol.
 *
 * \param surfref - Surface reference associated with symbol
 * \param symbol - Surface to get reference for
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSurface
 * \notefnerr
 * \note_string_api_deprecation_50
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaBindSurfaceToArray(const struct surfaceReference*, cudaArray_const_t, const struct cudaChannelFormatDesc*) "cudaBindSurfaceToArray (C API)",
 * ::cuModuleGetSurfRef
 */
extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetSurfaceReference(const struct surfaceReference **surfref, const void *symbol);

/** @} */ /* END CUDART_SURFACE */

/**
 * \defgroup CUDART_TEXTURE_OBJECT Texture Object Management
 *
 * ___MANBRIEF___ texture object management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the CUDA runtime application programming interface. The texture
 * object API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Get the channel descriptor of an array
 *
 * Returns in \p *desc the channel descriptor of the CUDA array \p array.
 *
 * \param desc  - Channel format
 * \param array - Memory array on device
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) "cudaCreateChannelDesc (C API)",
 * ::cudaCreateTextureObject, ::cudaCreateSurfaceObject
 */
extern __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array);

/**
 * \brief Returns a channel descriptor using the specified format
 *
 * Returns a channel descriptor with format \p f and number of bits of each
 * component \p x, \p y, \p z, and \p w.  The ::cudaChannelFormatDesc is
 * defined as:
 * \code
  struct cudaChannelFormatDesc {
    int x, y, z, w;
    enum cudaChannelFormatKind f;
  };
 * \endcode
 *
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, or ::cudaChannelFormatKindFloat.
 *
 * \param x - X component
 * \param y - Y component
 * \param z - Z component
 * \param w - W component
 * \param f - Channel format
 *
 * \return
 * Channel descriptor with format \p f
 *
 * \sa \ref ::cudaCreateChannelDesc(void) "cudaCreateChannelDesc (C++ API)",
 * ::cudaGetChannelDesc, ::cudaCreateTextureObject, ::cudaCreateSurfaceObject
 */
extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);

/**
 * \brief Creates a texture object
 *
 * Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
 * the data to texture from. \p pTexDesc describes how the data should be sampled.
 * \p pResViewDesc is an optional argument that specifies an alternate format for
 * the data described by \p pResDesc, and also describes the subresource region
 * to restrict access to when texturing. \p pResViewDesc can only be specified if
 * the type of resource is a CUDA array or a CUDA mipmapped array.
 *
 * Texture objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a texture object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * The ::cudaResourceDesc structure is defined as:
 * \code
        struct cudaResourceDesc {
            enum cudaResourceType resType;
            
            union {
                struct {
                    cudaArray_t array;
                } array;
                struct {
                    cudaMipmappedArray_t mipmap;
                } mipmap;
                struct {
                    void *devPtr;
                    struct cudaChannelFormatDesc desc;
                    size_t sizeInBytes;
                } linear;
                struct {
                    void *devPtr;
                    struct cudaChannelFormatDesc desc;
                    size_t width;
                    size_t height;
                    size_t pitchInBytes;
                } pitch2D;
            } res;
        };
 * \endcode
 * where:
 * - ::cudaResourceDesc::resType specifies the type of resource to texture from.
 * CUresourceType is defined as:
 * \code
        enum cudaResourceType {
            cudaResourceTypeArray          = 0x00,
            cudaResourceTypeMipmappedArray = 0x01,
            cudaResourceTypeLinear         = 0x02,
            cudaResourceTypePitch2D        = 0x03
        };
 * \endcode
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypeArray, ::cudaResourceDesc::res::array::array
 * must be set to a valid CUDA array handle.
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypeMipmappedArray, ::cudaResourceDesc::res::mipmap::mipmap
 * must be set to a valid CUDA mipmapped array handle and ::cudaTextureDesc::normalizedCoords must be set to true.
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypeLinear, ::cudaResourceDesc::res::linear::devPtr
 * must be set to a valid device pointer, that is aligned to ::cudaDeviceProp::textureAlignment.
 * ::cudaResourceDesc::res::linear::desc describes the format and the number of components per array element. ::cudaResourceDesc::res::linear::sizeInBytes
 * specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed 
 * ::cudaDeviceProp::maxTexture1DLinear. The number of elements is computed as (sizeInBytes / sizeof(desc)).
 *
 * \par
 * If ::cudaResourceDesc::resType is set to ::cudaResourceTypePitch2D, ::cudaResourceDesc::res::pitch2D::devPtr
 * must be set to a valid device pointer, that is aligned to ::cudaDeviceProp::textureAlignment.
 * ::cudaResourceDesc::res::pitch2D::desc describes the format and the number of components per array element. ::cudaResourceDesc::res::pitch2D::width
 * and ::cudaResourceDesc::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
 * ::cudaDeviceProp::maxTexture2DLinear[0] and ::cudaDeviceProp::maxTexture2DLinear[1] respectively.
 * ::cudaResourceDesc::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to 
 * ::cudaDeviceProp::texturePitchAlignment. Pitch cannot exceed ::cudaDeviceProp::maxTexture2DLinear[2].
 *
 *
 * The ::cudaTextureDesc struct is defined as
 * \code
        struct cudaTextureDesc {
            enum cudaTextureAddressMode addressMode[3];
            enum cudaTextureFilterMode  filterMode;
            enum cudaTextureReadMode    readMode;
            int                         sRGB;
            float                       borderColor[4];
            int                         normalizedCoords;
            unsigned int                maxAnisotropy;
            enum cudaTextureFilterMode  mipmapFilterMode;
            float                       mipmapLevelBias;
            float                       minMipmapLevelClamp;
            float                       maxMipmapLevelClamp;
            int                         disableTrilinearOptimization;
            int                         seamlessCubemap;
        };
 * \endcode
 * where
 * - ::cudaTextureDesc::addressMode specifies the addressing mode for each dimension of the texture data. ::cudaTextureAddressMode is defined as:
 *   \code
        enum cudaTextureAddressMode {
            cudaAddressModeWrap   = 0,
            cudaAddressModeClamp  = 1,
            cudaAddressModeMirror = 2,
            cudaAddressModeBorder = 3
        };
 *   \endcode
 *   This is ignored if ::cudaResourceDesc::resType is ::cudaResourceTypeLinear. Also, if ::cudaTextureDesc::normalizedCoords
 *   is set to zero, ::cudaAddressModeWrap and ::cudaAddressModeMirror won't be supported and will be switched to ::cudaAddressModeClamp.
 *
 * - ::cudaTextureDesc::filterMode specifies the filtering mode to be used when fetching from the texture. ::cudaTextureFilterMode is defined as:
 *   \code
        enum cudaTextureFilterMode {
            cudaFilterModePoint  = 0,
            cudaFilterModeLinear = 1
        };
 *   \endcode
 *   This is ignored if ::cudaResourceDesc::resType is ::cudaResourceTypeLinear.
 *
 * - ::cudaTextureDesc::readMode specifies whether integer data should be converted to floating point or not. ::cudaTextureReadMode is defined as:
 *   \code
        enum cudaTextureReadMode {
            cudaReadModeElementType     = 0,
            cudaReadModeNormalizedFloat = 1
        };
 *   \endcode
 *   Note that this applies only to 8-bit and 16-bit integer formats. 32-bit integer format would not be promoted, regardless of 
 *   whether or not this ::cudaTextureDesc::readMode is set ::cudaReadModeNormalizedFloat is specified.
 *
 * - ::cudaTextureDesc::sRGB specifies whether sRGB to linear conversion should be performed during texture fetch.
 *
 * - ::cudaTextureDesc::borderColor specifies the float values of color. where:
 *   ::cudaTextureDesc::borderColor[0] contains value of 'R', 
 *   ::cudaTextureDesc::borderColor[1] contains value of 'G',
 *   ::cudaTextureDesc::borderColor[2] contains value of 'B', 
 *   ::cudaTextureDesc::borderColor[3] contains value of 'A'
 *   Note that application using integer border color values will need to <reinterpret_cast> these values to float.
 *   The values are set only when the addressing mode specified by ::cudaTextureDesc::addressMode is cudaAddressModeBorder.
 *
 * - ::cudaTextureDesc::normalizedCoords specifies whether the texture coordinates will be normalized or not.
 *
 * - ::cudaTextureDesc::maxAnisotropy specifies the maximum anistropy ratio to be used when doing anisotropic filtering. This value will be
 *   clamped to the range [1,16].
 *
 * - ::cudaTextureDesc::mipmapFilterMode specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
 *
 * - ::cudaTextureDesc::mipmapLevelBias specifies the offset to be applied to the calculated mipmap level.
 *
 * - ::cudaTextureDesc::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 *
 * - ::cudaTextureDesc::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 *
 * - ::cudaTextureDesc::disableTrilinearOptimization specifies whether the trilinear filtering optimizations will be disabled.
 *
 * - ::cudaTextureDesc::seamlessCubemap specifies whether seamless cube map filtering is enabled. This flag can only be specified if the 
 *   underlying resource is a CUDA array or a CUDA mipmapped array that was created with the flag ::cudaArrayCubemap.
 *   When seamless cube map filtering is enabled, texture address modes specified by ::cudaTextureDesc::addressMode are ignored.
 *   Instead, if the ::cudaTextureDesc::filterMode is set to ::cudaFilterModePoint the address mode ::cudaAddressModeClamp will be applied for all dimensions.
 *   If the ::cudaTextureDesc::filterMode is set to ::cudaFilterModeLinear seamless cube map filtering will be performed when sampling along the cube face borders.
 *
 * The ::cudaResourceViewDesc struct is defined as
 * \code
        struct cudaResourceViewDesc {
            enum cudaResourceViewFormat format;
            size_t                      width;
            size_t                      height;
            size_t                      depth;
            unsigned int                firstMipmapLevel;
            unsigned int                lastMipmapLevel;
            unsigned int                firstLayer;
            unsigned int                lastLayer;
        };
 * \endcode
 * where:
 * - ::cudaResourceViewDesc::format specifies how the data contained in the CUDA array or CUDA mipmapped array should
 *   be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block
 *   compressed format, then the underlying CUDA array or CUDA mipmapped array has to have a 32-bit unsigned integer format
 *   with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying CUDA array to have
 *   a 32-bit unsigned int with 2 channels. The other BC formats require the underlying resource to have the same 32-bit unsigned int
 *   format but with 4 channels.
 *
 * - ::cudaResourceViewDesc::width specifies the new width of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::cudaResourceViewDesc::height specifies the new height of the texture data. If the resource view format is a block
 *   compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
 *   this value has to be equal to that of the original resource.
 *
 * - ::cudaResourceViewDesc::depth specifies the new depth of the texture data. This value has to be equal to that of the
 *   original resource.
 *
 * - ::cudaResourceViewDesc::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
 *   For non-mipmapped resources, this value has to be zero.::cudaTextureDesc::minMipmapLevelClamp and ::cudaTextureDesc::maxMipmapLevelClamp
 *   will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
 *   then the actual minimum mipmap level clamp will be 3.2.
 *
 * - ::cudaResourceViewDesc::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
 *   has to be zero.
 *
 * - ::cudaResourceViewDesc::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
 *   For non-layered resources, this value has to be zero.
 *
 * - ::cudaResourceViewDesc::lastLayer specifies the last layer index for layered textures. For non-layered resources, 
 *   this value has to be zero.
 *
 *
 * \param pTexObject   - Texture object to create
 * \param pResDesc     - Resource descriptor
 * \param pTexDesc     - Texture descriptor
 * \param pResViewDesc - Resource view descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDestroyTextureObject,
 * ::cuTexObjectCreate
 */

extern __host__ cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc);

/**
 * \brief Destroys a texture object
 *
 * Destroys the texture object specified by \p texObject.
 *
 * \param texObject - Texture object to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::cudaCreateTextureObject,
 * ::cuTexObjectDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject);

/**
 * \brief Returns a texture object's resource descriptor
 *
 * Returns the resource descriptor for the texture object specified by \p texObject.
 *
 * \param pResDesc  - Resource descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaCreateTextureObject,
 * ::cuTexObjectGetResourceDesc
 */
extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject);

/**
 * \brief Returns a texture object's texture descriptor
 *
 * Returns the texture descriptor for the texture object specified by \p texObject.
 *
 * \param pTexDesc  - Texture descriptor
 * \param texObject - Texture object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaCreateTextureObject,
 * ::cuTexObjectGetTextureDesc
 */
extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject);

/**
 * \brief Returns a texture object's resource view descriptor
 *
 * Returns the resource view descriptor for the texture object specified by \p texObject.
 * If no resource view was specified, ::cudaErrorInvalidValue is returned.
 *
 * \param pResViewDesc - Resource view descriptor
 * \param texObject    - Texture object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaCreateTextureObject,
 * ::cuTexObjectGetResourceViewDesc
 */
extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject);

/** @} */ /* END CUDART_TEXTURE_OBJECT */

/**
 * \defgroup CUDART_SURFACE_OBJECT Surface Object Management
 *
 * ___MANBRIEF___ surface object management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the low level texture object management functions
 * of the CUDA runtime application programming interface. The surface object 
 * API is only supported on devices of compute capability 3.0 or higher.
 *
 * @{
 */

/**
 * \brief Creates a surface object
 *
 * Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
 * the data to perform surface load/stores on. ::cudaResourceDesc::resType must be 
 * ::cudaResourceTypeArray and  ::cudaResourceDesc::res::array::array
 * must be set to a valid CUDA array handle.
 *
 * Surface objects are only supported on devices of compute capability 3.0 or higher.
 * Additionally, a surface object is an opaque value, and, as such, should only be
 * accessed through CUDA API calls.
 *
 * \param pSurfObject - Surface object to create
 * \param pResDesc    - Resource descriptor
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidChannelDescriptor,
 * ::cudaErrorInvalidResourceHandle
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDestroySurfaceObject,
 * ::cuSurfObjectCreate
 */

extern __host__ cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc);

/**
 * \brief Destroys a surface object
 *
 * Destroys the surface object specified by \p surfObject.
 *
 * \param surfObject - Surface object to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::cudaCreateSurfaceObject,
 * ::cuSurfObjectDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);

/**
 * \brief Returns a surface object's resource descriptor
 * Returns the resource descriptor for the surface object specified by \p surfObject.
 *
 * \param pResDesc   - Resource descriptor
 * \param surfObject - Surface object
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaCreateSurfaceObject,
 * ::cuSurfObjectGetResourceDesc
 */
extern __host__ cudaError_t CUDARTAPI cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject);

/** @} */ /* END CUDART_SURFACE_OBJECT */

/**
 * \defgroup CUDART__VERSION Version Management
 *
 * @{
 */

/**
 * \brief Returns the latest version of CUDA supported by the driver
 *
 * Returns in \p *driverVersion the latest version of CUDA supported by
 * the driver. The version is returned as (1000 &times; major + 10 &times; minor).
 * For example, CUDA 9.2 would be represented by 9020. If no driver is installed,
 * then 0 is returned as the driver version.
 *
 * This function automatically returns ::cudaErrorInvalidValue
 * if \p driverVersion is NULL.
 *
 * \param driverVersion - Returns the CUDA driver version.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaRuntimeGetVersion,
 * ::cuDriverGetVersion
 */
extern __host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion);

/**
 * \brief Returns the CUDA Runtime version
 *
 * Returns in \p *runtimeVersion the version number of the current CUDA
 * Runtime instance. The version is returned as
 * (1000 &times; major + 10 &times; minor). For example,
 * CUDA 9.2 would be represented by 9020.
 *
 * This function automatically returns ::cudaErrorInvalidValue if
 * the \p runtimeVersion argument is NULL.
 *
 * \param runtimeVersion - Returns the CUDA Runtime version.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDriverGetVersion,
 * ::cuDriverGetVersion
 */
extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion);

/** @} */ /* END CUDART__VERSION */

/**
 * \defgroup CUDART_GRAPH Graph Management
 *
 * ___MANBRIEF___ graph management functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the graph management functions of CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Creates a graph
 *
 * Creates an empty graph, which is returned via \p pGraph.
 *
 * \param pGraph - Returns newly created graph
 * \param flags   - Graph creation flags, must be 0
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode,
 * ::cudaGraphInstantiate,
 * ::cudaGraphDestroy,
 * ::cudaGraphGetNodes,
 * ::cudaGraphGetRootNodes,
 * ::cudaGraphGetEdges,
 * ::cudaGraphClone
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags);

/**
 * \brief Creates a kernel execution node and adds it to a graph
 *
 * Creates a new kernel execution node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and arguments specified in \p pNodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * The cudaKernelNodeParams structure is defined as:
 *
 * \code
 *  struct cudaKernelNodeParams
 *  {
 *      void* func;
 *      dim3 gridDim;
 *      dim3 blockDim;
 *      unsigned int sharedMemBytes;
 *      void **kernelParams;
 *      void **extra;
 *  };
 * \endcode
 *
 * When the graph is launched, the node will invoke kernel \p func on a (\p gridDim.x x
 * \p gridDim.y x \p gridDim.z) grid of blocks. Each block contains
 * (\p blockDim.x x \p blockDim.y x \p blockDim.z) threads.
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * Kernel parameters to \p func can be specified in one of two ways:
 *
 * 1) Kernel parameters can be specified via \p kernelParams. If the kernel has N
 * parameters, then \p kernelParams needs to be an array of N pointers. Each pointer,
 * from \p kernelParams[0] to \p kernelParams[N-1], points to the region of memory from which the actual
 * parameter will be copied. The number of kernel parameters and their offsets and sizes do not need
 * to be specified as that information is retrieved directly from the kernel's image.
 *
 * 2) Kernel parameters can also be packaged by the application into a single buffer that is passed in
 * via \p extra. This places the burden on the application of knowing each kernel
 * parameter's size and alignment/padding within the buffer. The \p extra parameter exists
 * to allow this function to take additional less commonly used arguments. \p extra specifies
 * a list of names of extra settings and their corresponding values. Each extra setting name is
 * immediately followed by the corresponding value. The list must be terminated with either NULL or
 * CU_LAUNCH_PARAM_END.
 *
 * - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
 *   array;
 * - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
 *   value in \p extra will be a pointer to a buffer
 *   containing all the kernel parameters for launching kernel
 *   \p func;
 * - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
 *   value in \p extra will be a pointer to a size_t
 *   containing the size of the buffer specified with
 *   ::CU_LAUNCH_PARAM_BUFFER_POINTER;
 *
 * The error ::cudaErrorInvalidValue will be returned if kernel parameters are specified with both
 * \p kernelParams and \p extra (i.e. both \p kernelParams and
 * \p extra are non-NULL).
 *
 * The \p kernelParams or \p extra array, as well as the argument values it points to,
 * are copied during this call.
 *
 * \note Kernels launched using graphs must not use texture and surface references. Reading or
 *       writing through any texture or surface reference is undefined behavior.
 *       This restriction does not apply to texture and surface objects.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pNodeParams      - Parameters for the GPU execution node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDeviceFunction
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchKernel,
 * ::cudaGraphKernelNodeGetParams,
 * ::cudaGraphKernelNodeSetParams,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams);

/**
 * \brief Returns a kernel node's parameters
 *
 * Returns the parameters of kernel node \p node in \p pNodeParams.
 * The \p kernelParams or \p extra array returned in \p pNodeParams,
 * as well as the argument values it points to, are owned by the node.
 * This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cudaGraphKernelNodeSetParams to update the
 * parameters of this node.
 *
 * The params will contain either \p kernelParams or \p extra,
 * according to which of these was most recently set on the node.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDeviceFunction
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchKernel,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphKernelNodeSetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams);

/**
 * \brief Sets a kernel node's parameters
 *
 * Sets the parameters of kernel node \p node to \p pNodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorMemoryAllocation
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchKernel,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphKernelNodeGetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);

/**
 * \brief Copies attributes from source node to destination node.
 *
 * Copies attributes from source node \p src to destination node \p dst.
 * Both node must have the same context.
 *
 * \param[out] dst Destination node
 * \param[in] src Source node
 * For list of attributes see ::cudaKernelNodeAttrID
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidContext
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeCopyAttributes(
        cudaGraphNode_t hSrc,
        cudaGraphNode_t hDst);

/**
 * \brief Queries node attribute.
 *
 * Queries attribute \p attr from node \p hNode and stores it in corresponding
 * member of \p value_out.
 *
 * \param[in] hNode
 * \param[in] attr
 * \param[out] value_out
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetAttribute(
    cudaGraphNode_t hNode,
    enum cudaKernelNodeAttrID attr,
    union cudaKernelNodeAttrValue *value_out);

/**
 * \brief Sets node attribute.
 *
 * Sets attribute \p attr on node \p hNode from corresponding attribute of
 * \p value.
 *
 * \param[out] hNode
 * \param[in] attr
 * \param[out] value
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 *
 * \sa
 * ::cudaAccessPolicyWindow
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetAttribute(
    cudaGraphNode_t hNode,
    enum cudaKernelNodeAttrID attr,
    const union cudaKernelNodeAttrValue *value);

/**
 * \brief Creates a memcpy node and adds it to a graph
 *
 * Creates a new memcpy node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will perform the memcpy described by \p pCopyParams.
 * See ::cudaMemcpy3D() for a description of the structure and its restrictions.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pCopyParams      - Parameters for the memory copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpy3D,
 * ::cudaGraphAddMemcpyNodeToSymbol,
 * ::cudaGraphAddMemcpyNodeFromSymbol,
 * ::cudaGraphAddMemcpyNode1D,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemsetNode
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);

/**
 * \brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph
 *
 * Creates a new memcpy node to copy to \p symbol and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p src to the memory area pointed to by \p offset bytes from the start
 * of symbol \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::cudaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyToSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeFromSymbol,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemsetNode
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNodeToSymbol(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph
 *
 * Creates a new memcpy node to copy from \p symbol and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p offset bytes from the start of symbol \p symbol to the memory area
 *  pointed to by \p dst. The memory areas may not overlap. \p symbol is a variable
 *  that resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyFromSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeToSymbol,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemsetNode
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNodeFromSymbol(
    cudaGraphNode_t* pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies,
    size_t numDependencies,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Creates a 1D memcpy node and adds it to a graph
 *
 * Creates a new 1D memcpy node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Launching a
 * memcpy node with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dst             - Destination memory address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpy,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParams1D,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemsetNode
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode1D(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Returns a memcpy node's parameters
 *
 * Returns the parameters of memcpy node \p node in \p pNodeParams.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpy3D,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeSetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams);

/**
 * \brief Sets a memcpy node's parameters
 *
 * Sets the parameters of memcpy node \p node to \p pNodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpy3D,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphMemcpyNodeSetParams1D,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);

/**
 * \brief Sets a memcpy node's parameters to copy to a symbol on the device
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p src to the memory area pointed to by \p offset bytes from the start
 * of symbol \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::cudaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param node            - Node to set the parameters for
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyToSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParamsToSymbol(
    cudaGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Sets a memcpy node's parameters to copy from a symbol on the device
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p offset bytes from the start of symbol \p symbol to the memory area
 *  pointed to by \p dst. The memory areas may not overlap. \p symbol is a variable
 *  that resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param node            - Node to set the parameters for
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyFromSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParamsFromSymbol(
    cudaGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Sets a memcpy node's parameters to perform a 1-dimensional copy
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory
 * area pointed to by \p src to the memory area pointed to by \p dst, where
 * \p kind specifies the direction of the copy, and must be one of
 * ::cudaMemcpyHostToHost, ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToHost,
 * ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault. Passing
 * ::cudaMemcpyDefault is recommended, in which case the type of transfer is
 * inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing. Launching a
 * memcpy node with dst and src pointers that do not match the direction of
 * the copy results in an undefined behavior.
 *
 * \param node            - Node to set the parameters for
 * \param dst             - Destination memory address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpy,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams1D(
    cudaGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Creates a memset node and adds it to a graph
 *
 * Creates a new memset node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * The element size must be 1, 2, or 4 bytes.
 * When the graph is launched, the node will perform the memset described by \p pMemsetParams.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pMemsetParams    - Parameters for the memory set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidDevice
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemset2D,
 * ::cudaGraphMemsetNodeGetParams,
 * ::cudaGraphMemsetNodeSetParams,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemcpyNode
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams);

/**
 * \brief Returns a memset node's parameters
 *
 * Returns the parameters of memset node \p node in \p pNodeParams.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemset2D,
 * ::cudaGraphAddMemsetNode,
 * ::cudaGraphMemsetNodeSetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams);

/**
 * \brief Sets a memset node's parameters
 *
 * Sets the parameters of memset node \p node to \p pNodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemset2D,
 * ::cudaGraphAddMemsetNode,
 * ::cudaGraphMemsetNodeGetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);

/**
 * \brief Creates a host execution node and adds it to a graph
 *
 * Creates a new CPU execution node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and arguments specified in \p pNodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will invoke the specified CPU function.
 * Host nodes are not supported under MPS with pre-Volta GPUs.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param pNodeParams      - Parameters for the host node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchHostFunc,
 * ::cudaGraphHostNodeGetParams,
 * ::cudaGraphHostNodeSetParams,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams);

/**
 * \brief Returns a host node's parameters
 *
 * Returns the parameters of host node \p node in \p pNodeParams.
 *
 * \param node        - Node to get the parameters for
 * \param pNodeParams - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchHostFunc,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphHostNodeSetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams);

/**
 * \brief Sets a host node's parameters
 *
 * Sets the parameters of host node \p node to \p nodeParams.
 *
 * \param node        - Node to set the parameters for
 * \param pNodeParams - Parameters to copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchHostFunc,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphHostNodeGetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);

/**
 * \brief Creates a child graph node and adds it to a graph
 *
 * Creates a new node which executes an embedded graph, and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * If \p hGraph contains allocation or free nodes, this call will return an error.
 *
 * The node executes an embedded child graph. The child graph is cloned in this call.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param childGraph      - The graph to clone into this node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphChildGraphNodeGetGraph,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode,
 * ::cudaGraphClone
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph);

/**
 * \brief Gets a handle to the embedded graph of a child graph node
 *
 * Gets a handle to the embedded graph in a child graph node. This call
 * does not clone the graph. Changes to the graph will be reflected in
 * the node, and the node retains ownership of the graph.
 *
 * Allocation and free nodes cannot be added to the returned graph.
 * Attempting to do so will return an error.
 *
 * \param node   - Node to get the embedded graph for
 * \param pGraph - Location to store a handle to the graph
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphNodeFindInClone
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph);

/**
 * \brief Creates an empty node and adds it to a graph
 *
 * Creates a new node which performs no operation, and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * An empty node performs no operation during execution, but can be used for
 * transitive ordering. For example, a phased execution graph with 2 groups of n
 * nodes with a barrier between them can be represented using an empty node and
 * 2*n dependency edges, rather than no empty node and n^2 dependency edges.
 *
 * \param pGraphNode     - Returns newly created node
 * \param graph          - Graph to which to add the node
 * \param pDependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies);

/**
 * \brief Creates an event record node and adds it to a graph
 *
 * Creates a new event record node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and event specified in \p event.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * Each launch of the graph will record \p event to capture execution of the
 * node's dependencies.
 *
 * These nodes may not be used in loops or conditionals.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param event           - Event for the node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventWaitNode,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode,
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
#endif

/**
 * \brief Returns the event associated with an event record node
 *
 * Returns the event of event record node \p hNode in \p event_out.
 *
 * \param hNode     - Node to get the event for
 * \param event_out - Pointer to return the event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphEventRecordNodeSetEvent,
 * ::cudaGraphEventWaitNodeGetEvent,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
#endif

/**
 * \brief Sets an event record node's event
 *
 * Sets the event of event record node \p hNode to \p event.
 *
 * \param hNode - Node to set the event for
 * \param event - Event to use
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphEventRecordNodeGetEvent,
 * ::cudaGraphEventWaitNodeSetEvent,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
#endif

/**
 * \brief Creates an event wait node and adds it to a graph
 *
 * Creates a new event wait node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies and event specified in \p event.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * The graph node will wait for all work captured in \p event.  See ::cuEventRecord()
 * for details on what is captured by an event.  The synchronization will be performed
 * efficiently on the device when applicable.  \p event may be from a different context
 * or device than the launch stream.
 *
 * These nodes may not be used in loops or conditionals.
 *
 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param event           - Event for the node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventRecordNode,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode,
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
#endif

/**
 * \brief Returns the event associated with an event wait node
 *
 * Returns the event of event wait node \p hNode in \p event_out.
 *
 * \param hNode     - Node to get the event for
 * \param event_out - Pointer to return the event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphEventWaitNodeSetEvent,
 * ::cudaGraphEventRecordNodeGetEvent,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
#endif

/**
 * \brief Sets an event wait node's event
 *
 * Sets the event of event wait node \p hNode to \p event.
 *
 * \param hNode - Node to set the event for
 * \param event - Event to use
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphEventWaitNodeGetEvent,
 * ::cudaGraphEventRecordNodeSetEvent,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
#endif

/**
 * \brief Creates an external semaphore signal node and adds it to a graph
 *
 * Creates a new external semaphore signal node and adds it to \p graph with \p
 * numDependencies dependencies specified via \p dependencies and arguments specified
 * in \p nodeParams. It is possible for \p numDependencies to be 0, in which case the
 * node will be placed at the root of the graph. \p dependencies may not have any
 * duplicate entries. A handle to the new node will be returned in \p pGraphNode.
 *
 * Performs a signal operation on a set of externally allocated semaphore objects
 * when the node is launched.  The operation(s) will occur after all of the node's
 * dependencies have completed.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphExternalSemaphoresSignalNodeGetParams,
 * ::cudaGraphExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaImportExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode,
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
#endif

/**
 * \brief Returns an external semaphore signal node's parameters
 *
 * Returns the parameters of an external semaphore signal node \p hNode in \p params_out.
 * The \p extSemArray and \p paramsArray returned in \p params_out,
 * are owned by the node.  This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cudaGraphExternalSemaphoresSignalNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchKernel,
 * ::cudaGraphAddExternalSemaphoresSignalNode,
 * ::cudaGraphExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out);
#endif

/**
 * \brief Sets an external semaphore signal node's parameters
 *
 * Sets the parameters of an external semaphore signal node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddExternalSemaphoresSignalNode,
 * ::cudaGraphExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
#endif

/**
 * \brief Creates an external semaphore wait node and adds it to a graph
 *
 * Creates a new external semaphore wait node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p dependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p pGraphNode.
 *
 * Performs a wait operation on a set of externally allocated semaphore objects
 * when the node is launched.  The node's dependencies will not be launched until
 * the wait operation has completed.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphExternalSemaphoresWaitNodeGetParams,
 * ::cudaGraphExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphAddExternalSemaphoresSignalNode,
 * ::cudaImportExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode,
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
#endif

/**
 * \brief Returns an external semaphore wait node's parameters
 *
 * Returns the parameters of an external semaphore wait node \p hNode in \p params_out.
 * The \p extSemArray and \p paramsArray returned in \p params_out,
 * are owned by the node.  This memory remains valid until the node is destroyed or its
 * parameters are modified, and should not be modified
 * directly. Use ::cudaGraphExternalSemaphoresSignalNodeSetParams to update the
 * parameters of this node.
 *
 * \param hNode      - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaLaunchKernel,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaGraphExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out);
#endif

/**
 * \brief Sets an external semaphore wait node's parameters
 *
 * Sets the parameters of an external semaphore wait node \p hNode to \p nodeParams.
 *
 * \param hNode      - Node to set the parameters for
 * \param nodeParams - Parameters to copy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaGraphExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
#endif

/**
 * \brief Creates an allocation node and adds it to a graph
 *
 * Creates a new allocation node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and arguments specified in \p nodeParams.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p pGraphNode.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param nodeParams      - Parameters for the node
 *
 * When ::cudaGraphAddMemAllocNode creates an allocation node, it returns the address of the allocation in
 * \p nodeParams.dptr.  The allocation's address remains fixed across instantiations and launches.
 *
 * If the allocation is freed in the same graph, by creating a free node using ::cudaGraphAddMemFreeNode,
 * the allocation can be accessed by nodes ordered after the allocation node but before the free node.
 * These allocations cannot be freed outside the owning graph, and they can only be freed once in the
 * owning graph.
 *
 * If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the
 * graph which are ordered after the allocation node, but also by stream operations ordered after the
 * graph's execution but before the allocation is freed.
 *
 * Allocations which are not freed in the same graph can be freed by:
 * - passing the allocation to ::cudaMemFreeAsync or ::cudaMemFree;
 * - launching a graph with a free node for that allocation; or
 * - specifying ::cudaGraphInstantiateFlagAutoFreeOnLaunch during instantiation, which makes
 *   each launch behave as though it called ::cudaMemFreeAsync for every unfreed allocation.
 *
 * It is not possible to free an allocation in both the owning graph and another graph.  If the allocation
 * is freed in the same graph, a free node cannot be added to another graph.  If the allocation is freed
 * in another graph, a free node can no longer be added to the owning graph.
 *
 * The following restrictions apply to graphs which contain allocation and/or memory free nodes:
 * - Nodes and edges of the graph cannot be deleted.
 * - The graph cannot be used in a child node.
 * - Only one instantiation of the graph may exist at any point in time.
 * - The graph cannot be cloned.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorCudartUnloading,
 * ::cudaErrorInitializationError,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorOutOfMemory
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cudaGraphAddMemFreeNode,
 * ::cudaGraphMemAllocNodeGetParams,
 * ::cudaDeviceGraphMemTrim,
 * ::cudaDeviceGetGraphMemAttribute,
 * ::cudaDeviceSetGraphMemAttribute,
 * ::cudaMallocAsync,
 * ::cudaFreeAsync,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphAddExternalSemaphoresSignalNode,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams *nodeParams);
#endif

/**
 * \brief Returns a memory alloc node's parameters
 *
 * Returns the parameters of a memory alloc node \p hNode in \p params_out.
 * The \p poolProps and \p accessDescs returned in \p params_out, are owned by the
 * node.  This memory remains valid until the node is destroyed.  The returned
 * parameters must not be modified.
 *
 * \param node       - Node to get the parameters for
 * \param params_out - Pointer to return the parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemAllocNode,
 * ::cudaGraphMemFreeNodeGetParams
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams *params_out);
#endif

/**
 * \brief Creates a memory free node and adds it to a graph
 *
 * Creates a new memory free node and adds it to \p graph with \p numDependencies
 * dependencies specified via \p pDependencies and address specified in \p dptr.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries. A handle
 * to the new node will be returned in \p pGraphNode.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dptr            - Address of memory to free
 *
 * ::cudaGraphAddMemFreeNode will return ::cudaErrorInvalidValue if the user attempts to free:
 * - an allocation twice in the same graph.
 * - an address that was not returned by an allocation node.
 * - an invalid address.
 *
 * The following restrictions apply to graphs which contain allocation and/or memory free nodes:
 * - Nodes and edges of the graph cannot be deleted.
 * - The graph cannot be used in a child node.
 * - Only one instantiation of the graph may exist at any point in time.
 * - The graph cannot be cloned.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorCudartUnloading,
 * ::cudaErrorInitializationError,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorOutOfMemory
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 * ::cudaGraphAddMemAllocNode,
 * ::cudaGraphMemFreeNodeGetParams,
 * ::cudaDeviceGraphMemTrim,
 * ::cudaDeviceGetGraphMemAttribute,
 * ::cudaDeviceSetGraphMemAttribute,
 * ::cudaMallocAsync,
 * ::cudaFreeAsync,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphAddExternalSemaphoresSignalNode,
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr);
#endif

/**
 * \brief Returns a memory free node's parameters
 *
 * Returns the address of a memory free node \p hNode in \p dptr_out.
 *
 * \param node     - Node to get the parameters for
 * \param dptr_out - Pointer to return the device address
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemFreeNode,
 * ::cudaGraphMemFreeNodeGetParams
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out);
#endif

/**
 * \brief Free unused memory that was cached on the specified device for use with graphs back to the OS.
 *
 * Blocks which are not in use by a graph that is either currently executing or scheduled to execute are
 * freed back to the operating system.
 *
 * \param device - The device for which cached memory should be freed.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemAllocNode,
 * ::cudaGraphAddMemFreeNode,
 * ::cudaDeviceGetGraphMemAttribute,
 * ::cudaDeviceSetGraphMemAttribute,
 * ::cudaMallocAsync,
 * ::cudaFreeAsync,
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaDeviceGraphMemTrim(int device);
#endif

/**
 * \brief Query asynchronous allocation attributes related to graphs
 *
 * Valid attributes are:
 *
 * - ::cudaGraphMemAttrUsedMemCurrent: Amount of memory, in bytes, currently associated with graphs
 * - ::cudaGraphMemAttrUsedMemHigh: High watermark of memory, in bytes, associated with graphs since the
 *   last time it was reset.  High watermark can only be reset to zero.
 * - ::cudaGraphMemAttrReservedMemCurrent: Amount of memory, in bytes, currently allocated for use by
 *   the CUDA graphs asynchronous allocator.
 * - ::cudaGraphMemAttrReservedMemHigh: High watermark of memory, in bytes, currently allocated for use by
 *   the CUDA graphs asynchronous allocator.
 *
 * \param device - Specifies the scope of the query
 * \param attr - attribute to get
 * \param value - retrieved value
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceSetGraphMemAttribute,
 * ::cudaGraphAddMemAllocNode,
 * ::cudaGraphAddMemFreeNode,
 * ::cudaDeviceGraphMemTrim,
 * ::cudaMallocAsync,
 * ::cudaFreeAsync,
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value);
#endif

/**
 * \brief Set asynchronous allocation attributes related to graphs
 *
 * Valid attributes are:
 *
 * - ::cudaGraphMemAttrUsedMemHigh: High watermark of memory, in bytes, associated with graphs since the
 *   last time it was reset.  High watermark can only be reset to zero.
 * - ::cudaGraphMemAttrReservedMemHigh: High watermark of memory, in bytes, currently allocated for use by
 *   the CUDA graphs asynchronous allocator.
 *
 * \param device - Specifies the scope of the query
 * \param attr - attribute to get
 * \param value - pointer to value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaDeviceGetGraphMemAttribute,
 * ::cudaGraphAddMemAllocNode,
 * ::cudaGraphAddMemFreeNode,
 * ::cudaDeviceGraphMemTrim,
 * ::cudaMallocAsync,
 * ::cudaFreeAsync,
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value);
#endif

/**
 * \brief Clones a graph
 *
 * This function creates a copy of \p originalGraph and returns it in \p pGraphClone.
 * All parameters are copied into the cloned graph. The original graph may be modified 
 * after this call without affecting the clone.
 *
 * Child graph nodes in the original graph are recursively copied into the clone.
 *
 * \param pGraphClone  - Returns newly created cloned graph
 * \param originalGraph - Graph to clone
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorMemoryAllocation
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphCreate,
 * ::cudaGraphNodeFindInClone
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);

/**
 * \brief Finds a cloned version of a node
 *
 * This function returns the node in \p clonedGraph corresponding to \p originalNode 
 * in the original graph.
 *
 * \p clonedGraph must have been cloned from \p originalGraph via ::cudaGraphClone. 
 * \p originalNode must have been in \p originalGraph at the time of the call to 
 * ::cudaGraphClone, and the corresponding cloned node in \p clonedGraph must not have 
 * been removed. The cloned node is then returned via \p pClonedNode.
 *
 * \param pNode  - Returns handle to the cloned node
 * \param originalNode - Handle to the original node
 * \param clonedGraph - Cloned graph to query
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphClone
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);

/**
 * \brief Returns a node's type
 *
 * Returns the node type of \p node in \p pType.
 *
 * \param node - Node to query
 * \param pType  - Pointer to return the node type
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphGetNodes,
 * ::cudaGraphGetRootNodes,
 * ::cudaGraphChildGraphNodeGetGraph,
 * ::cudaGraphKernelNodeGetParams,
 * ::cudaGraphKernelNodeSetParams,
 * ::cudaGraphHostNodeGetParams,
 * ::cudaGraphHostNodeSetParams,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemsetNodeGetParams,
 * ::cudaGraphMemsetNodeSetParams
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType);

/**
 * \brief Returns a graph's nodes
 *
 * Returns a list of \p graph's nodes. \p nodes may be NULL, in which case this
 * function will return the number of nodes in \p numNodes. Otherwise,
 * \p numNodes entries will be filled in. If \p numNodes is higher than the actual
 * number of nodes, the remaining entries in \p nodes will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p numNodes.
 *
 * \param graph    - Graph to query
 * \param nodes    - Pointer to return the nodes
 * \param numNodes - See description
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphCreate,
 * ::cudaGraphGetRootNodes,
 * ::cudaGraphGetEdges,
 * ::cudaGraphNodeGetType,
 * ::cudaGraphNodeGetDependencies,
 * ::cudaGraphNodeGetDependentNodes
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);

/**
 * \brief Returns a graph's root nodes
 *
 * Returns a list of \p graph's root nodes. \p pRootNodes may be NULL, in which case this
 * function will return the number of root nodes in \p pNumRootNodes. Otherwise,
 * \p pNumRootNodes entries will be filled in. If \p pNumRootNodes is higher than the actual
 * number of root nodes, the remaining entries in \p pRootNodes will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p pNumRootNodes.
 *
 * \param graph       - Graph to query
 * \param pRootNodes    - Pointer to return the root nodes
 * \param pNumRootNodes - See description
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphCreate,
 * ::cudaGraphGetNodes,
 * ::cudaGraphGetEdges,
 * ::cudaGraphNodeGetType,
 * ::cudaGraphNodeGetDependencies,
 * ::cudaGraphNodeGetDependentNodes
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes);

/**
 * \brief Returns a graph's dependency edges
 *
 * Returns a list of \p graph's dependency edges. Edges are returned via corresponding
 * indices in \p from and \p to; that is, the node in \p to[i] has a dependency on the
 * node in \p from[i]. \p from and \p to may both be NULL, in which
 * case this function only returns the number of edges in \p numEdges. Otherwise,
 * \p numEdges entries will be filled in. If \p numEdges is higher than the actual
 * number of edges, the remaining entries in \p from and \p to will be set to NULL, and
 * the number of edges actually returned will be written to \p numEdges.
 *
 * \param graph    - Graph to get the edges from
 * \param from     - Location to return edge endpoints
 * \param to       - Location to return edge endpoints
 * \param numEdges - See description
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphGetNodes,
 * ::cudaGraphGetRootNodes,
 * ::cudaGraphAddDependencies,
 * ::cudaGraphRemoveDependencies,
 * ::cudaGraphNodeGetDependencies,
 * ::cudaGraphNodeGetDependentNodes
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges);

/**
 * \brief Returns a node's dependencies
 *
 * Returns a list of \p node's dependencies. \p pDependencies may be NULL, in which case this
 * function will return the number of dependencies in \p pNumDependencies. Otherwise,
 * \p pNumDependencies entries will be filled in. If \p pNumDependencies is higher than the actual
 * number of dependencies, the remaining entries in \p pDependencies will be set to NULL, and the
 * number of nodes actually obtained will be returned in \p pNumDependencies.
 *
 * \param node           - Node to query
 * \param pDependencies    - Pointer to return the dependencies
 * \param pNumDependencies - See description
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphNodeGetDependentNodes,
 * ::cudaGraphGetNodes,
 * ::cudaGraphGetRootNodes,
 * ::cudaGraphGetEdges,
 * ::cudaGraphAddDependencies,
 * ::cudaGraphRemoveDependencies
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies);

/**
 * \brief Returns a node's dependent nodes
 *
 * Returns a list of \p node's dependent nodes. \p pDependentNodes may be NULL, in which
 * case this function will return the number of dependent nodes in \p pNumDependentNodes.
 * Otherwise, \p pNumDependentNodes entries will be filled in. If \p pNumDependentNodes is
 * higher than the actual number of dependent nodes, the remaining entries in
 * \p pDependentNodes will be set to NULL, and the number of nodes actually obtained will
 * be returned in \p pNumDependentNodes.
 *
 * \param node             - Node to query
 * \param pDependentNodes    - Pointer to return the dependent nodes
 * \param pNumDependentNodes - See description
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphNodeGetDependencies,
 * ::cudaGraphGetNodes,
 * ::cudaGraphGetRootNodes,
 * ::cudaGraphGetEdges,
 * ::cudaGraphAddDependencies,
 * ::cudaGraphRemoveDependencies
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes);

/**
 * \brief Adds dependency edges to a graph.
 *
 * The number of dependencies to be added is defined by \p numDependencies
 * Elements in \p pFrom and \p pTo at corresponding indices define a dependency.
 * Each node in \p pFrom and \p pTo must belong to \p graph.
 *
 * If \p numDependencies is 0, elements in \p pFrom and \p pTo will be ignored.
 * Specifying an existing dependency will return an error.
 *
 * \param graph - Graph to which dependencies are added
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param numDependencies - Number of dependencies to be added
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphRemoveDependencies,
 * ::cudaGraphGetEdges,
 * ::cudaGraphNodeGetDependencies,
 * ::cudaGraphNodeGetDependentNodes
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);

/**
 * \brief Removes dependency edges from a graph.
 *
 * The number of \p pDependencies to be removed is defined by \p numDependencies.
 * Elements in \p pFrom and \p pTo at corresponding indices define a dependency.
 * Each node in \p pFrom and \p pTo must belong to \p graph.
 *
 * If \p numDependencies is 0, elements in \p pFrom and \p pTo will be ignored.
 * Specifying a non-existing dependency will return an error.
 *
 * \param graph - Graph from which to remove dependencies
 * \param from - Array of nodes that provide the dependencies
 * \param to - Array of dependent nodes
 * \param numDependencies - Number of dependencies to be removed
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddDependencies,
 * ::cudaGraphGetEdges,
 * ::cudaGraphNodeGetDependencies,
 * ::cudaGraphNodeGetDependentNodes
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);

/**
 * \brief Remove a node from the graph
 *
 * Removes \p node from its graph. This operation also severs any dependencies of other nodes 
 * on \p node and vice versa.
 *
 * Dependencies cannot be removed from graphs which contain allocation or free nodes.
 * Any attempt to do so will return an error.
 *
 * \param node  - Node to remove
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemsetNode
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphDestroyNode(cudaGraphNode_t node);

/**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p graph as an executable graph. The graph is validated for any
 * structural constraints or intra-node constraints which were not previously
 * validated. If instantiation is successful, a handle to the instantiated graph
 * is returned in \p pGraphExec.
 *
 * If there are any errors, diagnostic information may be returned in \p pErrorNode and
 * \p pLogBuffer. This is the primary way to inspect instantiation errors. The output
 * will be null terminated unless the diagnostics overflow
 * the buffer. In this case, they will be truncated, and the last byte can be
 * inspected to determine if truncation occurred.
 *
 * \param pGraphExec - Returns instantiated graph
 * \param graph      - Graph to instantiate
 * \param pErrorNode - In case of an instantiation error, this may be modified to
 *                      indicate a node contributing to the error
 * \param pLogBuffer   - A character buffer to store diagnostic messages
 * \param bufferSize  - Size of the log buffer in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphInstantiateWithFlags,
 * ::cudaGraphCreate,
 * ::cudaGraphUpload,
 * ::cudaGraphLaunch,
 * ::cudaGraphExecDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize);

/**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p graph as an executable graph. The graph is validated for any
 * structural constraints or intra-node constraints which were not previously
 * validated. If instantiation is successful, a handle to the instantiated graph
 * is returned in \p pGraphExec.
 *
 * The \p flags parameter controls the behavior of instantiation and subsequent
 * graph launches.  Valid flags are:
 *
 * - ::cudaGraphInstantiateFlagAutoFreeOnLaunch, which configures a
 * graph containing memory allocation nodes to automatically free any
 * unfreed memory allocations before the graph is relaunched.
 *
 * If \p graph contains any allocation or free nodes, there can be at most one
 * executable graph in existence for that graph at a time.
 *
 * An attempt to instantiate a second executable graph before destroying the first
 * with ::cudaGraphExecDestroy will result in an error.
 *
 * \param pGraphExec - Returns instantiated graph
 * \param graph      - Graph to instantiate
 * \param flags      - Flags to control instantiation.  See ::CUgraphInstantiate_flags.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphInstantiate,
 * ::cudaGraphCreate,
 * ::cudaGraphUpload,
 * ::cudaGraphLaunch,
 * ::cudaGraphExecDestroy
 */
#if __CUDART_API_VERSION >= 11040
extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags);
#endif

/**
 * \brief Sets the parameters for a kernel node in the given graphExec
 *
 * Sets the parameters of a kernel node in an executable graph \p hGraphExec. 
 * The node is identified by the corresponding node \p node in the 
 * non-executable graph, from which the executable graph was instantiated. 
 *
 * \p node must not have been removed from the original graph. The \p func field 
 * of \p nodeParams cannot be modified and must match the original value.
 * All other values can be modified. 
 *
 * The modifications only affect future launches of \p hGraphExec. Already 
 * enqueued or running launches of \p hGraphExec are not affected by this call. 
 * \p node is also not modified by this call.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - kernel node from the graph from which graphExec was instantiated
 * \param pNodeParams - Updated Parameters to set
 * 
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained \p pNodeParams at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * The source and destination memory in \p pNodeParams must be allocated from the same 
 * contexts as the original source and destination memory.  Both the instantiation-time 
 * memory operands and the memory operands in \p pNodeParams must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::cudaErrorInvalidValue if the memory operands' mappings changed or
 * either the original or new memory operands are multidimensional.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - Memcpy node from the graph which was used to instantiate graphExec
 * \param pNodeParams - Updated Parameters to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphExecMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphExecMemcpyNodeSetParams1D,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the device
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p src and \p symbol must be allocated from the same contexts as the original source and
 * destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::cudaErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeToSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParamsToSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    const void* symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the device
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p symbol and \p dst must be allocated from the same contexts as the original source and
 * destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::cudaErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeFromSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const void* symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional copy
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p src and \p dst must be allocated from the same contexts as the original source
 * and destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::cudaErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param dst             - Destination memory address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNode1D,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParams1D,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams1D(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const void* src,
    size_t count,
    enum cudaMemcpyKind kind);
#endif

/**
 * \brief Sets the parameters for a memset node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained \p pNodeParams at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * The destination memory in \p pNodeParams must be allocated from the same 
 * context as the original destination memory.  Both the instantiation-time 
 * memory operand and the memory operand in \p pNodeParams must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns cudaErrorInvalidValue if the memory operand's mappings changed or
 * either the original or new memory operand are multidimensional.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - Memset node from the graph which was used to instantiate graphExec
 * \param pNodeParams - Updated Parameters to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemsetNode,
 * ::cudaGraphMemsetNodeSetParams,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);

/**
 * \brief Sets the parameters for a host node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained \p pNodeParams at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * \param hGraphExec  - The executable graph in which to set the specified node
 * \param node        - Host node from the graph which was used to instantiate graphExec
 * \param pNodeParams - Updated Parameters to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddHostNode,
 * ::cudaGraphHostNodeSetParams,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);

/**
 * \brief Updates node parameters in the child graph node in the given graphExec.
 *
 * Updates the work represented by \p node in \p hGraphExec as though the nodes contained
 * in \p node's graph had the parameters contained in \p childGraph's nodes at instantiation.
 * \p node must remain in the graph which was used to instantiate \p hGraphExec.
 * Changed edges to and from \p node are ignored.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also
 * not modified by this call.
 *
 * The topology of \p childGraph, as well as the node insertion order,  must match that
 * of the graph contained in \p node.  See ::cudaGraphExecUpdate() for a list of restrictions
 * on what can be updated in an instantiated graph.  The update is recursive, so child graph
 * nodes contained within the top level child graph will also be updated.

 * \param hGraphExec - The executable graph in which to set the specified node
 * \param node       - Host node from the graph which was used to instantiate graphExec
 * \param childGraph - The graph supplying the updated parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphChildGraphNodeGetGraph,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph);
#endif

/**
 * \brief Sets the event for an event record node in the given graphExec
 *
 * Sets the event of an event record node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Event record node from the graph from which graphExec was instantiated
 * \param event      - Updated event to use
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventRecordNode,
 * ::cudaGraphEventRecordNodeGetEvent,
 * ::cudaGraphEventWaitNodeSetEvent,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
#endif

/**
 * \brief Sets the event for an event wait node in the given graphExec
 *
 * Sets the event of an event wait node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Event wait node from the graph from which graphExec was instantiated
 * \param event      - Updated event to use
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddEventWaitNode,
 * ::cudaGraphEventWaitNodeGetEvent,
 * ::cudaGraphEventRecordNodeSetEvent,
 * ::cudaEventRecordWithFlags,
 * ::cudaStreamWaitEvent,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
#endif

/**
 * \brief Sets the parameters for an external semaphore signal node in the given graphExec
 *
 * Sets the parameters of an external semaphore signal node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * Changing \p nodeParams->numExtSems is not supported.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - semaphore signal node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddExternalSemaphoresSignalNode,
 * ::cudaImportExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresWaitNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
#endif

/**
 * \brief Sets the parameters for an external semaphore wait node in the given graphExec
 *
 * Sets the parameters of an external semaphore wait node in an executable graph \p hGraphExec.
 * The node is identified by the corresponding node \p hNode in the
 * non-executable graph, from which the executable graph was instantiated.
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * Changing \p nodeParams->numExtSems is not supported.
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - semaphore wait node from the graph from which graphExec was instantiated
 * \param nodeParams - Updated Parameters to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddExternalSemaphoresWaitNode,
 * ::cudaImportExternalSemaphore,
 * ::cudaSignalExternalSemaphoresAsync,
 * ::cudaWaitExternalSemaphoresAsync,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams,
 * ::cudaGraphExecChildGraphNodeSetParams,
 * ::cudaGraphExecEventRecordNodeSetEvent,
 * ::cudaGraphExecEventWaitNodeSetEvent,
 * ::cudaGraphExecExternalSemaphoresSignalNodeSetParams,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 */
#if __CUDART_API_VERSION >= 11020
extern __host__ cudaError_t CUDARTAPI cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
#endif

/**
 * \brief Enables or disables the specified node in the given graphExec
 *
 * Sets \p hNode to be either enabled or disabled. Disabled nodes are functionally equivalent 
 * to empty nodes until they are reenabled. Existing node parameters are not affected by 
 * disabling/enabling the node.
 *  
 * The node is identified by the corresponding node \p hNode in the non-executable 
 * graph, from which the executable graph was instantiated.   
 *
 * \p hNode must not have been removed from the original graph.
 *
 * The modifications only affect future launches of \p hGraphExec. Already
 * enqueued or running launches of \p hGraphExec are not affected by this call.
 * \p hNode is also not modified by this call.
 *
 * \note Currently only kernel nodes are supported. 
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Node from the graph from which graphExec was instantiated
 * \param isEnabled  - Node is enabled if != 0, otherwise the node is disabled
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphNodeGetEnabled,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 * ::cudaGraphLaunch
 */
#if __CUDART_API_VERSION >= 11060
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled);
#endif

/**
 * \brief Query whether a node in the given graphExec is enabled
 *
 * Sets isEnabled to 1 if \p hNode is enabled, or 0 if \p hNode is disabled.
 *
 * The node is identified by the corresponding node \p hNode in the non-executable 
 * graph, from which the executable graph was instantiated.   
 *
 * \p hNode must not have been removed from the original graph.
 *
 * \note Currently only kernel nodes are supported. 
 *
 * \param hGraphExec - The executable graph in which to set the specified node
 * \param hNode      - Node from the graph from which graphExec was instantiated
 * \param isEnabled  - Location to return the enabled status of the node
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphNodeSetEnabled,
 * ::cudaGraphExecUpdate,
 * ::cudaGraphInstantiate
 * ::cudaGraphLaunch
 */
#if __CUDART_API_VERSION >= 11060
extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int *isEnabled);
#endif

/**
 * \brief Check whether an executable graph can be updated with a graph and perform the update if possible
 *
 * Updates the node parameters in the instantiated graph specified by \p hGraphExec with the
 * node parameters in a topologically identical graph specified by \p hGraph.
 *
 * Limitations:
 *
 * - Kernel nodes:
 *   - The owning context of the function cannot change.
 *   - A node whose function originally did not use CUDA dynamic parallelism cannot be updated
 *     to a function which uses CDP.
 *   - A cooperative node cannot be updated to a non-cooperative node, and vice-versa.
 * - Memset and memcpy nodes:
 *   - The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.
 *   - The source/destination memory must be allocated from the same contexts as the original
 *     source/destination memory.
 *   - Only 1D memsets can be changed.
 * - Additional memcpy node restrictions:
 *   - Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE,
 *     CU_MEMORYTYPE_ARRAY, etc.) is not supported.
 *
 * Note:  The API may add further restrictions in future releases.  The return code should always be checked.
 *
 * cudaGraphExecUpdate sets \p updateResult_out to cudaGraphExecUpdateErrorTopologyChanged under
 * the following conditions:
 *
 * - The count of nodes directly in \p hGraphExec and \p hGraph differ, in which case \p hErrorNode_out
 *   is NULL.
 * - A node is deleted in \p hGraph but not not its pair from \p hGraphExec, in which case \p hErrorNode_out
 *   is NULL.
 * - A node is deleted in \p hGraphExec but not its pair from \p hGraph, in which case \p hErrorNode_out is
 *   the pairless node from \p hGraph.
 * - The dependent nodes of a pair differ, in which case \p hErrorNode_out is the node from \p hGraph.
 *
 * cudaGraphExecUpdate sets \p updateResult_out to:
 * - cudaGraphExecUpdateError if passed an invalid value.
 * - cudaGraphExecUpdateErrorTopologyChanged if the graph topology changed
 * - cudaGraphExecUpdateErrorNodeTypeChanged if the type of a node changed, in which case
 *   \p hErrorNode_out is set to the node from \p hGraph.
 * - cudaGraphExecUpdateErrorFunctionChanged if the function of a kernel node changed (CUDA driver < 11.2)
 * - cudaGraphExecUpdateErrorUnsupportedFunctionChange if the func field of a kernel changed in an
 *   unsupported way(see note above), in which case \p hErrorNode_out is set to the node from \p hGraph
 * - cudaGraphExecUpdateErrorParametersChanged if any parameters to a node changed in a way 
 *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph
 * - cudaGraphExecUpdateErrorAttributesChanged if any attributes of a node changed in a way 
 *   that is not supported, in which case \p hErrorNode_out is set to the node from \p hGraph
 * - cudaGraphExecUpdateErrorNotSupported if something about a node is unsupported, like 
 *   the node's type or configuration, in which case \p hErrorNode_out is set to the node from \p hGraph
 *
 * If \p updateResult_out isn't set in one of the situations described above, the update check passes
 * and cudaGraphExecUpdate updates \p hGraphExec to match the contents of \p hGraph.  If an error happens
 * during the update, \p updateResult_out will be set to cudaGraphExecUpdateError; otherwise,
 * \p updateResult_out is set to cudaGraphExecUpdateSuccess.
 *
 * cudaGraphExecUpdate returns cudaSuccess when the updated was performed successfully.  It returns
 * cudaErrorGraphExecUpdateFailure if the graph update was not performed because it included 
 * changes which violated constraints specific to instantiated graph update.
 *
 * \param hGraphExec The instantiated graph to be updated
 * \param hGraph The graph containing the updated parameters
 * \param hErrorNode_out The node which caused the permissibility check to forbid the update, if any
 * \param updateResult_out Whether the graph update was permitted.  If was forbidden, the reason why
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorGraphExecUpdateFailure,
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphInstantiate,
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out);

/**
 * \brief Uploads an executable graph in a stream
 *
 * Uploads \p hGraphExec to the device in \p hStream without executing it. Uploads of
 * the same \p hGraphExec will be serialized. Each upload is ordered behind both any
 * previous work in \p hStream and any previous launches of \p hGraphExec.
 * Uses memory cached by \p stream to back the allocations owned by \p graphExec.
 *
 * \param hGraphExec - Executable graph to upload
 * \param hStream    - Stream in which to upload the graph
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * \notefnerr
 * \note_init_rt
 *
 * \sa
 * ::cudaGraphInstantiate,
 * ::cudaGraphLaunch,
 * ::cudaGraphExecDestroy
 */
#if __CUDART_API_VERSION >= 11010
 extern __host__ cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
#endif

/**
 * \brief Launches an executable graph in a stream
 *
 * Executes \p graphExec in \p stream. Only one instance of \p graphExec may be executing
 * at a time. Each launch is ordered behind both any previous work in \p stream
 * and any previous launches of \p graphExec. To execute a graph concurrently, it must be
 * instantiated multiple times into multiple executable graphs.
 *
 * If any allocations created by \p graphExec remain unfreed (from a previous launch) and
 * \p graphExec was not instantiated with ::cudaGraphInstantiateFlagAutoFreeOnLaunch,
 * the launch will fail with ::cudaErrorInvalidValue.
 *
 * \param graphExec - Executable graph to launch
 * \param stream    - Stream in which to launch the graph
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphInstantiate,
 * ::cudaGraphUpload,
 * ::cudaGraphExecDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);

/**
 * \brief Destroys an executable graph
 *
 * Destroys the executable graph specified by \p graphExec.
 *
 * \param graphExec - Executable graph to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::cudaGraphInstantiate,
 * ::cudaGraphUpload,
 * ::cudaGraphLaunch
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec);

/**
 * \brief Destroys a graph
 *
 * Destroys the graph specified by \p graph, as well as all of its nodes.
 *
 * \param graph - Graph to destroy
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 * \note_destroy_ub
 *
 * \sa
 * ::cudaGraphCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph);

/**
 * \brief Write a DOT file describing graph structure
 *
 * Using the provided \p graph, write to \p path a DOT formatted description of the graph.
 * By default this includes the graph topology, node types, node id, kernel names and memcpy direction.
 * \p flags can be specified to write more detailed information about each node type such as
 * parameter values, kernel attributes, node and function handles.
 *
 * \param graph - The graph to create a DOT file from
 * \param path  - The path to write the DOT file to
 * \param flags - Flags from cudaGraphDebugDotFlags for specifying which additional node information to write
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorOperatingSystem
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags);

/**
 * \brief Create a user object
 *
 * Create a user object with the specified destructor callback and initial reference count. The
 * initial references are owned by the caller.
 *
 * Destructor callbacks cannot make CUDA API calls and should avoid blocking behavior, as they
 * are executed by a shared internal thread. Another thread may be signaled to perform such
 * actions, if it does not block forward progress of tasks scheduled through CUDA.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param object_out      - Location to return the user object handle
 * \param ptr             - The pointer to pass to the destroy function
 * \param destroy         - Callback to free the user object when it is no longer in use
 * \param initialRefcount - The initial refcount to create the object with, typically 1. The
 *                          initial references are owned by the calling thread.
 * \param flags           - Currently it is required to pass ::cudaUserObjectNoDestructorSync,
 *                          which is the only defined flag. This indicates that the destroy
 *                          callback cannot be waited on by any CUDA API. Users requiring
 *                          synchronization of the callback should signal its completion
 *                          manually.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa
 * ::cudaUserObjectRetain,
 * ::cudaUserObjectRelease,
 * ::cudaGraphRetainUserObject,
 * ::cudaGraphReleaseUserObject,
 * ::cudaGraphCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);

/**
 * \brief Retain a reference to a user object
 *
 * Retains new references to a user object. The new references are owned by the caller.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param object - The object to retain
 * \param count  - The number of references to retain, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa
 * ::cudaUserObjectCreate,
 * ::cudaUserObjectRelease,
 * ::cudaGraphRetainUserObject,
 * ::cudaGraphReleaseUserObject,
 * ::cudaGraphCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaUserObjectRetain(cudaUserObject_t object, unsigned int count __dv(1));

/**
 * \brief Release a reference to a user object
 *
 * Releases user object references owned by the caller. The object's destructor is invoked if
 * the reference count reaches zero.
 *
 * It is undefined behavior to release references not owned by the caller, or to use a user
 * object handle after all references are released.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param object - The object to release
 * \param count  - The number of references to release, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa
 * ::cudaUserObjectCreate,
 * ::cudaUserObjectRetain,
 * ::cudaGraphRetainUserObject,
 * ::cudaGraphReleaseUserObject,
 * ::cudaGraphCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaUserObjectRelease(cudaUserObject_t object, unsigned int count __dv(1));

/**
 * \brief Retain a reference to a user object from a graph
 *
 * Creates or moves user object references that will be owned by a CUDA graph.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param graph  - The graph to associate the reference with
 * \param object - The user object to retain a reference for
 * \param count  - The number of references to add to the graph, typically 1. Must be
 *                 nonzero and not larger than INT_MAX.
 * \param flags  - The optional flag ::cudaGraphUserObjectMove transfers references
 *                 from the calling thread, rather than create new references. Pass 0
 *                 to create new references.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa
 * ::cudaUserObjectCreate
 * ::cudaUserObjectRetain,
 * ::cudaUserObjectRelease,
 * ::cudaGraphReleaseUserObject,
 * ::cudaGraphCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count __dv(1), unsigned int flags __dv(0));

/**
 * \brief Release a user object reference from a graph
 *
 * Releases user object references owned by a graph.
 *
 * See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.
 *
 * \param graph  - The graph that will release the reference
 * \param object - The user object to release a reference for
 * \param count  - The number of references to release, typically 1. Must be nonzero
 *                 and not larger than INT_MAX.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa
 * ::cudaUserObjectCreate
 * ::cudaUserObjectRetain,
 * ::cudaUserObjectRelease,
 * ::cudaGraphRetainUserObject,
 * ::cudaGraphCreate
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count __dv(1));

/** @} */ /* END CUDART_GRAPH */

/**
 * \defgroup CUDART_DRIVER_ENTRY_POINT Driver Entry Point Access
 *
 * ___MANBRIEF___ driver entry point access functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the driver entry point access functions of CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Returns the requested driver API function pointer
 *
 * Returns in \p **funcPtr the address of the CUDA driver function for the requested flags.
 *
 * For a requested driver symbol, if the CUDA version in which the driver symbol was
 * introduced is less than or equal to the CUDA runtime version, the API will return
 * the function pointer to the corresponding versioned driver function.
 *
 * The pointer returned by the API should be cast to a function pointer matching the
 * requested driver function's definition in the API header file. The function pointer
 * typedef can be picked up from the corresponding typedefs header file. For example,
 * cudaTypedefs.h consists of function pointer typedefs for driver APIs defined in cuda.h.
 *
 * The API will return ::cudaErrorSymbolNotFound if the requested driver function is not
 * supported on the platform, no ABI compatible driver function exists for the CUDA runtime
 * version or if the driver symbol is invalid.
 *
 * The requested flags can be:
 * - ::cudaEnableDefault: This is the default mode. This is equivalent to
 *   ::cudaEnablePerThreadDefaultStream if the code is compiled with
 *   --default-stream per-thread compilation flag or the macro CUDA_API_PER_THREAD_DEFAULT_STREAM
 *   is defined; ::cudaEnableLegacyStream otherwise.
 * - ::cudaEnableLegacyStream: This will enable the search for all driver symbols
 *   that match the requested driver symbol name except the corresponding per-thread versions.
 * - ::cudaEnablePerThreadDefaultStream: This will enable the search for all
 *   driver symbols that match the requested driver symbol name including the per-thread
 *   versions. If a per-thread version is not found, the API will return the legacy version
 *   of the driver function.
 *
 * \param symbol - The base name of the driver API function to look for. As an example,
 *                 for the driver API ::cuMemAlloc_v2, \p symbol would be cuMemAlloc.
 *                 Note that the API will use the CUDA runtime version to return the
 *                 address to the most recent ABI compatible driver symbol, ::cuMemAlloc
 *                 or ::cuMemAlloc_v2.
 * \param funcPtr - Location to return the function pointer to the requested driver function
 * \param flags -  Flags to specify search options.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorNotSupported,
 * ::cudaErrorSymbolNotFound
 * \note_version_mixing
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cuGetProcAddress
 */
extern __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags);

/** @} */ /* END CUDART_DRIVER_ENTRY_POINT */

/** \cond impl_private */
extern __host__ cudaError_t CUDARTAPI cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId);
/** \endcond impl_private */

/**
 * \defgroup CUDART_HIGHLEVEL C++ API Routines
 *
 * ___MANBRIEF___ C++ high level API functions of the CUDA runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the C++ high level API functions of the CUDA runtime
 * application programming interface. To use these functions, your
 * application needs to be compiled with the \p nvcc compiler.
 *
 * \brief C++-style interface built on top of CUDA runtime API
 */

/**
 * \defgroup CUDART_DRIVER Interactions with the CUDA Driver API
 *
 * ___MANBRIEF___ interactions between CUDA Driver API and CUDA Runtime API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the interactions between the CUDA Driver API and the CUDA Runtime API
 *
 * @{
 *
 * \section CUDART_CUDA_primary Primary Contexts
 *
 * There exists a one to one relationship between CUDA devices in the CUDA Runtime
 * API and ::CUcontext s in the CUDA Driver API within a process.  The specific
 * context which the CUDA Runtime API uses for a device is called the device's
 * primary context.  From the perspective of the CUDA Runtime API, a device and 
 * its primary context are synonymous.
 *
 * \section CUDART_CUDA_init Initialization and Tear-Down
 *
 * CUDA Runtime API calls operate on the CUDA Driver API ::CUcontext which is current to
 * to the calling host thread.  
 *
 * The function ::cudaSetDevice() makes the primary context for the
 * specified device current to the calling thread by calling ::cuCtxSetCurrent().
 *
 * The CUDA Runtime API will automatically initialize the primary context for
 * a device at the first CUDA Runtime API call which requires an active context.
 * If no ::CUcontext is current to the calling thread when a CUDA Runtime API call 
 * which requires an active context is made, then the primary context for a device 
 * will be selected, made current to the calling thread, and initialized.
 *
 * The context which the CUDA Runtime API initializes will be initialized using 
 * the parameters specified by the CUDA Runtime API functions
 * ::cudaSetDeviceFlags(), 
 * ::cudaD3D9SetDirect3DDevice(), 
 * ::cudaD3D10SetDirect3DDevice(), 
 * ::cudaD3D11SetDirect3DDevice(), 
 * ::cudaGLSetGLDevice(), and
 * ::cudaVDPAUSetVDPAUDevice().
 * Note that these functions will fail with ::cudaErrorSetOnActiveProcess if they are 
 * called when the primary context for the specified device has already been initialized.
 * (or if the current device has already been initialized, in the case of 
 * ::cudaSetDeviceFlags()). 
 *
 * Primary contexts will remain active until they are explicitly deinitialized 
 * using ::cudaDeviceReset().  The function ::cudaDeviceReset() will deinitialize the 
 * primary context for the calling thread's current device immediately.  The context 
 * will remain current to all of the threads that it was current to.  The next CUDA 
 * Runtime API call on any thread which requires an active context will trigger the 
 * reinitialization of that device's primary context.
 *
 * Note that primary contexts are shared resources. It is recommended that
 * the primary context not be reset except just before exit or to recover from an
 * unspecified launch failure.
 * 
 * \section CUDART_CUDA_context Context Interoperability
 *
 * Note that the use of multiple ::CUcontext s per device within a single process 
 * will substantially degrade performance and is strongly discouraged.  Instead,
 * it is highly recommended that the implicit one-to-one device-to-context mapping
 * for the process provided by the CUDA Runtime API be used.
 *
 * If a non-primary ::CUcontext created by the CUDA Driver API is current to a
 * thread then the CUDA Runtime API calls to that thread will operate on that 
 * ::CUcontext, with some exceptions listed below.  Interoperability between data
 * types is discussed in the following sections.
 *
 * The function ::cudaPointerGetAttributes() will return the error 
 * ::cudaErrorIncompatibleDriverContext if the pointer being queried was allocated by a 
 * non-primary context.  The function ::cudaDeviceEnablePeerAccess() and the rest of 
 * the peer access API may not be called when a non-primary ::CUcontext is current.  
 * To use the pointer query and peer access APIs with a context created using the 
 * CUDA Driver API, it is necessary that the CUDA Driver API be used to access
 * these features.
 *
 * All CUDA Runtime API state (e.g, global variables' addresses and values) travels
 * with its underlying ::CUcontext.  In particular, if a ::CUcontext is moved from one 
 * thread to another then all CUDA Runtime API state will move to that thread as well.
 *
 * Please note that attaching to legacy contexts (those with a version of 3010 as returned
 * by ::cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return
 * ::cudaErrorIncompatibleDriverContext in such cases.
 *
 * \section CUDART_CUDA_stream Interactions between CUstream and cudaStream_t
 *
 * The types ::CUstream and ::cudaStream_t are identical and may be used interchangeably.
 *
 * \section CUDART_CUDA_event Interactions between CUevent and cudaEvent_t
 *
 * The types ::CUevent and ::cudaEvent_t are identical and may be used interchangeably.
 *
 * \section CUDART_CUDA_array Interactions between CUarray and cudaArray_t 
 *
 * The types ::CUarray and struct ::cudaArray * represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUarray in a CUDA Runtime API function which takes a struct ::cudaArray *,
 * it is necessary to explicitly cast the ::CUarray to a struct ::cudaArray *.
 *
 * In order to use a struct ::cudaArray * in a CUDA Driver API function which takes a ::CUarray,
 * it is necessary to explicitly cast the struct ::cudaArray * to a ::CUarray .
 *
 * \section CUDART_CUDA_graphicsResource Interactions between CUgraphicsResource and cudaGraphicsResource_t
 *
 * The types ::CUgraphicsResource and ::cudaGraphicsResource_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUgraphicsResource in a CUDA Runtime API function which takes a 
 * ::cudaGraphicsResource_t, it is necessary to explicitly cast the ::CUgraphicsResource 
 * to a ::cudaGraphicsResource_t.
 *
 * In order to use a ::cudaGraphicsResource_t in a CUDA Driver API function which takes a
 * ::CUgraphicsResource, it is necessary to explicitly cast the ::cudaGraphicsResource_t 
 * to a ::CUgraphicsResource.
 *
 * \section CUDART_CUDA_texture_objects Interactions between CUtexObject * and cudaTextureObject_t
 *
 * The types ::CUtexObject * and ::cudaTextureObject_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUtexObject * in a CUDA Runtime API function which takes a ::cudaTextureObject_t,
 * it is necessary to explicitly cast the ::CUtexObject * to a ::cudaTextureObject_t.
 *
 * In order to use a ::cudaTextureObject_t in a CUDA Driver API function which takes a ::CUtexObject *,
 * it is necessary to explicitly cast the ::cudaTextureObject_t to a ::CUtexObject *.
 *
 * \section CUDART_CUDA_surface_objects Interactions between CUsurfObject * and cudaSurfaceObject_t
 *
 * The types ::CUsurfObject * and ::cudaSurfaceObject_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::CUsurfObject * in a CUDA Runtime API function which takes a ::cudaSurfaceObjec_t,
 * it is necessary to explicitly cast the ::CUsurfObject * to a ::cudaSurfaceObject_t.
 *
 * In order to use a ::cudaSurfaceObject_t in a CUDA Driver API function which takes a ::CUsurfObject *,
 * it is necessary to explicitly cast the ::cudaSurfaceObject_t to a ::CUsurfObject *.
 *
 * \section CUDART_CUDA_module Interactions between CUfunction and cudaFunction_t
 *
 * The types ::CUfunction and ::cudaFunction_t represent the same data type and may be used
 * interchangeably by casting the two types between each other.
 *
 * In order to use a ::cudaFunction_t in a CUDA Driver API function which takes a ::CUfunction,
 * it is necessary to explicitly cast the ::cudaFunction_t to a ::CUfunction.
 *
 */

 /**
  * \brief Get pointer to device entry function that matches entry function \p symbolPtr
  *
  * Returns in \p functionPtr the device entry function corresponding to the symbol \p symbolPtr.
  *
  * \param functionPtr     - Returns the device entry function
  * \param symbolPtr       - Pointer to device entry function to search for
  *
  * \return
  * ::cudaSuccess
  *
  */
extern __host__ cudaError_t CUDARTAPI_CDECL cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr);

/** @} */ /* END CUDART_DRIVER */

#if defined(__CUDA_API_VERSION_INTERNAL)
    #undef cudaMemcpy
    #undef cudaMemcpyToSymbol
    #undef cudaMemcpyFromSymbol
    #undef cudaMemcpy2D
    #undef cudaMemcpyToArray
    #undef cudaMemcpy2DToArray
    #undef cudaMemcpyFromArray
    #undef cudaMemcpy2DFromArray
    #undef cudaMemcpyArrayToArray
    #undef cudaMemcpy2DArrayToArray
    #undef cudaMemcpy3D
    #undef cudaMemcpy3DPeer
    #undef cudaMemset
    #undef cudaMemset2D
    #undef cudaMemset3D
    #undef cudaMemcpyAsync
    #undef cudaMemcpyToSymbolAsync
    #undef cudaMemcpyFromSymbolAsync
    #undef cudaMemcpy2DAsync
    #undef cudaMemcpyToArrayAsync
    #undef cudaMemcpy2DToArrayAsync
    #undef cudaMemcpyFromArrayAsync
    #undef cudaMemcpy2DFromArrayAsync
    #undef cudaMemcpy3DAsync
    #undef cudaMemcpy3DPeerAsync
    #undef cudaMemsetAsync
    #undef cudaMemset2DAsync
    #undef cudaMemset3DAsync
    #undef cudaStreamQuery
    #undef cudaStreamGetFlags
    #undef cudaStreamGetPriority
    #undef cudaEventRecord
    #undef cudaEventRecordWithFlags
    #undef cudaStreamWaitEvent
    #undef cudaStreamAddCallback
    #undef cudaStreamAttachMemAsync
    #undef cudaStreamSynchronize
    #undef cudaLaunchKernel
    #undef cudaLaunchHostFunc
    #undef cudaMemPrefetchAsync
    #undef cudaLaunchCooperativeKernel
    #undef cudaSignalExternalSemaphoresAsync
    #undef cudaWaitExternalSemaphoresAsync
    #undef cudaGraphUpload
    #undef cudaGraphLaunch
    #undef cudaStreamBeginCapture
    #undef cudaStreamEndCapture
    #undef cudaStreamIsCapturing
    #undef cudaStreamGetCaptureInfo
    #undef cudaStreamGetCaptureInfo_v2
    #undef cudaStreamCopyAttributes
    #undef cudaStreamGetAttribute
    #undef cudaStreamSetAttribute
    #undef cudaMallocAsync
    #undef cudaFreeAsync
    #undef cudaMallocFromPoolAsync
    #undef cudaGetDriverEntryPoint

    extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
    extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count);
    extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
    extern __host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream);
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority);
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream __dv(0), unsigned int flags __dv(0));
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
    extern __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags);
    extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags);
    extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
    extern __host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams_v1 *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync_ptsz(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams_v1 *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams_v1 *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync_ptsz(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams_v1 *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
    extern __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
    extern __host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);
    extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out);
    extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out __dv(0), cudaGraph_t *graph_out __dv(0), const cudaGraphNode_t **dependencies_out __dv(0), size_t *numDependencies_out __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies_ptsz(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags __dv(0));
    extern __host__ cudaError_t CUDARTAPI cudaStreamCopyAttributes(cudaStream_t dstStream, cudaStream_t srcStream);
    extern __host__ cudaError_t CUDARTAPI cudaStreamGetAttribute(cudaStream_t stream, enum cudaStreamAttrID attr, union cudaStreamAttrValue *value);
    extern __host__ cudaError_t CUDARTAPI cudaStreamSetAttribute(cudaStream_t stream, enum cudaStreamAttrID attr, const union cudaStreamAttrValue *param);

    extern __host__ cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);
    extern __host__ cudaError_t CUDARTAPI cudaFreeAsync(void *devPtr, cudaStream_t hStream);
    extern __host__ cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags);

#elif defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)
    // nvcc stubs reference the 'cudaLaunch'/'cudaLaunchKernel' identifier even if it was defined
    // to 'cudaLaunch_ptsz'/'cudaLaunchKernel_ptsz'. Redirect through a static inline function.
    #undef cudaLaunchKernel
    static __inline__ __host__ cudaError_t cudaLaunchKernel(const void *func, 
                                                            dim3 gridDim, dim3 blockDim, 
                                                            void **args, size_t sharedMem, 
                                                            cudaStream_t stream)
    {
        return cudaLaunchKernel_ptsz(func, gridDim, blockDim, args, sharedMem, stream);
    }
    #define cudaLaunchKernel __CUDART_API_PTSZ(cudaLaunchKernel)
#endif

#if defined(__cplusplus)
}

#endif /* __cplusplus */

#undef EXCLUDE_FROM_RTC
#endif /* !__CUDACC_RTC__ */

#undef __dv
#undef __CUDA_DEPRECATED

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_API_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_API_H__
#endif

#endif /* !__CUDA_RUNTIME_API_H__ */
