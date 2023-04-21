/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUDA_DEVICE_RUNTIME_API_H__)
#define __CUDA_DEVICE_RUNTIME_API_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDACC_RTC__)

#if !defined(__CUDACC_INTERNAL_NO_STUBS__) && !defined(__CUDACC_RDC__) && !defined(__CUDACC_EWP__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350) && !defined(__CUDADEVRT_INTERNAL__)

#if defined(__cplusplus)
extern "C" {
#endif

struct cudaFuncAttributes;


inline __device__  cudaError_t CUDARTAPI cudaMalloc(void **p, size_t s) 
{ 
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c) 
{ 
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
  return cudaErrorUnknown;
}


#if defined(__cplusplus)
}
#endif

#endif /* !defined(__CUDACC_INTERNAL_NO_STUBS__) && !defined(__CUDACC_RDC__) &&  !defined(__CUDACC_EWP__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350) && !defined(__CUDADEVRT_INTERNAL__) */

#endif /* !defined(__CUDACC_RTC__) */

#if defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
# define __DEPRECATED__(msg)
#elif defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(__CUDA_ARCH__) && !defined(__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING)
# define __CDPRT_DEPRECATED(func_name) __DEPRECATED__("Use of "#func_name" from device code is deprecated and will not be supported in a future release. Disable this warning with -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING.")
#else
# define __CDPRT_DEPRECATED(func_name)
#endif

#if defined(__cplusplus) && defined(__CUDACC__)         /* Visible to nvcc front-end only */
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)   // Visible to SM>=3.5 and "__host__ __device__" only

#include "driver_types.h"
#include "crt/host_defines.h"

extern "C"
{
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
extern __device__ __cudart_builtin__ __CDPRT_DEPRECATED(cudaDeviceSynchronize) cudaError_t CUDARTAPI cudaDeviceSynchronize(void);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaDeviceSynchronizeDeprecationAvoidance(void);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void);
extern __device__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error);
extern __device__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync_ptsz(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync_ptsz(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion);

/**
 * \ingroup CUDART_EXECUTION
 * \brief Obtains a parameter buffer
 *
 * Obtains a parameter buffer which can be filled with parameters for a kernel launch.
 * Parameters passed to ::cudaLaunchDevice must be allocated via this function.
 *
 * This is a low level API and can only be accessed from Parallel Thread Execution (PTX).
 * CUDA user code should use <<< >>> to launch kernels.
 *
 * \param alignment - Specifies alignment requirement of the parameter buffer
 * \param size      - Specifies size requirement in bytes
 *
 * \return
 * Returns pointer to the allocated parameterBuffer
 * \notefnerr
 *
 * \sa cudaLaunchDevice
 */
extern __device__ __cudart_builtin__ void * CUDARTAPI cudaGetParameterBuffer(size_t alignment, size_t size);

/**
 * \ingroup CUDART_EXECUTION
 * \brief Launches a specified kernel
 *
 * Launches a specified kernel with the specified parameter buffer. A parameter buffer can be obtained
 * by calling ::cudaGetParameterBuffer().
 *
 * This is a low level API and can only be accessed from Parallel Thread Execution (PTX).
 * CUDA user code should use <<< >>> to launch the kernels.
 *
 * \param func            - Pointer to the kernel to be launched
 * \param parameterBuffer - Holds the parameters to the launched kernel. parameterBuffer can be NULL. (Optional)
 * \param gridDimension   - Specifies grid dimensions
 * \param blockDimension  - Specifies block dimensions
 * \param sharedMemSize   - Specifies size of shared memory
 * \param stream          - Specifies the stream to be used
 *
 * \return
 * ::cudaSuccess, ::cudaErrorInvalidDevice, ::cudaErrorLaunchMaxDepthExceeded, ::cudaErrorInvalidConfiguration,
 * ::cudaErrorStartupFailure, ::cudaErrorLaunchPendingCountExceeded, ::cudaErrorLaunchOutOfResources
 * \notefnerr
 * \n Please refer to Execution Configuration and Parameter Buffer Layout from the CUDA Programming
 * Guide for the detailed descriptions of launch configuration and parameter layout respectively.
 *
 * \sa cudaGetParameterBuffer
 */
extern __device__ __cudart_builtin__ void * CUDARTAPI cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream);

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) && defined(__CUDA_ARCH__)
    // When compiling for the device and per thread default stream is enabled, add
    // a static inline redirect to the per thread stream entry points.

    static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI
    cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream)
    {
        return cudaLaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream);
    }

    static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI
    cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
    {
        return cudaLaunchDeviceV2_ptsz(parameterBuffer, stream);
    }
#else
    extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
    extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);
#endif

extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags);

extern __device__ __cudart_builtin__ unsigned long long CUDARTAPI cudaCGGetIntrinsicHandle(enum cudaCGScope scope);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGSynchronize(unsigned long long handle, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGSynchronizeGrid(unsigned long long handle, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGGetSize(unsigned int *numThreads, unsigned int *numGrids, unsigned long long handle);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGGetRank(unsigned int *threadRank, unsigned int *gridRank, unsigned long long handle);
}

template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaMalloc(T **devPtr, size_t size);
template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, T *entry);
template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize);
template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned int flags);


#endif // !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
#endif /* defined(__cplusplus) && defined(__CUDACC__) */

#undef __DEPRECATED__
#undef __CDPRT_DEPRECATED

#endif /* !__CUDA_DEVICE_RUNTIME_API_H__ */
