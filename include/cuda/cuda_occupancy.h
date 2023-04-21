/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
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

/**
 * CUDA Occupancy Calculator
 *
 * NAME
 *
 *   cudaOccMaxActiveBlocksPerMultiprocessor,
 *   cudaOccMaxPotentialOccupancyBlockSize,
 *   cudaOccMaxPotentialOccupancyBlockSizeVariableSMem
 *   cudaOccAvailableDynamicSMemPerBlock
 *
 * DESCRIPTION
 *
 *   The CUDA occupancy calculator provides a standalone, programmatical
 *   interface to compute the occupancy of a function on a device. It can also
 *   provide occupancy-oriented launch configuration suggestions.
 *
 *   The function and device are defined by the user through
 *   cudaOccFuncAttributes, cudaOccDeviceProp, and cudaOccDeviceState
 *   structures. All APIs require all 3 of them.
 *
 *   See the structure definition for more details about the device / function
 *   descriptors.
 *
 *   See each API's prototype for API usage.
 *
 * COMPATIBILITY
 *
 *   The occupancy calculator will be updated on each major CUDA toolkit
 *   release. It does not provide forward compatibility, i.e. new hardwares
 *   released after this implementation's release will not be supported.
 *
 * NOTE
 *
 *   If there is access to CUDA runtime, and the sole intent is to calculate
 *   occupancy related values on one of the accessible CUDA devices, using CUDA
 *   runtime's occupancy calculation APIs is recommended.
 *
 */

#ifndef __cuda_occupancy_h__
#define __cuda_occupancy_h__

#include <stddef.h>
#include <limits.h>
#include <string.h>


// __OCC_INLINE will be undefined at the end of this header
//
#ifdef __CUDACC__
#define __OCC_INLINE inline __host__ __device__
#elif defined _MSC_VER
#define __OCC_INLINE __inline
#else // GNUCC assumed
#define __OCC_INLINE inline
#endif

enum cudaOccError_enum {
    CUDA_OCC_SUCCESS              = 0,  // no error encountered
    CUDA_OCC_ERROR_INVALID_INPUT  = 1,  // input parameter is invalid
    CUDA_OCC_ERROR_UNKNOWN_DEVICE = 2,  // requested device is not supported in
                                        // current implementation or device is
                                        // invalid
};
typedef enum cudaOccError_enum       cudaOccError;

typedef struct cudaOccResult         cudaOccResult;
typedef struct cudaOccDeviceProp     cudaOccDeviceProp;
typedef struct cudaOccFuncAttributes cudaOccFuncAttributes;
typedef struct cudaOccDeviceState    cudaOccDeviceState;

/**
 * The CUDA occupancy calculator computes the occupancy of the function
 * described by attributes with the given block size (blockSize), static device
 * properties (properties), dynamic device states (states) and per-block dynamic
 * shared memory allocation (dynamicSMemSize) in bytes, and output it through
 * result along with other useful information. The occupancy is computed in
 * terms of the maximum number of active blocks per multiprocessor. The user can
 * then convert it to other metrics, such as number of active warps.
 *
 * RETURN VALUE
 *
 * The occupancy and related information is returned through result.
 *
 * If result->activeBlocksPerMultiprocessor is 0, then the given parameter
 * combination cannot run on the device.
 *
 * ERRORS
 *
 *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
 *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 */
static __OCC_INLINE
cudaOccError cudaOccMaxActiveBlocksPerMultiprocessor(
    cudaOccResult               *result,           // out
    const cudaOccDeviceProp     *properties,       // in
    const cudaOccFuncAttributes *attributes,       // in
    const cudaOccDeviceState    *state,            // in
    int                          blockSize,        // in
    size_t                       dynamicSmemSize); // in

/**
 * The CUDA launch configurator C API suggests a grid / block size pair (in
 * minGridSize and blockSize) that achieves the best potential occupancy
 * (i.e. maximum number of active warps with the smallest number of blocks) for
 * the given function described by attributes, on a device described by
 * properties with settings in state.
 *
 * If per-block dynamic shared memory allocation is not needed, the user should
 * leave both blockSizeToDynamicSMemSize and dynamicSMemSize as 0.
 *
 * If per-block dynamic shared memory allocation is needed, then if the dynamic
 * shared memory size is constant regardless of block size, the size should be
 * passed through dynamicSMemSize, and blockSizeToDynamicSMemSize should be
 * NULL.
 *
 * Otherwise, if the per-block dynamic shared memory size varies with different
 * block sizes, the user needs to provide a pointer to an unary function through
 * blockSizeToDynamicSMemSize that computes the dynamic shared memory needed by
 * a block of the function for any given block size. dynamicSMemSize is
 * ignored. An example signature is:
 *
 *    // Take block size, returns dynamic shared memory needed
 *    size_t blockToSmem(int blockSize);
 *
 * RETURN VALUE
 *
 * The suggested block size and the minimum number of blocks needed to achieve
 * the maximum occupancy are returned through blockSize and minGridSize.
 *
 * If *blockSize is 0, then the given combination cannot run on the device.
 *
 * ERRORS
 *
 *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
 *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 *
 */
static __OCC_INLINE
cudaOccError cudaOccMaxPotentialOccupancyBlockSize(
    int                         *minGridSize,      // out
    int                         *blockSize,        // out
    const cudaOccDeviceProp     *properties,       // in
    const cudaOccFuncAttributes *attributes,       // in
    const cudaOccDeviceState    *state,            // in
    size_t                     (*blockSizeToDynamicSMemSize)(int), // in
    size_t                       dynamicSMemSize); // in

/**
 * The CUDA launch configurator C++ API suggests a grid / block size pair (in
 * minGridSize and blockSize) that achieves the best potential occupancy
 * (i.e. the maximum number of active warps with the smallest number of blocks)
 * for the given function described by attributes, on a device described by
 * properties with settings in state.
 *
 * If per-block dynamic shared memory allocation is 0 or constant regardless of
 * block size, the user can use cudaOccMaxPotentialOccupancyBlockSize to
 * configure the launch. A constant dynamic shared memory allocation size in
 * bytes can be passed through dynamicSMemSize.
 *
 * Otherwise, if the per-block dynamic shared memory size varies with different
 * block sizes, the user needs to use
 * cudaOccMaxPotentialOccupancyBlockSizeVariableSmem instead, and provide a
 * functor / pointer to an unary function (blockSizeToDynamicSMemSize) that
 * computes the dynamic shared memory needed by func for any given block
 * size. An example signature is:
 *
 *  // Take block size, returns per-block dynamic shared memory needed
 *  size_t blockToSmem(int blockSize);
 *
 * RETURN VALUE
 *
 * The suggested block size and the minimum number of blocks needed to achieve
 * the maximum occupancy are returned through blockSize and minGridSize.
 *
 * If *blockSize is 0, then the given combination cannot run on the device.
 *
 * ERRORS
 *
 *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
 *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 *
 */

#if defined(__cplusplus)
namespace {

__OCC_INLINE
cudaOccError cudaOccMaxPotentialOccupancyBlockSize(
    int                         *minGridSize,          // out
    int                         *blockSize,            // out
    const cudaOccDeviceProp     *properties,           // in
    const cudaOccFuncAttributes *attributes,           // in
    const cudaOccDeviceState    *state,                // in
    size_t                       dynamicSMemSize = 0); // in

template <typename UnaryFunction>
__OCC_INLINE
cudaOccError cudaOccMaxPotentialOccupancyBlockSizeVariableSMem(
    int                         *minGridSize,          // out
    int                         *blockSize,            // out
    const cudaOccDeviceProp     *properties,           // in
    const cudaOccFuncAttributes *attributes,           // in
    const cudaOccDeviceState    *state,                // in
    UnaryFunction                blockSizeToDynamicSMemSize); // in

} // namespace anonymous
#endif // defined(__cplusplus)

/**
 *
 * The CUDA dynamic shared memory calculator computes the maximum size of 
 * per-block dynamic shared memory if we want to place numBlocks blocks
 * on an SM.
 *
 * RETURN VALUE
 *
 * Returns in *dynamicSmemSize the maximum size of dynamic shared memory to allow 
 * numBlocks blocks per SM.
 *
 * ERRORS
 *
 *     CUDA_OCC_ERROR_INVALID_INPUT   input parameter is invalid.
 *     CUDA_OCC_ERROR_UNKNOWN_DEVICE  requested device is not supported in
 *     current implementation or device is invalid
 *
 */
static __OCC_INLINE
cudaOccError cudaOccAvailableDynamicSMemPerBlock(
    size_t                      *dynamicSmemSize,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    int                         numBlocks,
    int                         blockSize);

/**
 * Data structures
 *
 * These structures are subject to change for future architecture and CUDA
 * releases. C users should initialize the structure as {0}.
 *
 */

/**
 * Device descriptor
 *
 * This structure describes a device.
 */
struct cudaOccDeviceProp {
    int    computeMajor;                // Compute capability major version
    int    computeMinor;                // Compute capability minor
                                        // version. None supported minor version
                                        // may cause error
    int    maxThreadsPerBlock;          // Maximum number of threads per block
    int    maxThreadsPerMultiprocessor; // Maximum number of threads per SM
                                        // i.e. (Max. number of warps) x (warp
                                        // size)
    int    regsPerBlock;                // Maximum number of registers per block
    int    regsPerMultiprocessor;       // Maximum number of registers per SM
    int    warpSize;                    // Warp size
    size_t sharedMemPerBlock;           // Maximum shared memory size per block
    size_t sharedMemPerMultiprocessor;  // Maximum shared memory size per SM
    int    numSms;                      // Number of SMs available
    size_t sharedMemPerBlockOptin;      // Maximum optin shared memory size per block
    size_t reservedSharedMemPerBlock;   // Shared memory per block reserved by driver

#ifdef __cplusplus
    // This structure can be converted from a cudaDeviceProp structure for users
    // that use this header in their CUDA applications.
    //
    // If the application have access to the CUDA Runtime API, the application
    // can obtain the device properties of a CUDA device through
    // cudaGetDeviceProperties, and initialize a cudaOccDeviceProp with the
    // cudaDeviceProp structure.
    //
    // Example:
    /*
     {
         cudaDeviceProp prop;

         cudaGetDeviceProperties(&prop, ...);

         cudaOccDeviceProp occProp = prop;

         ...

         cudaOccMaxPotentialOccupancyBlockSize(..., &occProp, ...);
     }
     */
    //
    template<typename DeviceProp>
    __OCC_INLINE
    cudaOccDeviceProp(const DeviceProp &props)
    :   computeMajor                (props.major),
        computeMinor                (props.minor),
        maxThreadsPerBlock          (props.maxThreadsPerBlock),
        maxThreadsPerMultiprocessor (props.maxThreadsPerMultiProcessor),
        regsPerBlock                (props.regsPerBlock),
        regsPerMultiprocessor       (props.regsPerMultiprocessor),
        warpSize                    (props.warpSize),
        sharedMemPerBlock           (props.sharedMemPerBlock),
        sharedMemPerMultiprocessor  (props.sharedMemPerMultiprocessor),
        numSms                      (props.multiProcessorCount),
        sharedMemPerBlockOptin      (props.sharedMemPerBlockOptin),
        reservedSharedMemPerBlock   (props.reservedSharedMemPerBlock)
    {}

    __OCC_INLINE
    cudaOccDeviceProp()
    :   computeMajor                (0),
        computeMinor                (0),
        maxThreadsPerBlock          (0),
        maxThreadsPerMultiprocessor (0),
        regsPerBlock                (0),
        regsPerMultiprocessor       (0),
        warpSize                    (0),
        sharedMemPerBlock           (0),
        sharedMemPerMultiprocessor  (0),
        numSms                      (0),
        sharedMemPerBlockOptin      (0),
        reservedSharedMemPerBlock   (0)
    {}
#endif // __cplusplus
};

/**
 * Partitioned global caching option
 */
typedef enum cudaOccPartitionedGCConfig_enum {
    PARTITIONED_GC_OFF,        // Disable partitioned global caching
    PARTITIONED_GC_ON,         // Prefer partitioned global caching
    PARTITIONED_GC_ON_STRICT   // Force partitioned global caching
} cudaOccPartitionedGCConfig;

/**
 * Per function opt in maximum dynamic shared memory limit
 */
typedef enum cudaOccFuncShmemConfig_enum {
    FUNC_SHMEM_LIMIT_DEFAULT,   // Default shmem limit
    FUNC_SHMEM_LIMIT_OPTIN,     // Use the optin shmem limit
} cudaOccFuncShmemConfig;

/**
 * Function descriptor
 *
 * This structure describes a CUDA function.
 */
struct cudaOccFuncAttributes {
    int maxThreadsPerBlock; // Maximum block size the function can work with. If
                            // unlimited, use INT_MAX or any value greater than
                            // or equal to maxThreadsPerBlock of the device
    int numRegs;            // Number of registers used. When the function is
                            // launched on device, the register count may change
                            // due to internal tools requirements.
    size_t sharedSizeBytes; // Number of static shared memory used

    cudaOccPartitionedGCConfig partitionedGCConfig; 
                            // Partitioned global caching is required to enable
                            // caching on certain chips, such as sm_52
                            // devices. Partitioned global caching can be
                            // automatically disabled if the occupancy
                            // requirement of the launch cannot support caching.
                            //
                            // To override this behavior with caching on and
                            // calculate occupancy strictly according to the
                            // preference, set partitionedGCConfig to
                            // PARTITIONED_GC_ON_STRICT. This is especially
                            // useful for experimenting and finding launch
                            // configurations (MaxPotentialOccupancyBlockSize)
                            // that allow global caching to take effect.
                            //
                            // This flag only affects the occupancy calculation.

    cudaOccFuncShmemConfig shmemLimitConfig;
                            // Certain chips like sm_70 allow a user to opt into
                            // a higher per block limit of dynamic shared memory
                            // This optin is performed on a per function basis
                            // using the cuFuncSetAttribute function

    size_t maxDynamicSharedSizeBytes;
                            // User set limit on maximum dynamic shared memory
                            // usable by the kernel
                            // This limit is set using the cuFuncSetAttribute
                            // function.
#ifdef __cplusplus
    // This structure can be converted from a cudaFuncAttributes structure for
    // users that use this header in their CUDA applications.
    //
    // If the application have access to the CUDA Runtime API, the application
    // can obtain the function attributes of a CUDA kernel function through
    // cudaFuncGetAttributes, and initialize a cudaOccFuncAttributes with the
    // cudaFuncAttributes structure.
    //
    // Example:
    /*
      __global__ void foo() {...}

      ...

      {
          cudaFuncAttributes attr;

          cudaFuncGetAttributes(&attr, foo);

          cudaOccFuncAttributes occAttr = attr;

          ...

          cudaOccMaxPotentialOccupancyBlockSize(..., &occAttr, ...);
      }
     */
    //
    template<typename FuncAttributes>
    __OCC_INLINE
    cudaOccFuncAttributes(const FuncAttributes &attr)
    :   maxThreadsPerBlock  (attr.maxThreadsPerBlock),
        numRegs             (attr.numRegs),
        sharedSizeBytes     (attr.sharedSizeBytes),
        partitionedGCConfig (PARTITIONED_GC_OFF),
        shmemLimitConfig    (FUNC_SHMEM_LIMIT_OPTIN),
        maxDynamicSharedSizeBytes (attr.maxDynamicSharedSizeBytes)
    {}

    __OCC_INLINE
    cudaOccFuncAttributes()
    :   maxThreadsPerBlock  (0),
        numRegs             (0),
        sharedSizeBytes     (0),
        partitionedGCConfig (PARTITIONED_GC_OFF),
        shmemLimitConfig    (FUNC_SHMEM_LIMIT_DEFAULT),
        maxDynamicSharedSizeBytes (0)
    {}
#endif
};

typedef enum cudaOccCacheConfig_enum {
    CACHE_PREFER_NONE   = 0x00, // no preference for shared memory or L1 (default)
    CACHE_PREFER_SHARED = 0x01, // prefer larger shared memory and smaller L1 cache
    CACHE_PREFER_L1     = 0x02, // prefer larger L1 cache and smaller shared memory
    CACHE_PREFER_EQUAL  = 0x03  // prefer equal sized L1 cache and shared memory
} cudaOccCacheConfig;

typedef enum cudaOccCarveoutConfig_enum {
    SHAREDMEM_CARVEOUT_DEFAULT       = -1,  // no preference for shared memory or L1 (default)
    SHAREDMEM_CARVEOUT_MAX_SHARED    = 100, // prefer maximum available shared memory, minimum L1 cache
    SHAREDMEM_CARVEOUT_MAX_L1        = 0,    // prefer maximum available L1 cache, minimum shared memory
    SHAREDMEM_CARVEOUT_HALF          = 50   // prefer half of maximum available shared memory, with the rest as L1 cache
} cudaOccCarveoutConfig;

/**
 * Device state descriptor
 *
 * This structure describes device settings that affect occupancy calculation.
 */
struct cudaOccDeviceState
{
    // Cache / shared memory split preference. Deprecated on Volta 
    cudaOccCacheConfig cacheConfig; 
    // Shared memory / L1 split preference. Supported on only Volta
    int carveoutConfig;

#ifdef __cplusplus
    __OCC_INLINE
    cudaOccDeviceState()
    :   cacheConfig     (CACHE_PREFER_NONE),
        carveoutConfig  (SHAREDMEM_CARVEOUT_DEFAULT)
    {}
#endif
};

typedef enum cudaOccLimitingFactor_enum {
                                    // Occupancy limited due to:
    OCC_LIMIT_WARPS         = 0x01, // - warps available
    OCC_LIMIT_REGISTERS     = 0x02, // - registers available
    OCC_LIMIT_SHARED_MEMORY = 0x04, // - shared memory available
    OCC_LIMIT_BLOCKS        = 0x08  // - blocks available
} cudaOccLimitingFactor;

/**
 * Occupancy output
 *
 * This structure contains occupancy calculator's output.
 */
struct cudaOccResult {
    int activeBlocksPerMultiprocessor; // Occupancy
    unsigned int limitingFactors;      // Factors that limited occupancy. A bit
                                       // field that counts the limiting
                                       // factors, see cudaOccLimitingFactor
    int blockLimitRegs;                // Occupancy due to register
                                       // usage, INT_MAX if the kernel does not
                                       // use any register.
    int blockLimitSharedMem;           // Occupancy due to shared memory
                                       // usage, INT_MAX if the kernel does not
                                       // use shared memory.
    int blockLimitWarps;               // Occupancy due to block size limit
    int blockLimitBlocks;              // Occupancy due to maximum number of blocks
                                       // managable per SM
    int allocatedRegistersPerBlock;    // Actual number of registers allocated per
                                       // block
    size_t allocatedSharedMemPerBlock; // Actual size of shared memory allocated
                                       // per block
    cudaOccPartitionedGCConfig partitionedGCConfig;
                                       // Report if partitioned global caching
                                       // is actually enabled.
};

/**
 * Partitioned global caching support
 *
 * See cudaOccPartitionedGlobalCachingModeSupport
 */
typedef enum cudaOccPartitionedGCSupport_enum {
    PARTITIONED_GC_NOT_SUPPORTED,  // Partitioned global caching is not supported
    PARTITIONED_GC_SUPPORTED,      // Partitioned global caching is supported
} cudaOccPartitionedGCSupport;

/**
 * Implementation
 */

/**
 * Max compute capability supported
 */







#define __CUDA_OCC_MAJOR__ 8
#define __CUDA_OCC_MINOR__ 6


//////////////////////////////////////////
//    Mathematical Helper Functions     //
//////////////////////////////////////////

static __OCC_INLINE int __occMin(int lhs, int rhs)
{
    return rhs < lhs ? rhs : lhs;
}

static __OCC_INLINE int __occDivideRoundUp(int x, int y)
{
    return (x + (y - 1)) / y;
}

static __OCC_INLINE int __occRoundUp(int x, int y)
{
    return y * __occDivideRoundUp(x, y);
}

//////////////////////////////////////////
//      Architectural Properties        //
//////////////////////////////////////////

/**
 * Granularity of shared memory allocation
 */
static __OCC_INLINE cudaOccError cudaOccSMemAllocationGranularity(int *limit, const cudaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 3:
        case 5:
        case 6:
        case 7:
            value = 256;
            break;
        case 8:



            value = 128;
            break;
        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return CUDA_OCC_SUCCESS;
}

/**
 * Maximum number of registers per thread
 */
static __OCC_INLINE cudaOccError cudaOccRegAllocationMaxPerThread(int *limit, const cudaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 3:
        case 5:
        case 6:
            value = 255;
            break;
        case 7:
        case 8:



            value = 256;
            break;
        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return CUDA_OCC_SUCCESS;
}

/**
 * Granularity of register allocation
 */
static __OCC_INLINE cudaOccError cudaOccRegAllocationGranularity(int *limit, const cudaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 3:
        case 5:
        case 6:
        case 7:
        case 8:



            value = 256;
            break;
        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return CUDA_OCC_SUCCESS;
}

/**
 * Number of sub-partitions
 */
static __OCC_INLINE cudaOccError cudaOccSubPartitionsPerMultiprocessor(int *limit, const cudaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 3:
        case 5:
        case 7:
        case 8:



            value = 4;
            break;
        case 6:
            value = properties->computeMinor ? 4 : 2;
            break;
        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return CUDA_OCC_SUCCESS;
}


/**
 * Maximum number of blocks that can run simultaneously on a multiprocessor
 */
static __OCC_INLINE cudaOccError cudaOccMaxBlocksPerMultiprocessor(int* limit, const cudaOccDeviceProp *properties)
{
    int value;

    switch(properties->computeMajor) {
        case 3:
            value = 16;
            break;
        case 5:
        case 6:
            value = 32;
            break;
        case 7: {
            int isTuring = properties->computeMinor == 5;
            value = (isTuring) ? 16 : 32;
            break;
        }
        case 8:
            if (properties->computeMinor == 0) {
                value = 32;
            }






            else {
                value = 16;
            }
            break;





        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = value;

    return CUDA_OCC_SUCCESS;
}

/** 
 * Align up shared memory based on compute major configurations
 */
static __OCC_INLINE cudaOccError cudaOccAlignUpShmemSizeVoltaPlus(size_t *shMemSize, const cudaOccDeviceProp *properties)
{
    // Volta and Turing have shared L1 cache / shared memory, and support cache
    // configuration to trade one for the other. These values are needed to
    // map carveout config ratio to the next available architecture size
    size_t size = *shMemSize;

    switch (properties->computeMajor) {
    case 7: {
        // Turing supports 32KB and 64KB shared mem.
        int isTuring = properties->computeMinor == 5;
        if (isTuring) {
            if      (size <= 32 * 1024) {
                *shMemSize = 32 * 1024;
            }
            else if (size <= 64 * 1024) {
                *shMemSize = 64 * 1024;
            }
            else {
                return CUDA_OCC_ERROR_INVALID_INPUT;
            }
        }
        // Volta supports 0KB, 8KB, 16KB, 32KB, 64KB, and 96KB shared mem.
        else {
            if      (size == 0) {
                *shMemSize = 0;
            }
            else if (size <= 8 * 1024) {
                *shMemSize = 8 * 1024;
            }
            else if (size <= 16 * 1024) {
                *shMemSize = 16 * 1024;
            }
            else if (size <= 32 * 1024) {
                *shMemSize = 32 * 1024;
            }
            else if (size <= 64 * 1024) {
                *shMemSize = 64 * 1024;
            }
            else if (size <= 96 * 1024) {
                *shMemSize = 96 * 1024;
            }
            else {
                return CUDA_OCC_ERROR_INVALID_INPUT;
            }
        }
        break;
    }
    case 8:
        if (properties->computeMinor == 0 || properties->computeMinor == 7) {
            if      (size == 0) {
                *shMemSize = 0;
            }
            else if (size <= 8 * 1024) {
                *shMemSize = 8 * 1024;
            }
            else if (size <= 16 * 1024) {
                *shMemSize = 16 * 1024;
            }
            else if (size <= 32 * 1024) {
                *shMemSize = 32 * 1024;
            }
            else if (size <= 64 * 1024) {
                *shMemSize = 64 * 1024;
            }
            else if (size <= 100 * 1024) {
                *shMemSize = 100 * 1024;
            }
            else if (size <= 132 * 1024) {
                *shMemSize = 132 * 1024;
            }
            else if (size <= 164 * 1024) {
                *shMemSize = 164 * 1024;
            }
            else {
                return CUDA_OCC_ERROR_INVALID_INPUT;
            }
        }
        else {
            if      (size == 0) {
                *shMemSize = 0;
            }
            else if (size <= 8 * 1024) {
                *shMemSize = 8 * 1024;
            }
            else if (size <= 16 * 1024) {
                *shMemSize = 16 * 1024;
            }
            else if (size <= 32 * 1024) {
                *shMemSize = 32 * 1024;
            }
            else if (size <= 64 * 1024) {
                *shMemSize = 64 * 1024;
            }
            else if (size <= 100 * 1024) {
                *shMemSize = 100 * 1024;
            }
            else {
                return CUDA_OCC_ERROR_INVALID_INPUT;
            }
        }
        break;






































    default:
        return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    return CUDA_OCC_SUCCESS;
}

/**
 * Shared memory based on the new carveoutConfig API introduced with Volta
 */
static __OCC_INLINE cudaOccError cudaOccSMemPreferenceVoltaPlus(size_t *limit, const cudaOccDeviceProp *properties, const cudaOccDeviceState *state)
{
    cudaOccError status = CUDA_OCC_SUCCESS;
    size_t preferenceShmemSize;

    // CUDA 9.0 introduces a new API to set shared memory - L1 configuration on supported
    // devices. This preference will take precedence over the older cacheConfig setting.
    // Map cacheConfig to its effective preference value.
    int effectivePreference = state->carveoutConfig;
    if ((effectivePreference < SHAREDMEM_CARVEOUT_DEFAULT) || (effectivePreference > SHAREDMEM_CARVEOUT_MAX_SHARED)) {
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }
    
    if (effectivePreference == SHAREDMEM_CARVEOUT_DEFAULT) {
        switch (state->cacheConfig)
        {
        case CACHE_PREFER_L1:
            effectivePreference = SHAREDMEM_CARVEOUT_MAX_L1;
            break;
        case CACHE_PREFER_SHARED:
            effectivePreference = SHAREDMEM_CARVEOUT_MAX_SHARED;
            break;
        case CACHE_PREFER_EQUAL:
            effectivePreference = SHAREDMEM_CARVEOUT_HALF;
            break;
        default:
            effectivePreference = SHAREDMEM_CARVEOUT_DEFAULT;
            break;
        }
    }

    if (effectivePreference == SHAREDMEM_CARVEOUT_DEFAULT) {
        preferenceShmemSize = properties->sharedMemPerMultiprocessor;
    }
    else {
        preferenceShmemSize = (size_t) (effectivePreference * properties->sharedMemPerMultiprocessor) / 100;
    }

    status = cudaOccAlignUpShmemSizeVoltaPlus(&preferenceShmemSize, properties);
    *limit = preferenceShmemSize;
    return status;
}

/**
 * Shared memory based on the cacheConfig
 */
static __OCC_INLINE cudaOccError cudaOccSMemPreference(size_t *limit, const cudaOccDeviceProp *properties, const cudaOccDeviceState *state)
{
    size_t bytes                          = 0;
    size_t sharedMemPerMultiprocessorHigh = properties->sharedMemPerMultiprocessor;
    cudaOccCacheConfig cacheConfig        = state->cacheConfig;

    // Kepler has shared L1 cache / shared memory, and support cache
    // configuration to trade one for the other. These values are needed to
    // calculate the correct shared memory size for user requested cache
    // configuration.
    //
    size_t minCacheSize                   = 16384;
    size_t maxCacheSize                   = 49152;
    size_t cacheAndSharedTotal            = sharedMemPerMultiprocessorHigh + minCacheSize;
    size_t sharedMemPerMultiprocessorLow  = cacheAndSharedTotal - maxCacheSize;

    switch (properties->computeMajor) {
        case 3:
            // Kepler supports 16KB, 32KB, or 48KB partitions for L1. The rest
            // is shared memory.
            //
            switch (cacheConfig) {
                default :
                case CACHE_PREFER_NONE:
                case CACHE_PREFER_SHARED:
                    bytes = sharedMemPerMultiprocessorHigh;
                    break;
                case CACHE_PREFER_L1:
                    bytes = sharedMemPerMultiprocessorLow;
                    break;
                case CACHE_PREFER_EQUAL:
                    // Equal is the mid-point between high and low. It should be
                    // equivalent to low + 16KB.
                    //
                    bytes = (sharedMemPerMultiprocessorHigh + sharedMemPerMultiprocessorLow) / 2;
                    break;
            }
            break;
        case 5:
        case 6:
            // Maxwell and Pascal have dedicated shared memory.
            //
            bytes = sharedMemPerMultiprocessorHigh;
            break;
        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    *limit = bytes;

    return CUDA_OCC_SUCCESS;
}

/**
 * Shared memory based on config requested by User
 */
static __OCC_INLINE cudaOccError cudaOccSMemPerMultiprocessor(size_t *limit, const cudaOccDeviceProp *properties, const cudaOccDeviceState *state)
{
    // Volta introduces a new API that allows for shared memory carveout preference. Because it is a shared memory preference,
    // it is handled separately from the cache config preference.
    if (properties->computeMajor >= 7) {
        return cudaOccSMemPreferenceVoltaPlus(limit, properties, state);
    }
    return cudaOccSMemPreference(limit, properties, state);
}

/**
 * Return the per block shared memory limit based on function config
 */
static __OCC_INLINE cudaOccError cudaOccSMemPerBlock(size_t *limit, const cudaOccDeviceProp *properties, cudaOccFuncShmemConfig shmemLimitConfig, size_t smemPerCta)
{
    switch (properties->computeMajor) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
            *limit = properties->sharedMemPerBlock;
            break;
        case 7:
        case 8:



            switch (shmemLimitConfig) {
                default:
                case FUNC_SHMEM_LIMIT_DEFAULT:
                    *limit = properties->sharedMemPerBlock;
                    break;
                case FUNC_SHMEM_LIMIT_OPTIN:
                    if (smemPerCta > properties->sharedMemPerBlock) {
                        *limit = properties->sharedMemPerBlockOptin;
                    }
                    else {
                        *limit = properties->sharedMemPerBlock;
                    }
                    break;
            }
            break;
        default:
            return CUDA_OCC_ERROR_UNKNOWN_DEVICE;
    }

    // Starting Ampere, CUDA driver reserves additional shared memory per block
    if (properties->computeMajor >= 8) {
        *limit += properties->reservedSharedMemPerBlock;
    }

    return CUDA_OCC_SUCCESS;
}

/**
 * Partitioned global caching mode support
 */
static __OCC_INLINE cudaOccError cudaOccPartitionedGlobalCachingModeSupport(cudaOccPartitionedGCSupport *limit, const cudaOccDeviceProp *properties)
{
    *limit = PARTITIONED_GC_NOT_SUPPORTED;

    if ((properties->computeMajor == 5 && (properties->computeMinor == 2 || properties->computeMinor == 3)) ||
        properties->computeMajor == 6) {
        *limit = PARTITIONED_GC_SUPPORTED;
    }

    if (properties->computeMajor == 6 && properties->computeMinor == 0) {
        *limit = PARTITIONED_GC_NOT_SUPPORTED;
    }

    return CUDA_OCC_SUCCESS;
}

///////////////////////////////////////////////
//            User Input Sanity              //
///////////////////////////////////////////////

static __OCC_INLINE cudaOccError cudaOccDevicePropCheck(const cudaOccDeviceProp *properties)
{
    // Verify device properties
    //
    // Each of these limits must be a positive number.
    //
    // Compute capacity is checked during the occupancy calculation
    //
    if (properties->maxThreadsPerBlock          <= 0 ||
        properties->maxThreadsPerMultiprocessor <= 0 ||
        properties->regsPerBlock                <= 0 ||
        properties->regsPerMultiprocessor       <= 0 ||
        properties->warpSize                    <= 0 ||
        properties->sharedMemPerBlock           <= 0 ||
        properties->sharedMemPerMultiprocessor  <= 0 ||
        properties->numSms                      <= 0) {
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }

    return CUDA_OCC_SUCCESS;
}

static __OCC_INLINE cudaOccError cudaOccFuncAttributesCheck(const cudaOccFuncAttributes *attributes)
{
    // Verify function attributes
    //
    if (attributes->maxThreadsPerBlock <= 0 ||
        attributes->numRegs < 0) {            // Compiler may choose not to use
                                              // any register (empty kernels,
                                              // etc.)
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }

    return CUDA_OCC_SUCCESS;
}

static __OCC_INLINE cudaOccError cudaOccDeviceStateCheck(const cudaOccDeviceState *state)
{
    (void)state;   // silence unused-variable warning
    // Placeholder
    //

    return CUDA_OCC_SUCCESS;
}

static __OCC_INLINE cudaOccError cudaOccInputCheck(
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state)
{
    cudaOccError status = CUDA_OCC_SUCCESS;

    status = cudaOccDevicePropCheck(properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    status = cudaOccFuncAttributesCheck(attributes);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    status = cudaOccDeviceStateCheck(state);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    return status;
}

///////////////////////////////////////////////
//    Occupancy calculation Functions        //
///////////////////////////////////////////////

static __OCC_INLINE cudaOccPartitionedGCConfig cudaOccPartitionedGCExpected(
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes)
{
    cudaOccPartitionedGCSupport gcSupport;
    cudaOccPartitionedGCConfig gcConfig;

    cudaOccPartitionedGlobalCachingModeSupport(&gcSupport, properties);

    gcConfig = attributes->partitionedGCConfig;

    if (gcSupport == PARTITIONED_GC_NOT_SUPPORTED) {
        gcConfig = PARTITIONED_GC_OFF;
    }

    return gcConfig;
}

// Warp limit
//
static __OCC_INLINE cudaOccError cudaOccMaxBlocksPerSMWarpsLimit(
    int                         *limit,
    cudaOccPartitionedGCConfig   gcConfig,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    int                          blockSize)
{
    cudaOccError status = CUDA_OCC_SUCCESS;
    int maxWarpsPerSm;
    int warpsAllocatedPerCTA;
    int maxBlocks;
    (void)attributes;   // silence unused-variable warning

    if (blockSize > properties->maxThreadsPerBlock) {
        maxBlocks = 0;
    }
    else {
        maxWarpsPerSm = properties->maxThreadsPerMultiprocessor / properties->warpSize;
        warpsAllocatedPerCTA = __occDivideRoundUp(blockSize, properties->warpSize);
        maxBlocks = 0;

        if (gcConfig != PARTITIONED_GC_OFF) {
            int maxBlocksPerSmPartition;
            int maxWarpsPerSmPartition;

            // If partitioned global caching is on, then a CTA can only use a SM
            // partition (a half SM), and thus a half of the warp slots
            // available per SM
            //
            maxWarpsPerSmPartition  = maxWarpsPerSm / 2;
            maxBlocksPerSmPartition = maxWarpsPerSmPartition / warpsAllocatedPerCTA;
            maxBlocks               = maxBlocksPerSmPartition * 2;
        }
        // On hardware that supports partitioned global caching, each half SM is
        // guaranteed to support at least 32 warps (maximum number of warps of a
        // CTA), so caching will not cause 0 occupancy due to insufficient warp
        // allocation slots.
        //
        else {
            maxBlocks = maxWarpsPerSm / warpsAllocatedPerCTA;
        }
    }

    *limit = maxBlocks;

    return status;
}

// Shared memory limit
//
static __OCC_INLINE cudaOccError cudaOccMaxBlocksPerSMSmemLimit(
    int                         *limit,
    cudaOccResult               *result,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    int                          blockSize,
    size_t                       dynamicSmemSize)
{
    cudaOccError status = CUDA_OCC_SUCCESS;
    int allocationGranularity;
    size_t userSmemPreference = 0;
    size_t totalSmemUsagePerCTA;
    size_t maxSmemUsagePerCTA;
    size_t smemAllocatedPerCTA;
    size_t staticSmemSize;
    size_t sharedMemPerMultiprocessor;
    size_t smemLimitPerCTA;
    int maxBlocks;
    int dynamicSmemSizeExceeded = 0;
    int totalSmemSizeExceeded = 0;
    (void)blockSize;   // silence unused-variable warning

    status = cudaOccSMemAllocationGranularity(&allocationGranularity, properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    // Obtain the user preferred shared memory size. This setting is ignored if
    // user requests more shared memory than preferred.
    //
    status = cudaOccSMemPerMultiprocessor(&userSmemPreference, properties, state);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    staticSmemSize = attributes->sharedSizeBytes + properties->reservedSharedMemPerBlock;
    totalSmemUsagePerCTA = staticSmemSize + dynamicSmemSize;
    smemAllocatedPerCTA = __occRoundUp((int)totalSmemUsagePerCTA, (int)allocationGranularity);

    maxSmemUsagePerCTA = staticSmemSize + attributes->maxDynamicSharedSizeBytes;

    dynamicSmemSizeExceeded = 0;
    totalSmemSizeExceeded   = 0;

    // Obtain the user set maximum dynamic size if it exists
    // If so, the current launch dynamic shared memory must not
    // exceed the set limit
    if (attributes->shmemLimitConfig != FUNC_SHMEM_LIMIT_DEFAULT &&
        dynamicSmemSize > attributes->maxDynamicSharedSizeBytes) {
        dynamicSmemSizeExceeded = 1;
    }

    status = cudaOccSMemPerBlock(&smemLimitPerCTA, properties, attributes->shmemLimitConfig, maxSmemUsagePerCTA);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    if (smemAllocatedPerCTA > smemLimitPerCTA) {
        totalSmemSizeExceeded = 1;
    }

    if (dynamicSmemSizeExceeded || totalSmemSizeExceeded) {
        maxBlocks = 0;
    }
    else {
        // User requested shared memory limit is used as long as it is greater
        // than the total shared memory used per CTA, i.e. as long as at least
        // one CTA can be launched.
        if (userSmemPreference >= smemAllocatedPerCTA) {
            sharedMemPerMultiprocessor = userSmemPreference;
        }
        else {
            // On Volta+, user requested shared memory will limit occupancy
            // if it's less than shared memory per CTA. Otherwise, the
            // maximum shared memory limit is used.
            if (properties->computeMajor >= 7) {
                sharedMemPerMultiprocessor = smemAllocatedPerCTA;
                status = cudaOccAlignUpShmemSizeVoltaPlus(&sharedMemPerMultiprocessor, properties);
                if (status != CUDA_OCC_SUCCESS) {
                    return status;
                }
            }
            else {
                sharedMemPerMultiprocessor = properties->sharedMemPerMultiprocessor;
            }
        }

        if (smemAllocatedPerCTA > 0) {
            maxBlocks = (int)(sharedMemPerMultiprocessor / smemAllocatedPerCTA);
        }
        else {
            maxBlocks = INT_MAX;
        }
    }

    result->allocatedSharedMemPerBlock = smemAllocatedPerCTA;

    *limit = maxBlocks;

    return status;
}

static __OCC_INLINE
cudaOccError cudaOccMaxBlocksPerSMRegsLimit(
    int                         *limit,
    cudaOccPartitionedGCConfig  *gcConfig,
    cudaOccResult               *result,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    int                          blockSize)
{
    cudaOccError status = CUDA_OCC_SUCCESS;
    int allocationGranularity;
    int warpsAllocatedPerCTA;
    int regsAllocatedPerCTA;
    int regsAssumedPerCTA;
    int regsPerWarp;
    int regsAllocatedPerWarp;
    int numSubPartitions;
    int numRegsPerSubPartition;
    int numWarpsPerSubPartition;
    int numWarpsPerSM;
    int maxBlocks;
    int maxRegsPerThread;

    status = cudaOccRegAllocationGranularity(
        &allocationGranularity,
        properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    status = cudaOccRegAllocationMaxPerThread(
        &maxRegsPerThread,
        properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    status = cudaOccSubPartitionsPerMultiprocessor(&numSubPartitions, properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    warpsAllocatedPerCTA = __occDivideRoundUp(blockSize, properties->warpSize);

    // GPUs of compute capability 2.x and higher allocate registers to warps
    //
    // Number of regs per warp is regs per thread x warp size, rounded up to
    // register allocation granularity
    //
    regsPerWarp          = attributes->numRegs * properties->warpSize;
    regsAllocatedPerWarp = __occRoundUp(regsPerWarp, allocationGranularity);
    regsAllocatedPerCTA  = regsAllocatedPerWarp * warpsAllocatedPerCTA;

    // Hardware verifies if a launch fits the per-CTA register limit. For
    // historical reasons, the verification logic assumes register
    // allocations are made to all partitions simultaneously. Therefore, to
    // simulate the hardware check, the warp allocation needs to be rounded
    // up to the number of partitions.
    //
    regsAssumedPerCTA = regsAllocatedPerWarp * __occRoundUp(warpsAllocatedPerCTA, numSubPartitions);

    if (properties->regsPerBlock < regsAssumedPerCTA ||   // Hardware check
        properties->regsPerBlock < regsAllocatedPerCTA || // Software check
        attributes->numRegs > maxRegsPerThread) {         // Per thread limit check
        maxBlocks = 0;
    }
    else {
        if (regsAllocatedPerWarp > 0) {
            // Registers are allocated in each sub-partition. The max number
            // of warps that can fit on an SM is equal to the max number of
            // warps per sub-partition x number of sub-partitions.
            //
            numRegsPerSubPartition  = properties->regsPerMultiprocessor / numSubPartitions;
            numWarpsPerSubPartition = numRegsPerSubPartition / regsAllocatedPerWarp;

            maxBlocks = 0;

            if (*gcConfig != PARTITIONED_GC_OFF) {
                int numSubPartitionsPerSmPartition;
                int numWarpsPerSmPartition;
                int maxBlocksPerSmPartition;

                // If partitioned global caching is on, then a CTA can only
                // use a half SM, and thus a half of the registers available
                // per SM
                //
                numSubPartitionsPerSmPartition = numSubPartitions / 2;
                numWarpsPerSmPartition         = numWarpsPerSubPartition * numSubPartitionsPerSmPartition;
                maxBlocksPerSmPartition        = numWarpsPerSmPartition / warpsAllocatedPerCTA;
                maxBlocks                      = maxBlocksPerSmPartition * 2;
            }

            // Try again if partitioned global caching is not enabled, or if
            // the CTA cannot fit on the SM with caching on (maxBlocks == 0).  In the latter
            // case, the device will automatically turn off caching, except
            // if the user forces enablement via PARTITIONED_GC_ON_STRICT to calculate
            // occupancy and launch configuration.
            //
            if (maxBlocks == 0 && *gcConfig != PARTITIONED_GC_ON_STRICT) {
               // In case *gcConfig was PARTITIONED_GC_ON flip it OFF since
               // this is what it will be if we spread CTA across partitions.
               //
               *gcConfig = PARTITIONED_GC_OFF;
               numWarpsPerSM = numWarpsPerSubPartition * numSubPartitions;
               maxBlocks     = numWarpsPerSM / warpsAllocatedPerCTA;
            }
        }
        else {
            maxBlocks = INT_MAX;
        }
    }


    result->allocatedRegistersPerBlock = regsAllocatedPerCTA;

    *limit = maxBlocks;

    return status;
}

///////////////////////////////////
//      API Implementations      //
///////////////////////////////////

static __OCC_INLINE
cudaOccError cudaOccMaxActiveBlocksPerMultiprocessor(
    cudaOccResult               *result,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    int                          blockSize,
    size_t                       dynamicSmemSize)
{
    cudaOccError status          = CUDA_OCC_SUCCESS;
    int          ctaLimitWarps   = 0;
    int          ctaLimitBlocks  = 0;
    int          ctaLimitSMem    = 0;
    int          ctaLimitRegs    = 0;
    int          ctaLimit        = 0;
    unsigned int limitingFactors = 0;
    
    cudaOccPartitionedGCConfig gcConfig = PARTITIONED_GC_OFF;

    if (!result || !properties || !attributes || !state || blockSize <= 0) {
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }

    ///////////////////////////
    // Check user input
    ///////////////////////////

    status = cudaOccInputCheck(properties, attributes, state);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    ///////////////////////////
    // Initialization
    ///////////////////////////

    gcConfig = cudaOccPartitionedGCExpected(properties, attributes);

    ///////////////////////////
    // Compute occupancy
    ///////////////////////////

    // Limits due to registers/SM
    // Also compute if partitioned global caching has to be turned off
    //
    status = cudaOccMaxBlocksPerSMRegsLimit(&ctaLimitRegs, &gcConfig, result, properties, attributes, blockSize);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    // SMs on GP100 (6.0) have 2 subpartitions, while those on GP10x have 4.
    // As a result, an SM on GP100 may be able to run more CTAs than the one on GP10x.
    // For forward compatibility within Pascal family, if a function cannot run on GP10x (maxBlock == 0),
    // we do not let it run on any Pascal processor, even though it may be able to run on GP100.
    // Therefore, we check the occupancy on GP10x when it can run on GP100
    //
    if (properties->computeMajor == 6 && properties->computeMinor == 0 && ctaLimitRegs) {
        cudaOccDeviceProp propertiesGP10x;
        cudaOccPartitionedGCConfig gcConfigGP10x = gcConfig;
        int ctaLimitRegsGP10x = 0;

        // Set up properties for GP10x
        memcpy(&propertiesGP10x, properties, sizeof(propertiesGP10x));
        propertiesGP10x.computeMinor = 1;

        status = cudaOccMaxBlocksPerSMRegsLimit(&ctaLimitRegsGP10x, &gcConfigGP10x, result, &propertiesGP10x, attributes, blockSize);
        if (status != CUDA_OCC_SUCCESS) {
            return status;
        }

        if (ctaLimitRegsGP10x == 0) {
            ctaLimitRegs = 0;
        }
    }

    // Limits due to warps/SM
    //
    status = cudaOccMaxBlocksPerSMWarpsLimit(&ctaLimitWarps, gcConfig, properties, attributes, blockSize);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    // Limits due to blocks/SM
    //
    status = cudaOccMaxBlocksPerMultiprocessor(&ctaLimitBlocks, properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    // Limits due to shared memory/SM
    //
    status = cudaOccMaxBlocksPerSMSmemLimit(&ctaLimitSMem, result, properties, attributes, state, blockSize, dynamicSmemSize);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    ///////////////////////////
    // Overall occupancy
    ///////////////////////////

    // Overall limit is min() of limits due to above reasons
    //
    ctaLimit = __occMin(ctaLimitRegs, __occMin(ctaLimitSMem, __occMin(ctaLimitWarps, ctaLimitBlocks)));

    // Fill in the return values
    //
    // Determine occupancy limiting factors
    //
    if (ctaLimit == ctaLimitWarps) {
        limitingFactors |= OCC_LIMIT_WARPS;
    }
    if (ctaLimit == ctaLimitRegs) {
        limitingFactors |= OCC_LIMIT_REGISTERS;
    }
    if (ctaLimit == ctaLimitSMem) {
        limitingFactors |= OCC_LIMIT_SHARED_MEMORY;
    }
    if (ctaLimit == ctaLimitBlocks) {
        limitingFactors |= OCC_LIMIT_BLOCKS;
    }
    result->limitingFactors = limitingFactors;

    result->blockLimitRegs      = ctaLimitRegs;
    result->blockLimitSharedMem = ctaLimitSMem;
    result->blockLimitWarps     = ctaLimitWarps;
    result->blockLimitBlocks    = ctaLimitBlocks;
    result->partitionedGCConfig = gcConfig;

    // Final occupancy
    result->activeBlocksPerMultiprocessor = ctaLimit;

    return CUDA_OCC_SUCCESS;
}

static __OCC_INLINE
cudaOccError cudaOccAvailableDynamicSMemPerBlock(
    size_t                      *bytesAvailable,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    int                         numBlocks,
    int                         blockSize)
{
    int allocationGranularity;
    size_t smemLimitPerBlock;
    size_t smemAvailableForDynamic;
    size_t userSmemPreference = 0;
    size_t sharedMemPerMultiprocessor;
    cudaOccResult result;
    cudaOccError status = CUDA_OCC_SUCCESS;

    if (numBlocks <= 0)
        return CUDA_OCC_ERROR_INVALID_INPUT;

    // First compute occupancy of potential kernel launch.
    //
    status = cudaOccMaxActiveBlocksPerMultiprocessor(&result, properties, attributes, state, blockSize, 0);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }
    // Check if occupancy is achievable given user requested number of blocks. 
    //
    if (result.activeBlocksPerMultiprocessor < numBlocks) {
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }

    status = cudaOccSMemAllocationGranularity(&allocationGranularity, properties);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    // Return the per block shared memory limit based on function config.
    //
    status = cudaOccSMemPerBlock(&smemLimitPerBlock, properties, attributes->shmemLimitConfig, properties->sharedMemPerMultiprocessor);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    // If there is only a single block needed per SM, then the user preference can be ignored and the fully SW
    // limit is allowed to be used as shared memory otherwise if more than one block is needed, then the user
    // preference sets the total limit of available shared memory.
    //
    cudaOccSMemPerMultiprocessor(&userSmemPreference, properties, state);
    if (numBlocks == 1) {
        sharedMemPerMultiprocessor = smemLimitPerBlock;
    }
    else {
        if (!userSmemPreference) {
            userSmemPreference = 1 ;
            status = cudaOccAlignUpShmemSizeVoltaPlus(&userSmemPreference, properties);
            if (status != CUDA_OCC_SUCCESS) {
                return status;
            }
        }
        sharedMemPerMultiprocessor = userSmemPreference;
    }

    // Compute total shared memory available per SM
    //
    smemAvailableForDynamic =  sharedMemPerMultiprocessor / numBlocks;
    smemAvailableForDynamic = (smemAvailableForDynamic / allocationGranularity) * allocationGranularity;

    // Cap shared memory
    //
    if (smemAvailableForDynamic > smemLimitPerBlock) {
        smemAvailableForDynamic = smemLimitPerBlock;
    }

    // Now compute dynamic shared memory size
    smemAvailableForDynamic = smemAvailableForDynamic - attributes->sharedSizeBytes; 

    // Cap computed dynamic SM by user requested limit specified via cuFuncSetAttribute()
    //
    if (smemAvailableForDynamic > attributes->maxDynamicSharedSizeBytes)
        smemAvailableForDynamic = attributes->maxDynamicSharedSizeBytes;

    *bytesAvailable = smemAvailableForDynamic;
    return CUDA_OCC_SUCCESS;
}

static __OCC_INLINE
cudaOccError cudaOccMaxPotentialOccupancyBlockSize(
    int                         *minGridSize,
    int                         *blockSize,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    size_t                     (*blockSizeToDynamicSMemSize)(int),
    size_t                       dynamicSMemSize)
{
    cudaOccError  status = CUDA_OCC_SUCCESS;
    cudaOccResult result;

    // Limits
    int occupancyLimit;
    int granularity;
    int blockSizeLimit;

    // Recorded maximum
    int maxBlockSize = 0;
    int numBlocks    = 0;
    int maxOccupancy = 0;

    // Temporary
    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int occupancyInBlocks;
    int occupancyInThreads;

    ///////////////////////////
    // Check user input
    ///////////////////////////

    if (!minGridSize || !blockSize || !properties || !attributes || !state) {
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }

    status = cudaOccInputCheck(properties, attributes, state);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Try each block size, and pick the block size with maximum occupancy
    /////////////////////////////////////////////////////////////////////////////////

    occupancyLimit = properties->maxThreadsPerMultiprocessor;
    granularity    = properties->warpSize;

    blockSizeLimit        = __occMin(properties->maxThreadsPerBlock, attributes->maxThreadsPerBlock);
    blockSizeLimitAligned = __occRoundUp(blockSizeLimit, granularity);

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
        blockSizeToTry = __occMin(blockSizeLimit, blockSizeToTryAligned);

        // Ignore dynamicSMemSize if the user provides a mapping
        //
        if (blockSizeToDynamicSMemSize) {
            dynamicSMemSize = (*blockSizeToDynamicSMemSize)(blockSizeToTry);
        }

        status = cudaOccMaxActiveBlocksPerMultiprocessor(
            &result,
            properties,
            attributes,
            state,
            blockSizeToTry,
            dynamicSMemSize);

        if (status != CUDA_OCC_SUCCESS) {
            return status;
        }

        occupancyInBlocks = result.activeBlocksPerMultiprocessor;
        occupancyInThreads = blockSizeToTry * occupancyInBlocks;

        if (occupancyInThreads > maxOccupancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks    = occupancyInBlocks;
            maxOccupancy = occupancyInThreads;
        }

        // Early out if we have reached the maximum
        //
        if (occupancyLimit == maxOccupancy) {
            break;
        }
    }

    ///////////////////////////
    // Return best available
    ///////////////////////////

    // Suggested min grid size to achieve a full machine launch
    //
    *minGridSize = numBlocks * properties->numSms;
    *blockSize = maxBlockSize;

    return status;
}


#if defined(__cplusplus)

namespace {

__OCC_INLINE
cudaOccError cudaOccMaxPotentialOccupancyBlockSize(
    int                         *minGridSize,
    int                         *blockSize,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    size_t                       dynamicSMemSize)
{
    return cudaOccMaxPotentialOccupancyBlockSize(
        minGridSize,
        blockSize,
        properties,
        attributes,
        state,
        NULL,
        dynamicSMemSize);
}

template <typename UnaryFunction>
__OCC_INLINE
cudaOccError cudaOccMaxPotentialOccupancyBlockSizeVariableSMem(
    int                         *minGridSize,
    int                         *blockSize,
    const cudaOccDeviceProp     *properties,
    const cudaOccFuncAttributes *attributes,
    const cudaOccDeviceState    *state,
    UnaryFunction                blockSizeToDynamicSMemSize)
{
    cudaOccError  status = CUDA_OCC_SUCCESS;
    cudaOccResult result;

    // Limits
    int occupancyLimit;
    int granularity;
    int blockSizeLimit;

    // Recorded maximum
    int maxBlockSize = 0;
    int numBlocks    = 0;
    int maxOccupancy = 0;

    // Temporary
    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int occupancyInBlocks;
    int occupancyInThreads;
    size_t dynamicSMemSize;

    ///////////////////////////
    // Check user input
    ///////////////////////////

    if (!minGridSize || !blockSize || !properties || !attributes || !state) {
        return CUDA_OCC_ERROR_INVALID_INPUT;
    }

    status = cudaOccInputCheck(properties, attributes, state);
    if (status != CUDA_OCC_SUCCESS) {
        return status;
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Try each block size, and pick the block size with maximum occupancy
    /////////////////////////////////////////////////////////////////////////////////

    occupancyLimit = properties->maxThreadsPerMultiprocessor;
    granularity    = properties->warpSize;
    blockSizeLimit        = __occMin(properties->maxThreadsPerBlock, attributes->maxThreadsPerBlock);
    blockSizeLimitAligned = __occRoundUp(blockSizeLimit, granularity);

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
        blockSizeToTry = __occMin(blockSizeLimit, blockSizeToTryAligned);

        dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);

        status = cudaOccMaxActiveBlocksPerMultiprocessor(
            &result,
            properties,
            attributes,
            state,
            blockSizeToTry,
            dynamicSMemSize);

        if (status != CUDA_OCC_SUCCESS) {
            return status;
        }

        occupancyInBlocks = result.activeBlocksPerMultiprocessor;

        occupancyInThreads = blockSizeToTry * occupancyInBlocks;

        if (occupancyInThreads > maxOccupancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks    = occupancyInBlocks;
            maxOccupancy = occupancyInThreads;
        }

        // Early out if we have reached the maximum
        //
        if (occupancyLimit == maxOccupancy) {
            break;
        }
    }

    ///////////////////////////
    // Return best available
    ///////////////////////////

    // Suggested min grid size to achieve a full machine launch
    //
    *minGridSize = numBlocks * properties->numSms;
    *blockSize = maxBlockSize;

    return status;
}

} // namespace anonymous

#endif /*__cplusplus */

#undef __OCC_INLINE

#endif /*__cuda_occupancy_h__*/
