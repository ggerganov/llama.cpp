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

#ifndef cuda_profiler_H
#define cuda_profiler_H

#include <cuda.h>

#if defined(__CUDA_API_VERSION_INTERNAL) || defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Profiler Output Modes
 */
/*DEVICE_BUILTIN*/
typedef enum CUoutput_mode_enum
{
    CU_OUT_KEY_VALUE_PAIR  = 0x00, /**< Output mode Key-Value pair format. */
    CU_OUT_CSV             = 0x01  /**< Output mode Comma separated values format. */
}CUoutput_mode;


/**
 * \ingroup CUDA_DRIVER
 * \defgroup CUDA_PROFILER_DEPRECATED Profiler Control [DEPRECATED]
 *
 * ___MANBRIEF___ profiler control functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the profiler control functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Initialize the profiling.
 *
 * \deprecated
 *
 * Using this API user can initialize the CUDA profiler by specifying
 * the configuration file, output file and output file format. This
 * API is generally used to profile different set of counters by
 * looping the kernel launch. The \p configFile parameter can be used
 * to select profiling options including profiler counters. Refer to
 * the "Compute Command Line Profiler User Guide" for supported
 * profiler options and counters.
 *
 * Limitation: The CUDA profiler cannot be initialized with this API
 * if another profiling tool is already active, as indicated by the
 * ::CUDA_ERROR_PROFILER_DISABLED return code.
 *
 * Typical usage of the profiling APIs is as follows: 
 *
 * for each set of counters/options\n
 * {\n
 *     cuProfilerInitialize(); //Initialize profiling, set the counters or options in the config file \n
 *     ...\n
 *     cuProfilerStart(); \n
 *     // code to be profiled \n
 *     cuProfilerStop(); \n
 *     ...\n
 *     cuProfilerStart(); \n
 *     // code to be profiled \n
 *     cuProfilerStop(); \n
 *     ...\n
 * }\n
 *
 * \param configFile - Name of the config file that lists the counters/options
 * for profiling.
 * \param outputFile - Name of the outputFile where the profiling results will
 * be stored.
 * \param outputMode - outputMode, can be ::CU_OUT_KEY_VALUE_PAIR or ::CU_OUT_CSV.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_PROFILER_DISABLED
 * \notefnerr
 *
 * \sa
 * ::cuProfilerStart,
 * ::cuProfilerStop,
 * ::cudaProfilerInitialize
 */
__CUDA_DEPRECATED CUresult CUDAAPI cuProfilerInitialize(const char *configFile, const char *outputFile, CUoutput_mode outputMode);
 
/** @} */ /* END CUDA_PROFILER_DEPRECATED */

/**
 * \ingroup CUDA_DRIVER
 * \defgroup CUDA_PROFILER Profiler Control 
 *
 * ___MANBRIEF___ profiler control functions of the low-level CUDA driver API
 * (___CURRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the profiler control functions of the low-level CUDA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Enable profiling.
 *
 * Enables profile collection by the active profiling tool for the
 * current context. If profiling is already enabled, then
 * cuProfilerStart() has no effect.
 *
 * cuProfilerStart and cuProfilerStop APIs are used to
 * programmatically control the profiling granularity by allowing
 * profiling to be done only on selective pieces of code.
 * 
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::cuProfilerInitialize,
 * ::cuProfilerStop,
 * ::cudaProfilerStart
 */
CUresult CUDAAPI cuProfilerStart(void);

/**
 * \brief Disable profiling.
 *
 * Disables profile collection by the active profiling tool for the
 * current context. If profiling is already disabled, then
 * cuProfilerStop() has no effect.
 *
 * cuProfilerStart and cuProfilerStop APIs are used to
 * programmatically control the profiling granularity by allowing
 * profiling to be done only on selective pieces of code.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_INVALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::cuProfilerInitialize,
 * ::cuProfilerStart,
 * ::cudaProfilerStop
 */
CUresult CUDAAPI cuProfilerStop(void);

/** @} */ /* END CUDA_PROFILER */

#ifdef __cplusplus
};
#endif

#undef __CUDA_DEPRECATED

#endif

