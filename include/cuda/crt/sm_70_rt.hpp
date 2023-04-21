/*
 * Copyright 2017-2021 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/sm_70_rt.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/sm_70_rt.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_70_RT_HPP__
#endif

#if !defined(__SM_70_RT_HPP__)
#define __SM_70_RT_HPP__

#if defined(__CUDACC_RTC__)
#define __SM_70_RT_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_70_RT_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*  Below are implementations of SM-7.0 builtin functions which are included as *
*  source (instead of being built in to the compiler)                          *
*                                                                              *
*******************************************************************************/

//
// __match_any_sync
//
__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, unsigned value) {
  return __match32_any_sync(mask, value);
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, int value) {
  return __match32_any_sync(mask, value);
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, unsigned long value) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_any_sync(mask, (unsigned long long)value):
    __match32_any_sync(mask, (unsigned)value);
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, long value) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_any_sync(mask, (unsigned long long)value):
    __match32_any_sync(mask, (unsigned)value);
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, unsigned long long value) {
  return __match64_any_sync(mask, value);
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, long long value) {
  return __match64_any_sync(mask, value);
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, float value) {
  return __match32_any_sync(mask, __float_as_uint(value));
}

__SM_70_RT_DECL__ unsigned int __match_any_sync(unsigned mask, double value) {
  return __match64_any_sync(mask, __double_as_longlong(value));
}

//
// __match_all_sync
//
__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, unsigned value, int *pred) {
  return __match32_all_sync(mask, value, pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, int value, int *pred) {
  return __match32_all_sync(mask, value, pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, unsigned long value, int *pred) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_all_sync(mask, (unsigned long long)value, pred):
    __match32_all_sync(mask, (unsigned)value, pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, long value, int *pred) {
  return (sizeof(long) == sizeof(long long)) ?
    __match64_all_sync(mask, (unsigned long long)value, pred):
    __match32_all_sync(mask, (unsigned)value, pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, unsigned long long value, int *pred) {
  return __match64_all_sync(mask, value, pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, long long value, int *pred) {
  return __match64_all_sync(mask, value, pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, float value, int *pred) {
  return __match32_all_sync(mask, __float_as_uint(value), pred);
}

__SM_70_RT_DECL__ unsigned int __match_all_sync(unsigned mask, double value, int *pred) {
  return __match64_all_sync(mask, __double_as_longlong(value), pred);
}

__SM_70_RT_DECL__ void __nanosleep(unsigned int ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}


extern "C" __device__ __device_builtin__
unsigned short __usAtomicCAS(unsigned short *, unsigned short, unsigned short);

__SM_70_RT_DECL__ unsigned short int atomicCAS(unsigned short int *address, unsigned short int compare, unsigned short int val) {
  return __usAtomicCAS(address, compare, val);
}


#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_70_RT_DECL__

#endif /* !__SM_70_RT_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_70_RT_HPP__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_70_RT_HPP__
#endif
