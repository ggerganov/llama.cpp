/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
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
#pragma message("crt/device_functions.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/device_functions.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_FUNCTIONS_HPP__
#endif

#if !defined(__DEVICE_FUNCTIONS_HPP__)
#define __DEVICE_FUNCTIONS_HPP__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

#if defined(__CUDACC_RTC__)
#define __DEVICE_FUNCTIONS_DECL__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ __device__
#else
#define __DEVICE_FUNCTIONS_DECL__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_FUNCTIONS_STATIC_DECL__ int mulhi(const int a, const int b)
{
  return __mulhi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(const unsigned int a, const unsigned int b)
{
  return __umulhi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(const int a, const unsigned int b)
{
  return __umulhi(static_cast<unsigned int>(a), b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int mulhi(const unsigned int a, const int b)
{
  return __umulhi(a, static_cast<unsigned int>(b));
}

__DEVICE_FUNCTIONS_STATIC_DECL__ long long int mul64hi(const long long int a, const long long int b)
{
  return __mul64hi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(const unsigned long long int a, const unsigned long long int b)
{
  return __umul64hi(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(const long long int a, const unsigned long long int b)
{
  return __umul64hi(static_cast<unsigned long long int>(a), b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned long long int mul64hi(const unsigned long long int a, const long long int b)
{
  return __umul64hi(a, static_cast<unsigned long long int>(b));
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int float_as_int(const float a)
{
  return __float_as_int(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float int_as_float(const int a)
{
  return __int_as_float(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int float_as_uint(const float a)
{
  return __float_as_uint(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float uint_as_float(const unsigned int a)
{
  return __uint_as_float(a);
}
__DEVICE_FUNCTIONS_STATIC_DECL__ float saturate(const float a)
{
  return __saturatef(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int mul24(const int a, const int b)
{
  return __mul24(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int umul24(const unsigned int a, const unsigned int b)
{
  return __umul24(a, b);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ int float2int(const float a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundNearest) ? __float2int_rn(a) :
         (mode == cudaRoundPosInf ) ? __float2int_ru(a) :
         (mode == cudaRoundMinInf ) ? __float2int_rd(a) :
                                      __float2int_rz(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ unsigned int float2uint(const float a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundNearest) ? __float2uint_rn(a) :
         (mode == cudaRoundPosInf ) ? __float2uint_ru(a) :
         (mode == cudaRoundMinInf ) ? __float2uint_rd(a) :
                                      __float2uint_rz(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float int2float(const int a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundZero  ) ? __int2float_rz(a) :
         (mode == cudaRoundPosInf) ? __int2float_ru(a) :
         (mode == cudaRoundMinInf) ? __int2float_rd(a) :
                                     __int2float_rn(a);
}

__DEVICE_FUNCTIONS_STATIC_DECL__ float uint2float(const unsigned int a, const enum cudaRoundMode mode)
{
  return (mode == cudaRoundZero  ) ? __uint2float_rz(a) :
         (mode == cudaRoundPosInf) ? __uint2float_ru(a) :
         (mode == cudaRoundMinInf) ? __uint2float_rd(a) :
                                     __uint2float_rn(a);
}

#undef __DEVICE_FUNCTIONS_DECL__
#undef __DEVICE_FUNCTIONS_STATIC_DECL__

#endif /* __cplusplus && __CUDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* !__DEVICE_FUNCTIONS_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_FUNCTIONS_HPP__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_FUNCTIONS_HPP__
#endif
