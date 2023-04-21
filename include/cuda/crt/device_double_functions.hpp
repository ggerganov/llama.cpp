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

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/device_double_functions.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/device_double_functions.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_DOUBLE_FUNCTIONS_HPP__
#endif

#if !defined(__DEVICE_DOUBLE_FUNCTIONS_HPP__)
#define __DEVICE_DOUBLE_FUNCTIONS_HPP__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__)
#define __DEVICE_DOUBLE_FUNCTIONS_DECL__ __device__
#else
#define __DEVICE_DOUBLE_FUNCTIONS_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double fma(double a, double b, double c, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __fma_rz(a, b, c) :
         mode == cudaRoundPosInf ? __fma_ru(a, b, c) :
         mode == cudaRoundMinInf ? __fma_rd(a, b, c) :
                                   __fma_rn(a, b, c);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double dmul(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __dmul_rz(a, b) :
         mode == cudaRoundPosInf ? __dmul_ru(a, b) :
         mode == cudaRoundMinInf ? __dmul_rd(a, b) :
                                   __dmul_rn(a, b);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double dadd(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __dadd_rz(a, b) :
         mode == cudaRoundPosInf ? __dadd_ru(a, b) :
         mode == cudaRoundMinInf ? __dadd_rd(a, b) :
                                   __dadd_rn(a, b);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double dsub(double a, double b, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __dsub_rz(a, b) :
         mode == cudaRoundPosInf ? __dsub_ru(a, b) :
         mode == cudaRoundMinInf ? __dsub_rd(a, b) :
                                   __dsub_rn(a, b);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ int double2int(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2int_rn(a) :
         mode == cudaRoundPosInf  ? __double2int_ru(a) :
         mode == cudaRoundMinInf  ? __double2int_rd(a) :
                                    __double2int_rz(a);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ unsigned int double2uint(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2uint_rn(a) :
         mode == cudaRoundPosInf  ? __double2uint_ru(a) :
         mode == cudaRoundMinInf  ? __double2uint_rd(a) :
                                    __double2uint_rz(a);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ long long int double2ll(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2ll_rn(a) :
         mode == cudaRoundPosInf  ? __double2ll_ru(a) :
         mode == cudaRoundMinInf  ? __double2ll_rd(a) :
                                    __double2ll_rz(a);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ unsigned long long int double2ull(double a, enum cudaRoundMode mode)
{
  return mode == cudaRoundNearest ? __double2ull_rn(a) :
         mode == cudaRoundPosInf  ? __double2ull_ru(a) :
         mode == cudaRoundMinInf  ? __double2ull_rd(a) :
                                    __double2ull_rz(a);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double ll2double(long long int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __ll2double_rz(a) :
         mode == cudaRoundPosInf ? __ll2double_ru(a) :
         mode == cudaRoundMinInf ? __ll2double_rd(a) :
                                   __ll2double_rn(a);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double ull2double(unsigned long long int a, enum cudaRoundMode mode)
{
  return mode == cudaRoundZero   ? __ull2double_rz(a) :
         mode == cudaRoundPosInf ? __ull2double_ru(a) :
         mode == cudaRoundMinInf ? __ull2double_rd(a) :
                                   __ull2double_rn(a);
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double int2double(int a, enum cudaRoundMode mode)
{
  return (double)a;
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double uint2double(unsigned int a, enum cudaRoundMode mode)
{
  return (double)a;
}

__DEVICE_DOUBLE_FUNCTIONS_DECL__ double float2double(float a, enum cudaRoundMode mode)
{
  return (double)a;
}

#undef __DEVICE_DOUBLE_FUNCTIONS_DECL__

#endif /* __cplusplus && __CUDACC__ */

#endif /* !__DEVICE_DOUBLE_FUNCTIONS_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_DOUBLE_FUNCTIONS_HPP__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_DOUBLE_FUNCTIONS_HPP__
#endif
