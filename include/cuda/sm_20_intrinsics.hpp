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

#if !defined(__SM_20_INTRINSICS_HPP__)
#define __SM_20_INTRINSICS_HPP__

#if defined(__CUDACC_RTC__)
#define __SM_20_INTRINSICS_DECL__ __device__
#else /* __CUDACC_RTC__ */
#define __SM_20_INTRINSICS_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SM_20_INTRINSICS_DECL__ unsigned int ballot(bool pred)
{
  return __ballot((int)pred);
}

__SM_20_INTRINSICS_DECL__ int syncthreads_count(bool pred)
{
  return __syncthreads_count((int)pred);
}

__SM_20_INTRINSICS_DECL__ bool syncthreads_and(bool pred)
{
  return (bool)__syncthreads_and((int)pred);
}

__SM_20_INTRINSICS_DECL__ bool syncthreads_or(bool pred)
{
  return (bool)__syncthreads_or((int)pred);
}


extern "C" {
  __device__ unsigned __nv_isGlobal_impl(const void *);
  __device__ unsigned __nv_isShared_impl(const void *);
  __device__ unsigned __nv_isConstant_impl(const void *);
  __device__ unsigned __nv_isLocal_impl(const void *);
}

__SM_20_INTRINSICS_DECL__ unsigned int __isGlobal(const void *ptr)
{
  return __nv_isGlobal_impl(ptr); 
}

__SM_20_INTRINSICS_DECL__ unsigned int __isShared(const void *ptr)
{
  return __nv_isShared_impl(ptr); 
}

__SM_20_INTRINSICS_DECL__ unsigned int __isConstant(const void *ptr)
{
  return __nv_isConstant_impl(ptr); 
}

__SM_20_INTRINSICS_DECL__ unsigned int __isLocal(const void *ptr)
{
  return __nv_isLocal_impl(ptr); 
}

extern "C" {
  __device__ size_t __nv_cvta_generic_to_global_impl(const void *);
  __device__ size_t __nv_cvta_generic_to_shared_impl(const void *);
  __device__ size_t __nv_cvta_generic_to_constant_impl(const void *);
  __device__ size_t __nv_cvta_generic_to_local_impl(const void *);
  __device__ void * __nv_cvta_global_to_generic_impl(size_t);
  __device__ void * __nv_cvta_shared_to_generic_impl(size_t);
  __device__ void * __nv_cvta_constant_to_generic_impl(size_t);
  __device__ void * __nv_cvta_local_to_generic_impl(size_t);
}

__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_global(const void *p)
{
  return __nv_cvta_generic_to_global_impl(p);
}

__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_shared(const void *p)
{
  return __nv_cvta_generic_to_shared_impl(p);
}

__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_constant(const void *p)
{
  return __nv_cvta_generic_to_constant_impl(p);
}

__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_local(const void *p)
{
  return __nv_cvta_generic_to_local_impl(p);
}

__SM_20_INTRINSICS_DECL__ void * __cvta_global_to_generic(size_t rawbits)
{
  return __nv_cvta_global_to_generic_impl(rawbits);
}

__SM_20_INTRINSICS_DECL__ void * __cvta_shared_to_generic(size_t rawbits)
{
  return __nv_cvta_shared_to_generic_impl(rawbits);
}

__SM_20_INTRINSICS_DECL__ void * __cvta_constant_to_generic(size_t rawbits)
{
  return __nv_cvta_constant_to_generic_impl(rawbits);
}

__SM_20_INTRINSICS_DECL__ void * __cvta_local_to_generic(size_t rawbits)
{
  return __nv_cvta_local_to_generic_impl(rawbits);
}

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_20_INTRINSICS_DECL__

#endif /* !__SM_20_INTRINSICS_HPP__ */

