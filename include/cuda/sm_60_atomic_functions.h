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

#if !defined(__SM_60_ATOMIC_FUNCTIONS_H__)
#define __SM_60_ATOMIC_FUNCTIONS_H__


#if defined(__CUDACC_RTC__)
#define __SM_60_ATOMIC_FUNCTIONS_DECL__ __device__
#else /* __CUDACC_RTC__ */
#define __SM_60_ATOMIC_FUNCTIONS_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */



#ifdef __CUDA_ARCH__ 
extern "C"
{
extern __device__ __device_builtin__ double __dAtomicAdd(double *address, double val);

extern __device__ __device_builtin__
int __iAtomicAdd_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicAdd_system(int *address, int val);

extern __device__ __device_builtin__
unsigned int __uAtomicAdd_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicAdd_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicAdd_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicAdd_system(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
float __fAtomicAdd_block(float *address, float val);

extern __device__ __device_builtin__
float __fAtomicAdd_system(float *address, float val);

extern __device__ __device_builtin__
double __dAtomicAdd_block(double *address, double val);

extern __device__ __device_builtin__
double __dAtomicAdd_system(double *address, double val);

extern __device__ __device_builtin__
int __iAtomicExch_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicExch_system(int *address, int val);

extern __device__ __device_builtin__
unsigned int __uAtomicExch_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicExch_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicExch_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicExch_system(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
float __fAtomicExch_block(float *address, float val);

extern __device__ __device_builtin__
float __fAtomicExch_system(float *address, float val);

extern __device__ __device_builtin__
int __iAtomicMin_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicMin_system(int *address, int val);

extern __device__ __device_builtin__
long long __illAtomicMin_block(long long *address, long long val);

extern __device__ __device_builtin__
long long __illAtomicMin_system(long long *address, long long val);

extern __device__ __device_builtin__
unsigned int __uAtomicMin_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicMin_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicMin_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicMin_system(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
int __iAtomicMax_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicMax_system(int *address, int val);

extern __device__ __device_builtin__
long long __illAtomicMax_block(long long *address, long long val);

extern __device__ __device_builtin__
long long __illAtomicMax_system(long long *address, long long val);

extern __device__ __device_builtin__
unsigned int __uAtomicMax_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicMax_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicMax_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicMax_system(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned int __uAtomicInc_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicInc_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicDec_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicDec_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
int __iAtomicCAS_block(int *address, int compare, int val);

extern __device__ __device_builtin__
int __iAtomicCAS_system(int *address, int compare, int val);

extern __device__ __device_builtin__
unsigned int __uAtomicCAS_block(unsigned int *address, unsigned int compare,
                                unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicCAS_system(unsigned int *address, unsigned int compare,
                                 unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicCAS_block(unsigned long long int *address,
                                        unsigned long long int compare,
                                        unsigned long long int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicCAS_system(unsigned long long int *address,
                                         unsigned long long int compare,
                                         unsigned long long int val);

extern __device__ __device_builtin__
int __iAtomicAnd_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicAnd_system(int *address, int val);

extern __device__ __device_builtin__
long long __llAtomicAnd_block(long long *address, long long val);

extern __device__ __device_builtin__
long long __llAtomicAnd_system(long long *address, long long val);

extern __device__ __device_builtin__
unsigned int __uAtomicAnd_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicAnd_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicAnd_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicAnd_system(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
int __iAtomicOr_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicOr_system(int *address, int val);

extern __device__ __device_builtin__
long long __llAtomicOr_block(long long *address, long long val);

extern __device__ __device_builtin__
long long __llAtomicOr_system(long long *address, long long val);

extern __device__ __device_builtin__
unsigned int __uAtomicOr_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicOr_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicOr_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicOr_system(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
int __iAtomicXor_block(int *address, int val);

extern __device__ __device_builtin__
int __iAtomicXor_system(int *address, int val);

extern __device__ __device_builtin__
long long __llAtomicXor_block(long long *address, long long val);

extern __device__ __device_builtin__
long long __llAtomicXor_system(long long *address, long long val);

extern __device__ __device_builtin__
unsigned int __uAtomicXor_block(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned int __uAtomicXor_system(unsigned int *address, unsigned int val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicXor_block(unsigned long long *address, unsigned long long val);

extern __device__ __device_builtin__
unsigned long long __ullAtomicXor_system(unsigned long long *address, unsigned long long val);
}
#endif /* __CUDA_ARCH__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SM_60_ATOMIC_FUNCTIONS_DECL__ double atomicAdd(double *address, double val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAdd_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAdd_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAdd_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAdd_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicAdd_block(float *address, float val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicAdd_system(float *address, float val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
double atomicAdd_block(double *address, double val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
double atomicAdd_system(double *address, double val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicSub_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicSub_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicSub_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicSub_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicExch_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicExch_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicExch_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicExch_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicExch_block(float *address, float val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicExch_system(float *address, float val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMin_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMin_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMin_block(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMin_system(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMin_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMin_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMax_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMax_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMax_block(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMax_system(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMax_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMax_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicInc_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicInc_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicDec_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicDec_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicCAS_block(int *address, int compare, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicCAS_system(int *address, int compare, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicCAS_block(unsigned int *address, unsigned int compare,
                             unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicCAS_system(unsigned int *address, unsigned int compare,
                              unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long int atomicCAS_block(unsigned long long int *address,
                                       unsigned long long int compare,
                                       unsigned long long int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long int atomicCAS_system(unsigned long long int *address,
                                        unsigned long long int compare,
                                        unsigned long long int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAnd_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAnd_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicAnd_block(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicAnd_system(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAnd_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAnd_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicOr_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicOr_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicOr_block(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicOr_system(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicOr_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicOr_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicXor_block(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicXor_system(int *address, int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicXor_block(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicXor_system(long long *address, long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicXor_block(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicXor_system(unsigned int *address, unsigned int val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) __DEF_IF_HOST

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 600 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_60_ATOMIC_FUNCTIONS_DECL__
#undef __DEF_IF_HOST

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
#include "sm_60_atomic_functions.hpp"
#endif /* !__CUDACC_RTC__  && defined(__CUDA_ARCH__)  */

#endif /* !__SM_60_ATOMIC_FUNCTIONS_H__ */

