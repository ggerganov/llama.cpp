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

#if !defined(__SM_60_ATOMIC_FUNCTIONS_HPP__)
#define __SM_60_ATOMIC_FUNCTIONS_HPP__

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

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__SM_60_ATOMIC_FUNCTIONS_DECL__ double atomicAdd(double *address, double val)
{
  return __dAtomicAdd(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAdd_block(int *address, int val)
{
  return __iAtomicAdd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAdd_system(int *address, int val)
{
  return __iAtomicAdd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAdd_block(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAdd_system(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAdd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAdd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicAdd_block(float *address, float val)
{
  return __fAtomicAdd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicAdd_system(float *address, float val)
{
  return __fAtomicAdd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
double atomicAdd_block(double *address, double val)
{
  return __dAtomicAdd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
double atomicAdd_system(double *address, double val)
{
  return __dAtomicAdd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicSub_block(int *address, int val)
{
  return __iAtomicAdd_block(address, (unsigned int)-(int)val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicSub_system(int *address, int val)
{
  return __iAtomicAdd_system(address, (unsigned int)-(int)val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicSub_block(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_block(address, (unsigned int)-(int)val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicSub_system(unsigned int *address, unsigned int val)
{
  return __uAtomicAdd_system(address, (unsigned int)-(int)val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicExch_block(int *address, int val)
{
  return __iAtomicExch_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicExch_system(int *address, int val)
{
  return __iAtomicExch_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicExch_block(unsigned int *address, unsigned int val)
{
  return __uAtomicExch_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicExch_system(unsigned int *address, unsigned int val)
{
  return __uAtomicExch_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicExch_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicExch_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicExch_block(float *address, float val)
{
  return __fAtomicExch_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
float atomicExch_system(float *address, float val)
{
  return __fAtomicExch_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMin_block(int *address, int val)
{
  return __iAtomicMin_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMin_system(int *address, int val)
{
  return __iAtomicMin_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMin_block(long long *address, long long val)
{
  return __illAtomicMin_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMin_system(long long *address, long long val)
{
  return __illAtomicMin_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMin_block(unsigned int *address, unsigned int val)
{
  return __uAtomicMin_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMin_system(unsigned int *address, unsigned int val)
{
  return __uAtomicMin_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMin_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMin_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMax_block(int *address, int val)
{
  return __iAtomicMax_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicMax_system(int *address, int val)
{
  return __iAtomicMax_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMax_block(long long *address, long long val)
{
  return __illAtomicMax_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicMax_system(long long *address, long long val)
{
  return __illAtomicMax_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMax_block(unsigned int *address, unsigned int val)
{
  return __uAtomicMax_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicMax_system(unsigned int *address, unsigned int val)
{
  return __uAtomicMax_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMax_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicMax_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicInc_block(unsigned int *address, unsigned int val)
{
  return __uAtomicInc_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicInc_system(unsigned int *address, unsigned int val)
{
  return __uAtomicInc_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicDec_block(unsigned int *address, unsigned int val)
{
  return __uAtomicDec_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicDec_system(unsigned int *address, unsigned int val)
{
  return __uAtomicDec_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicCAS_block(int *address, int compare, int val)
{
  return __iAtomicCAS_block(address, compare, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicCAS_system(int *address, int compare, int val)
{
  return __iAtomicCAS_system(address, compare, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicCAS_block(unsigned int *address, unsigned int compare,
                             unsigned int val)
{
  return __uAtomicCAS_block(address, compare, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicCAS_system(unsigned int *address, unsigned int compare,
                              unsigned int val)
{
  return __uAtomicCAS_system(address, compare, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long int atomicCAS_block(unsigned long long int *address,
                                       unsigned long long int compare,
                                       unsigned long long int val)
{
  return __ullAtomicCAS_block(address, compare, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long int atomicCAS_system(unsigned long long int *address,
                                        unsigned long long int compare,
                                        unsigned long long int val)
{
  return __ullAtomicCAS_system(address, compare, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAnd_block(int *address, int val)
{
  return __iAtomicAnd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicAnd_system(int *address, int val)
{
  return __iAtomicAnd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicAnd_block(long long *address, long long val)
{
  return __llAtomicAnd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicAnd_system(long long *address, long long val)
{
  return __llAtomicAnd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAnd_block(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicAnd_system(unsigned int *address, unsigned int val)
{
  return __uAtomicAnd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAnd_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicAnd_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicOr_block(int *address, int val)
{
  return __iAtomicOr_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicOr_system(int *address, int val)
{
  return __iAtomicOr_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicOr_block(long long *address, long long val)
{
  return __llAtomicOr_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicOr_system(long long *address, long long val)
{
  return __llAtomicOr_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicOr_block(unsigned int *address, unsigned int val)
{
  return __uAtomicOr_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicOr_system(unsigned int *address, unsigned int val)
{
  return __uAtomicOr_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicOr_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicOr_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicXor_block(int *address, int val)
{
  return __iAtomicXor_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
int atomicXor_system(int *address, int val)
{
  return __iAtomicXor_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicXor_block(long long *address, long long val)
{
  return __llAtomicXor_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
long long atomicXor_system(long long *address, long long val)
{
  return __llAtomicXor_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicXor_block(unsigned int *address, unsigned int val)
{
  return __uAtomicXor_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned int atomicXor_system(unsigned int *address, unsigned int val)
{
  return __uAtomicXor_system(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicXor_block(address, val);
}

__SM_60_ATOMIC_FUNCTIONS_DECL__
unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val)
{
  return __ullAtomicXor_system(address, val);
}

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 600 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_60_ATOMIC_FUNCTIONS_DECL__

#endif /* !__SM_60_ATOMIC_FUNCTIONS_HPP__ */

