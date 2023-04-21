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

#if !defined(__SM_30_INTRINSICS_HPP__)
#define __SM_30_INTRINSICS_HPP__

#if defined(__CUDACC_RTC__)
#define __SM_30_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_30_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

// In here are intrinsics which are built in to the compiler. These may be
// referenced by intrinsic implementations from this file.
extern "C"
{
}

/*******************************************************************************
*                                                                              *
*  Below are implementations of SM-3.0 intrinsics which are included as        *
*  source (instead of being built in to the compiler)                          *
*                                                                              *
*******************************************************************************/

#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif

__SM_30_INTRINSICS_DECL__
unsigned __fns(unsigned mask, unsigned base, int offset) {
  extern __device__ __device_builtin__ unsigned int __nvvm_fns(unsigned int mask, unsigned int base, int offset);
  return __nvvm_fns(mask, base, offset);
}

__SM_30_INTRINSICS_DECL__ 
void  __barrier_sync(unsigned id) {
  extern __device__ __device_builtin__ void __nvvm_barrier_sync(unsigned id);
  return __nvvm_barrier_sync(id);
}

__SM_30_INTRINSICS_DECL__ 
void  __barrier_sync_count(unsigned id, unsigned cnt) {
  extern __device__ __device_builtin__ void __nvvm_barrier_sync_cnt(unsigned id, unsigned cnt);
  return __nvvm_barrier_sync_cnt(id, cnt);
}

__SM_30_INTRINSICS_DECL__ 
void  __syncwarp(unsigned mask) {
  extern __device__ __device_builtin__ void __nvvm_bar_warp_sync(unsigned mask);
  return __nvvm_bar_warp_sync(mask);
}

__SM_30_INTRINSICS_DECL__ 
int __all_sync(unsigned mask, int pred) {
  extern __device__ __device_builtin__ int __nvvm_vote_all_sync(unsigned int mask, int pred); 
  return __nvvm_vote_all_sync(mask, pred);
}

__SM_30_INTRINSICS_DECL__ 
int __any_sync(unsigned mask, int pred) {
  extern __device__ __device_builtin__ int __nvvm_vote_any_sync(unsigned int mask, int pred); 
  return __nvvm_vote_any_sync(mask, pred);
}

__SM_30_INTRINSICS_DECL__ 
int __uni_sync(unsigned mask, int pred) {
  extern __device__ __device_builtin__ int __nvvm_vote_uni_sync(unsigned int mask, int pred); 
  return __nvvm_vote_uni_sync(mask, pred);
}

__SM_30_INTRINSICS_DECL__ 
unsigned __ballot_sync(unsigned mask, int pred) {
  extern __device__ __device_builtin__ unsigned int __nvvm_vote_ballot_sync(unsigned int mask, int pred); 
  return __nvvm_vote_ballot_sync(mask, pred);
}

__SM_30_INTRINSICS_DECL__
unsigned __activemask() {
    unsigned ret;
    asm volatile ("activemask.b32 %0;" : "=r"(ret));
    return ret;
}

// These are removed starting with compute_70 and onwards
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700

__SM_30_INTRINSICS_DECL__ int __shfl(int var, int srcLane, int width) {
	int ret;
	int c = ((warpSize-width) << 8) | 0x1f;
	asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(srcLane), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl(unsigned int var, int srcLane, int width) {
	return (unsigned int) __shfl((int)var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ int __shfl_up(int var, unsigned int delta, int width) {
	int ret;
	int c = (warpSize-width) << 8;
	asm volatile ("shfl.up.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(delta), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_up(unsigned int var, unsigned int delta, int width) {
	return (unsigned int) __shfl_up((int)var, delta, width);
}

__SM_30_INTRINSICS_DECL__ int __shfl_down(int var, unsigned int delta, int width) {
	int ret;
	int c = ((warpSize-width) << 8) | 0x1f;
	asm volatile ("shfl.down.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(delta), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_down(unsigned int var, unsigned int delta, int width) {
	return (unsigned int) __shfl_down((int)var, delta, width);
}

__SM_30_INTRINSICS_DECL__ int __shfl_xor(int var, int laneMask, int width) {
	int ret;
	int c = ((warpSize-width) << 8) | 0x1f;
	asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(var), "r"(laneMask), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_xor(unsigned int var, int laneMask, int width) {
	return (unsigned int) __shfl_xor((int)var, laneMask, width);
}

__SM_30_INTRINSICS_DECL__ float __shfl(float var, int srcLane, int width) {
	float ret;
        int c;
	c = ((warpSize-width) << 8) | 0x1f;
	asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(srcLane), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ float __shfl_up(float var, unsigned int delta, int width) {
	float ret;
        int c;
	c = (warpSize-width) << 8;
	asm volatile ("shfl.up.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(delta), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ float __shfl_down(float var, unsigned int delta, int width) {
	float ret;
        int c;
	c = ((warpSize-width) << 8) | 0x1f;
	asm volatile ("shfl.down.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(delta), "r"(c));
	return ret;
}

__SM_30_INTRINSICS_DECL__ float __shfl_xor(float var, int laneMask, int width) {
	float ret;
        int c;
	c = ((warpSize-width) << 8) | 0x1f;
	asm volatile ("shfl.bfly.b32 %0, %1, %2, %3;" : "=f"(ret) : "f"(var), "r"(laneMask), "r"(c));
	return ret;
}

// 64-bits SHFL

__SM_30_INTRINSICS_DECL__ long long __shfl(long long var, int srcLane, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl(hi, srcLane, width);
	lo = __shfl(lo, srcLane, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl(unsigned long long var, int srcLane, int width) {
	return (unsigned long long) __shfl((long long) var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ long long __shfl_up(long long var, unsigned int delta, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_up(hi, delta, width);
	lo = __shfl_up(lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_up(unsigned long long var, unsigned int delta, int width) {
	return (unsigned long long) __shfl_up((long long) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long long __shfl_down(long long var, unsigned int delta, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_down(hi, delta, width);
	lo = __shfl_down(lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_down(unsigned long long var, unsigned int delta, int width) {
	return (unsigned long long) __shfl_down((long long) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long long __shfl_xor(long long var, int laneMask, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_xor(hi, laneMask, width);
	lo = __shfl_xor(lo, laneMask, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width) {
	return (unsigned long long) __shfl_xor((long long) var, laneMask, width);
}

__SM_30_INTRINSICS_DECL__ double __shfl(double var, int srcLane, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl(hi, srcLane, width);
	lo = __shfl(lo, srcLane, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ double __shfl_up(double var, unsigned int delta, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_up(hi, delta, width);
	lo = __shfl_up(lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ double __shfl_down(double var, unsigned int delta, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_down(hi, delta, width);
	lo = __shfl_down(lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ double __shfl_xor(double var, int laneMask, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_xor(hi, laneMask, width);
	lo = __shfl_xor(lo, laneMask, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ long __shfl(long var, int srcLane, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl((long long) var, srcLane, width) :
		__shfl((int) var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl(unsigned long var, int srcLane, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl((unsigned long long) var, srcLane, width) :
		__shfl((unsigned int) var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ long __shfl_up(long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_up((long long) var, delta, width) :
		__shfl_up((int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_up(unsigned long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_up((unsigned long long) var, delta, width) :
		__shfl_up((unsigned int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long __shfl_down(long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_down((long long) var, delta, width) :
		__shfl_down((int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_down(unsigned long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_down((unsigned long long) var, delta, width) :
		__shfl_down((unsigned int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long __shfl_xor(long var, int laneMask, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_xor((long long) var, laneMask, width) :
		__shfl_xor((int) var, laneMask, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_xor(unsigned long var, int laneMask, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_xor((unsigned long long) var, laneMask, width) :
		__shfl_xor((unsigned int) var, laneMask, width);
}

#endif /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700 */

// Warp register exchange (shuffle) intrinsics.
// Notes:
// a) Warp size is hardcoded to 32 here, because the compiler does not know
//    the "warpSize" constant at this time
// b) we cannot map the float __shfl to the int __shfl because it'll mess with
//    the register number (especially if you're doing two shfls to move a double).
__SM_30_INTRINSICS_DECL__ int __shfl_sync(unsigned mask, int var, int srcLane, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
	int c = ((warpSize-width) << 8) | 0x1f;
        ret = __nvvm_shfl_idx_sync(mask, var, srcLane, c);
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_sync(unsigned mask, unsigned int var, int srcLane, int width) {
        return (unsigned int) __shfl_sync(mask, (int)var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ int __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
	int c = (warpSize-width) << 8;
        ret = __nvvm_shfl_up_sync(mask, var, delta, c);
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_up_sync(unsigned mask, unsigned int var, unsigned int delta, int width) {
        return (unsigned int) __shfl_up_sync(mask, (int)var, delta, width);
}

__SM_30_INTRINSICS_DECL__ int __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
	int c = ((warpSize-width) << 8) | 0x1f;
        ret = __nvvm_shfl_down_sync(mask, var, delta, c);
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_down_sync(unsigned mask, unsigned int var, unsigned int delta, int width) {
        return (unsigned int) __shfl_down_sync(mask, (int)var, delta, width);
}

__SM_30_INTRINSICS_DECL__ int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_bfly_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
	int c = ((warpSize-width) << 8) | 0x1f;
        ret = __nvvm_shfl_bfly_sync(mask, var, laneMask, c);
	return ret;
}

__SM_30_INTRINSICS_DECL__ unsigned int __shfl_xor_sync(unsigned mask, unsigned int var, int laneMask, int width) {
	return (unsigned int) __shfl_xor_sync(mask, (int)var, laneMask, width);
}

__SM_30_INTRINSICS_DECL__ float __shfl_sync(unsigned mask, float var, int srcLane, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_idx_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
        int ret;
        int c;
	c = ((warpSize-width) << 8) | 0x1f;
        ret = __nvvm_shfl_idx_sync(mask, __float_as_int(var), srcLane, c);
	return __int_as_float(ret);
}

__SM_30_INTRINSICS_DECL__ float __shfl_up_sync(unsigned mask, float var, unsigned int delta, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_up_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
        int c;
	c = (warpSize-width) << 8;
        ret = __nvvm_shfl_up_sync(mask, __float_as_int(var), delta, c);
	return __int_as_float(ret);
}

__SM_30_INTRINSICS_DECL__ float __shfl_down_sync(unsigned mask, float var, unsigned int delta, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_down_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
        int c;
	c = ((warpSize-width) << 8) | 0x1f;
        ret = __nvvm_shfl_down_sync(mask, __float_as_int(var), delta, c);
	return __int_as_float(ret);
}

__SM_30_INTRINSICS_DECL__ float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width) {
        extern __device__ __device_builtin__ unsigned __nvvm_shfl_bfly_sync(unsigned mask, unsigned a, unsigned b, unsigned c);
	int ret;
        int c;
	c = ((warpSize-width) << 8) | 0x1f;
        ret = __nvvm_shfl_bfly_sync(mask, __float_as_int(var), laneMask, c);
	return __int_as_float(ret);
}

// 64-bits SHFL
__SM_30_INTRINSICS_DECL__ long long __shfl_sync(unsigned mask, long long var, int srcLane, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_sync(mask, hi, srcLane, width);
	lo = __shfl_sync(mask, lo, srcLane, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width) {
        return (unsigned long long) __shfl_sync(mask, (long long) var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ long long __shfl_up_sync(unsigned mask, long long var, unsigned int delta, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_up_sync(mask, hi, delta, width);
	lo = __shfl_up_sync(mask, lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned int delta, int width) {
        return (unsigned long long) __shfl_up_sync(mask, (long long) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long long __shfl_down_sync(unsigned mask, long long var, unsigned int delta, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_down_sync(mask, hi, delta, width);
	lo = __shfl_down_sync(mask, lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned int delta, int width) {
        return (unsigned long long) __shfl_down_sync(mask, (long long) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width) {
	int lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
	hi = __shfl_xor_sync(mask, hi, laneMask, width);
	lo = __shfl_xor_sync(mask, lo, laneMask, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=l"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width) {
        return (unsigned long long) __shfl_xor_sync(mask, (long long) var, laneMask, width);
}

__SM_30_INTRINSICS_DECL__ double __shfl_sync(unsigned mask, double var, int srcLane, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_sync(mask, hi, srcLane, width);
	lo = __shfl_sync(mask, lo, srcLane, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ double __shfl_up_sync(unsigned mask, double var, unsigned int delta, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_up_sync(mask, hi, delta, width);
	lo = __shfl_up_sync(mask, lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ double __shfl_down_sync(unsigned mask, double var, unsigned int delta, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_down_sync(mask, hi, delta, width);
	lo = __shfl_down_sync(mask, lo, delta, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

__SM_30_INTRINSICS_DECL__ double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width) {
	unsigned lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
	hi = __shfl_xor_sync(mask, hi, laneMask, width);
	lo = __shfl_xor_sync(mask, lo, laneMask, width);
	asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "r"(lo), "r"(hi));
	return var;
}

// long needs some help to choose between 32-bits and 64-bits

__SM_30_INTRINSICS_DECL__ long __shfl_sync(unsigned mask, long var, int srcLane, int width) {
	return (sizeof(long) == sizeof(long long)) ?
                __shfl_sync(mask, (long long) var, srcLane, width) :
		__shfl_sync(mask, (int) var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width) {
	return (sizeof(long) == sizeof(long long)) ?
                __shfl_sync(mask, (unsigned long long) var, srcLane, width) :
		__shfl_sync(mask, (unsigned int) var, srcLane, width);
}

__SM_30_INTRINSICS_DECL__ long __shfl_up_sync(unsigned mask, long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_up_sync(mask, (long long) var, delta, width) :
		__shfl_up_sync(mask, (int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_up_sync(mask, (unsigned long long) var, delta, width) :
		__shfl_up_sync(mask, (unsigned int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long __shfl_down_sync(unsigned mask, long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_down_sync(mask, (long long) var, delta, width) :
		__shfl_down_sync(mask, (int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned int delta, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_down_sync(mask, (unsigned long long) var, delta, width) :
		__shfl_down_sync(mask, (unsigned int) var, delta, width);
}

__SM_30_INTRINSICS_DECL__ long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_xor_sync(mask, (long long) var, laneMask, width) :
		__shfl_xor_sync(mask, (int) var, laneMask, width);
}

__SM_30_INTRINSICS_DECL__ unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width) {
	return (sizeof(long) == sizeof(long long)) ?
		__shfl_xor_sync(mask, (unsigned long long) var, laneMask, width) :
		__shfl_xor_sync(mask, (unsigned int) var, laneMask, width);
}

#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 300 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_30_INTRINSICS_DECL__

#endif /* !__SM_30_INTRINSICS_HPP__ */

