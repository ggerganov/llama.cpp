/*
 * Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
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
#pragma message("crt/mma.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/mma.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_HPP__
#endif

#if !defined(__CUDA_MMA_HPP__)
#define __CUDA_MMA_HPP__

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define __CUDA_MMA_DEVICE_DECL__ static __device__ __inline__

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 720
#define __CUDA_IMMA__ 1
#endif  /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 720 */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 730
#define __CUDA_SUBBYTE_IMMA__ 1
#endif  /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 730 */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define __CUDA_AMPERE_MMA__ 1
#endif  /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800 */

namespace nvcuda {
namespace wmma {

  // 
  // Load functions for frags of shape m16n16k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_m16n16k16_ld_a((int*)&a, (const int*)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_m16n16k16_ld_a((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b,16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_m16n16k16_ld_b((int*)&a, (const int*)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b,16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_m16n16k16_ld_b((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator,16, 16, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m16n16k16_ld_c_f16((int*)&a, (const int*)p, ldm, 0);
    else
      __hmma_m16n16k16_ld_c_f16((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator,16, 16, 16, float>& a, const float* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m16n16k16_ld_c_f32((float*)&a, (const float*)p, ldm, 0);
    else
      __hmma_m16n16k16_ld_c_f32((float*)&a, (const float*)p, ldm, 1);
  }

#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_m16n16k16_ld_a_s8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_m16n16k16_ld_a_s8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m16n16k16_ld_a_u8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m16n16k16_ld_a_u8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_m16n16k16_ld_b_s8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_m16n16k16_ld_b_s8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m16n16k16_ld_b_u8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m16n16k16_ld_b_u8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator,16, 16, 16, int>& a, const int* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m16n16k16_ld_c((int *)&a, (const int*)p, ldm, 0);
    else
      __imma_m16n16k16_ld_c((int *)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m16n16k16_ld_a((int*)&a, (const int*)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m16n16k16_ld_a((int*)&a, (const int*)p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm)  {
    __mma_bf16_m16n16k16_ld_b((int*)&a, (const int*)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm)  {
    __mma_bf16_m16n16k16_ld_b((int*)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_AMPERE_MMA__ */


  // 
  // Load functions for frags of shape m32n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_m32n8k16_ld_a((int*)&a, (const int*)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_m32n8k16_ld_a((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_m32n8k16_ld_b((int*)&a, (const int*)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_m32n8k16_ld_b((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 8, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m32n8k16_ld_c_f16((int*)&a, (const int*)p, ldm, 0);
    else
      __hmma_m32n8k16_ld_c_f16((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 8, 16, float>& a, const float* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m32n8k16_ld_c_f32((float*)&a, (const float*)p, ldm, 0);
    else
      __hmma_m32n8k16_ld_c_f32((float*)&a, (const float*)p, ldm, 1);
  }

#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_m32n8k16_ld_a_s8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_m32n8k16_ld_a_s8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m32n8k16_ld_a_u8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m32n8k16_ld_a_u8((int *)&a, (const int *)p, ldm, 1);
  }
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_m32n8k16_ld_b_s8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_m32n8k16_ld_b_s8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m32n8k16_ld_b_u8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m32n8k16_ld_b_u8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 8, 16, int>& a, const int* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m32n8k16_ld_c((int *)&a, (const int*)p, ldm, 0);
    else
      __imma_m32n8k16_ld_c((int *)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_IMMA__ */ 

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m32n8k16_ld_a((int*)&a, (const int*)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m32n8k16_ld_a((int*)&a, (const int*)p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m32n8k16_ld_b((int*)&a, (const int*)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m32n8k16_ld_b((int*)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_AMPERE_MMA__ */


  // 
  // Load functions for frags of shape m8n32k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_m8n32k16_ld_a((int*)&a, (const int*)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_m8n32k16_ld_a((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_m8n32k16_ld_b((int*)&a, (const int*)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_m8n32k16_ld_b((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 32, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m8n32k16_ld_c_f16((int*)&a, (const int*)p, ldm, 0);
    else
      __hmma_m8n32k16_ld_c_f16((int*)&a, (const int*)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 32, 16, float>& a, const float* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m8n32k16_ld_c_f32((float*)&a, (const float*)p, ldm, 0);
    else
      __hmma_m8n32k16_ld_c_f32((float*)&a, (const float*)p, ldm, 1);
  }
  
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_m8n32k16_ld_a_s8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_m8n32k16_ld_a_s8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m8n32k16_ld_a_u8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m8n32k16_ld_a_u8((int *)&a, (const int *)p, ldm, 1);
  }
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_m8n32k16_ld_b_s8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_m8n32k16_ld_b_s8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m8n32k16_ld_b_u8((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_m8n32k16_ld_b_u8((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 32, 16, int>& a, const int* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m8n32k16_ld_c((int *)&a, (const int*)p, ldm, 0);
    else
      __imma_m8n32k16_ld_c((int *)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_IMMA__ */ 

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m8n32k16_ld_a((int*)&a, (const int*)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m8n32k16_ld_a((int*)&a, (const int*)p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m8n32k16_ld_b((int*)&a, (const int*)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) {
    __mma_bf16_m8n32k16_ld_b((int*)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_AMPERE_MMA__ */
  

#ifdef __CUDA_SUBBYTE_IMMA__
  //
  // Load functions for frags of shape m8n8k32
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major>& a, const void* p, unsigned ldm) {
      __imma_m8n8k32_ld_a_s4((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 32, experimental::precision::u4, row_major>& a, const void* p, unsigned ldm) {
      __imma_m8n8k32_ld_a_u4((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major>& a, const void* p, unsigned ldm) {
      __imma_m8n8k32_ld_b_s4((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 32, experimental::precision::u4, col_major>& a, const void* p, unsigned ldm) {
      __imma_m8n8k32_ld_b_u4((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 32, int>& a, const int* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m8n8k32_ld_c((int *)&a, (const int*)p, ldm, 0);
    else
      __imma_m8n8k32_ld_c((int *)&a, (const int*)p, ldm, 1);
  }

  //
  // Load functions for frags of shape m8n8k128
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major>& a, const void* p, unsigned ldm) {
    __bmma_m8n8k128_ld_a_b1((int *)&a, (const int *)p, ldm, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major>& a, const void* p, unsigned ldm) {
    __bmma_m8n8k128_ld_b_b1((int *)&a, (const int *)p, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 128, int>& a, const int* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __bmma_m8n8k128_ld_c((int *)&a, (const int*)p, ldm, 0);
    else
      __bmma_m8n8k128_ld_c((int *)&a, (const int*)p, ldm, 1);
  }
#endif  /* __CUDA_SUBBYTE_IMMA__ */



#ifdef __CUDA_AMPERE_MMA__
  // load functions for frags of shape m16n16k8
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const float* p, unsigned ldm) {
    __mma_tf32_m16n16k8_ld_a((int *)&a, (const int *)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const float* p, unsigned ldm) {
    __mma_tf32_m16n16k8_ld_a((int *)&a, (const int *)p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& a, const float* p, unsigned ldm) {
    __mma_tf32_m16n16k8_ld_b((int *)&a, (const int *)p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& a, const float* p, unsigned ldm) {
    __mma_tf32_m16n16k8_ld_b((int *)&a, (const int *)p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 8, float>& a, const float* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __mma_tf32_m16n16k8_ld_c((float *)&a, p, ldm, 0);
    else
      __mma_tf32_m16n16k8_ld_c((float *)&a, p, ldm, 1);      
  }
  
  // load functions for frags of shape m8n8k4
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, double, row_major>& a, const double* p, unsigned ldm) {
    __dmma_m8n8k4_ld_a((double *)&a, p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, double, col_major>& a, const double* p, unsigned ldm) {
    __dmma_m8n8k4_ld_a((double *)&a, p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, double, row_major>& a, const double* p, unsigned ldm) {
    __dmma_m8n8k4_ld_b((double *)&a, p, ldm, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, double, col_major>& a, const double* p, unsigned ldm) {
    __dmma_m8n8k4_ld_b((double *)&a, p, ldm, 1);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 4, double>& a, const double* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __dmma_m8n8k4_ld_c((double *)&a, p, ldm, 0);
    else
      __dmma_m8n8k4_ld_c((double *)&a, p, ldm, 1);      
  }
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // Store functions for frags of shape m16n16k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator,16, 16, 16, __half>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m16n16k16_st_c_f16((int*)p, (int*)&a, ldm, 0);
    else
      __hmma_m16n16k16_st_c_f16((int*)p, (int*)&a, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator,16, 16, 16, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m16n16k16_st_c_f32((float*)p, (float*)&a, ldm, 0);
    else
      __hmma_m16n16k16_st_c_f32((float*)p, (float*)&a, ldm, 1);
  }
  
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator,16, 16, 16, int>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m16n16k16_st_c_i32(p, (const int*)&a, ldm, 0);
    else
      __imma_m16n16k16_st_c_i32(p, (const int*)&a, ldm, 1);
  }
#endif  /* __CUDA_IMMA__ */

  // 
  // Store functions for frags of shape m32n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 32, 8, 16, __half>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m32n8k16_st_c_f16((int*)p, (int*)&a, ldm, 0);
    else
      __hmma_m32n8k16_st_c_f16((int*)p, (int*)&a, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 32, 8, 16, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m32n8k16_st_c_f32((float*)p, (float*)&a, ldm, 0);
    else
      __hmma_m32n8k16_st_c_f32((float*)p, (float*)&a, ldm, 1);
  }
  
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 32, 8, 16, int>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m32n8k16_st_c_i32(p, (const int*)&a, ldm, 0);
    else
      __imma_m32n8k16_st_c_i32(p, (const int*)&a, ldm, 1);
  }
#endif  /* __CUDA_IMMA__ */

  // 
  // Store functions for frags of shape m8n32k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 8, 32, 16, __half>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m8n32k16_st_c_f16((int*)p, (int*)&a, ldm, 0);
    else
      __hmma_m8n32k16_st_c_f16((int*)p, (int*)&a, ldm, 1);
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 8, 32, 16, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __hmma_m8n32k16_st_c_f32((float*)p, (float*)&a, ldm, 0);
    else
      __hmma_m8n32k16_st_c_f32((float*)p, (float*)&a, ldm, 1);
  }

#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 8, 32, 16, int>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m8n32k16_st_c_i32(p, (const int*)&a, ldm, 0);
    else
      __imma_m8n32k16_st_c_i32(p, (const int*)&a, ldm, 1);
  }
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_SUBBYTE_IMMA__
  // 
  // Store functions for frags of shape m8n8k32
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 8, 8, 32, int>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_m8n8k32_st_c_i32(p, (const int*)&a, ldm, 0);
    else
      __imma_m8n8k32_st_c_i32(p, (const int*)&a, ldm, 1);
  }

  // 
  // Store functions for frags of shape m8n8k128
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 8, 8, 128, int>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __bmma_m8n8k128_st_c_i32(p, (const int*)&a, ldm, 0);
    else
      __bmma_m8n8k128_st_c_i32(p, (const int*)&a, ldm, 1);
  }
#endif  /* __CUDA_SUBBYTE_IMMA__ */


#ifdef __CUDA_AMPERE_MMA__

  //
  // Store functions for frags of shape m16n16k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 8, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __mma_m16n16k8_st_c_f32(p, (const float*)&a, ldm, 0);
    else
      __mma_m16n16k8_st_c_f32(p, (const float*)&a, ldm, 1);
  }

  
  // 
  // Store functions for frags of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(double *p, const fragment<accumulator, 8, 8, 4, double>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __dmma_m8n8k4_st_c_f64(p, (const double*)&a, ldm, 0);
    else
      __dmma_m8n8k4_st_c_f64(p, (const double*)&a, ldm, 1);
  }
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // MMA functions for shape m16n16k16
  // 
  // D fp16, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 3, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 2, 0);
  }

  // D fp32, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 3, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
      __hmma_m16n16k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) {
    __hmma_m16n16k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 2, 0);
  }

  // D fp32, C fp32
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);
  }

  // D fp16, C fp32
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __hmma_m16n16k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);
  }

#ifdef __CUDA_IMMA__  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, row_major>& a, const fragment<matrix_b,16, 16, 16, signed char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 1, 1);
    else
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 1, 0);
  }
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, col_major>& a, const fragment<matrix_b,16, 16, 16, signed char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 3, 1);
    else
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 3, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, row_major>& a, const fragment<matrix_b,16, 16, 16, signed char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 0, 1);
    else
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 0, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, col_major>& a, const fragment<matrix_b,16, 16, 16, signed char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 2, 1);
    else
      __imma_m16n16k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 2, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, row_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 1, 1);
    else
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 1, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, col_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 3, 1);
    else
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 3, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, row_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 0, 1);
    else
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 0, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, col_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf) {
    if (satf)
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 2, 1);
    else
      __imma_m16n16k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int *)&c, 2, 0);
  }
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __mma_bf16_m16n16k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __mma_bf16_m16n16k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);    
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __mma_bf16_m16n16k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);    
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) {
    __mma_bf16_m16n16k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);    
  }
#endif  /* __CUDA_AMPERE_MMA__ */


  // 
  // MMA functions for shape m32n8k16
  // 
  // D fp16, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 3, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 2, 0);
  }

  // D fp32, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, __half>& c) {
    __hmma_m32n8k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 2, 0);
  }

  // D fp32, C fp32
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);
  }

  // D fp16, C fp32
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, col_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b,32, 8, 16, __half, row_major>& b, const fragment<accumulator,32, 8, 16, float>& c) {
    __hmma_m32n8k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);
  }

#ifdef __CUDA_IMMA__  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, row_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 1);
    else
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, col_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 1);
    else
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, row_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 1);
    else
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, col_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 1);
    else
      __imma_m32n8k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, row_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 1);
    else
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, col_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 1);
    else
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 0);

  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, row_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 1);
    else
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 0);

  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, col_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf) {
    if (satf)
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 1);
    else
      __imma_m32n8k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 0);

  }
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) {
    __mma_bf16_m32n8k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }  
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) {
    __mma_bf16_m32n8k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);    
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) {
    __mma_bf16_m32n8k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);    
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) {
    __mma_bf16_m32n8k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);    
  }
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // MMA functions for shape m8n32k16
  // 
  // D fp16, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f16f16((int*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 2, 0);
  }

  // D fp32, C fp16
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 0, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, __half>& c) {
    __hmma_m8n32k16_mma_f32f16((float*)&d, (const int*)&a, (const int*)&b, (const int*)&c, 2, 0);
  }

  // D fp32, C fp32
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f32f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);
  }

  // D fp16, C fp32
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, col_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);
  }
    
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b,8, 32, 16, __half, row_major>& b, const fragment<accumulator,8, 32, 16, float>& c) {
    __hmma_m8n32k16_mma_f16f32((int*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);
  }

#ifdef __CUDA_IMMA__  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, row_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 1);
    else
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, col_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 1);
    else
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, row_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 1);
    else
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, col_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 1);
    else
      __imma_m8n32k16_mma_s8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, row_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 1);
    else
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, col_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 1);
    else
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, row_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 1);
    else
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, col_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf) {
    if (satf)
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 1);
    else
      __imma_m8n32k16_mma_u8((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 2, 0);
  }
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) {
    __mma_bf16_m8n32k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) {
    __mma_bf16_m8n32k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) {
    __mma_bf16_m8n32k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);    
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) {
    __mma_bf16_m8n32k16_mma_f32((float*)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);        
  }
#endif  /* __CUDA_AMPERE_MMA__ */


#ifdef __CUDA_SUBBYTE_IMMA__  
  // 
  // MMA functions for shape m8n8k32
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 32, int>& d, const fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major>& a, const fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major>& b, const fragment<accumulator, 8, 8, 32, int>& c, bool satf) {
    if (satf)
      __imma_m8n8k32_mma_s4((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 1);
    else
      __imma_m8n8k32_mma_s4((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 0);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 32, int>& d, const fragment<matrix_a, 8, 8, 32, experimental::precision::u4, row_major>& a, const fragment<matrix_b, 8, 8, 32, experimental::precision::u4, col_major>& b, const fragment<accumulator, 8, 8, 32, int>& c, bool satf) {
    if (satf)
      __imma_m8n8k32_mma_u4((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 1);
    else
      __imma_m8n8k32_mma_u4((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1, 0);
  }

  // 
  // MMA functions for shape m8n8k128
  // 
  __CUDA_MMA_DEVICE_DECL__  void bmma_sync(fragment<accumulator, 8, 8, 128, int>& d, const fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major>& a, const fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major>& b, const fragment<accumulator, 8, 8, 128, int>& c,
                                           experimental::bmmaBitOp op, experimental::bmmaAccumulateOp)
  {
     
#ifdef __CUDA_AMPERE_MMA__
    if (op == experimental::bmmaBitOpAND) 
      __bmma_m8n8k128_mma_and_popc_b1((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1);
    else 
#endif  /* __CUDA_AMPERE_MMA__ */      
      __bmma_m8n8k128_mma_xor_popc_b1((int*)&d, (const int *)&a, (const int *)&b, (const int*)&c, 1);
  }


#endif  /* __CUDA_SUBBYTE_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  // 
  // MMA functions for shape m16n16k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& b, const fragment<accumulator, 16, 16, 8, float>& c) {
    __mma_tf32_m16n16k8_mma_f32((float *)&d, (const int*)&a, (const int*)&b, (const float*)&c, 1, 0);    
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& b, const fragment<accumulator, 16, 16, 8, float>& c) {
    __mma_tf32_m16n16k8_mma_f32((float *)&d, (const int*)&a, (const int*)&b, (const float*)&c, 3, 0);    
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& b, const fragment<accumulator, 16, 16, 8, float>& c)  {
    __mma_tf32_m16n16k8_mma_f32((float *)&d, (const int*)&a, (const int*)&b, (const float*)&c, 0, 0);    
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& b, const fragment<accumulator, 16, 16, 8, float>& c)  {
    __mma_tf32_m16n16k8_mma_f32((float *)&d, (const int*)&a, (const int*)&b, (const float*)&c, 2, 0);    
  }

  
  // 
  // MMA functions for shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, row_major>& a, const fragment<matrix_b, 8, 8, 4, double, col_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) {
    __dmma_m8n8k4_mma_f64((double *)&d, (const double*)&a, (const double*)&b, (const double*)&c, 1, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, col_major>& a, const fragment<matrix_b, 8, 8, 4, double, col_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) {
    __dmma_m8n8k4_mma_f64((double *)&d, (const double*)&a, (const double*)&b, (const double*)&c, 3, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, row_major>& a, const fragment<matrix_b, 8, 8, 4, double, row_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) {
    __dmma_m8n8k4_mma_f64((double *)&d, (const double*)&a, (const double*)&b, (const double*)&c, 0, 0);
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, col_major>& a, const fragment<matrix_b, 8, 8, 4, double, row_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) {
    __dmma_m8n8k4_mma_f64((double *)&d, (const double*)&a, (const double*)&b, (const double*)&c, 2, 0);
  }
  
#endif  /* __CUDA_AMPERE_MMA__ */

};
};

#undef __CUDA_IMMA__
#undef __CUDA_SUBBYTE_IMMA__
#undef __CUDA_MMA_DEVICE_DECL__
#undef __CUDA_AMPERE_MMA__

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */


#endif   /* __CUDA_MMA_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_HPP__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_HPP__
#endif
