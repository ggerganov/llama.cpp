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
#pragma message("crt/mma.h is an internal header file and must not be used directly.  Please use mma.h instead.")
#else
#warning "crt/mma.h is an internal header file and must not be used directly.  Please use mma.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_H__
#endif

#if !defined(__CUDA_MMA_H__)
#define __CUDA_MMA_H__

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define __CUDA_MMA_DEVICE_DECL__ static __device__ __inline__

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700


#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

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
  
  // utility functions
#ifdef __CUDA_AMPERE_MMA__
  inline __device__ float __float_to_tf32(float in) 
  { 
    float ret; 
    asm("{\n  .reg .b32 __$1;"
        "\n   cvt.rna.tf32.f32 __$1, %1;"
        "\n   mov.b32 %0, __$1;\n}\n" : "=f"(ret) : "f"(in) ); 
    return ret; 
  }
#endif  /* __CUDA_AMPERE_MMA__ */  
  
  // 
  // tags 
  // 
  struct row_major;
  struct col_major;
  struct matrix_a;
  struct matrix_b;
  struct accumulator;

#ifdef __CUDA_AMPERE_MMA__
  namespace precision {
    struct tf32;
  }
#endif  /* __CUDA_AMPERE_MMA__ */  
#ifdef __CUDA_SUBBYTE_IMMA__
  namespace experimental {
    namespace precision {
      struct u4; // 4-bit unsigned
      struct s4; // 4-bit signed
      struct b1; // 1-bit
    }
    enum bmmaBitOp { bmmaBitOpXOR = 1
#ifdef __CUDA_AMPERE_MMA__
                    , bmmaBitOpAND = 2
#endif  /* __CUDA_AMPERE_MMA__ */
    };
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
  }
#endif  /* __CUDA_SUBBYTE_IMMA__ */

  // 
  // layout
  //
  enum layout_t {
    mem_row_major, mem_col_major
  };
  
  template <typename T>
  struct helper_traits {
    typedef T element_type;
    typedef T storage_element_type;
    typedef T fill_argument_type;
  };

#ifdef __CUDA_SUBBYTE_IMMA__
  template<> struct helper_traits<experimental::precision::u4> {
    typedef experimental::precision::u4 element_type;
    typedef unsigned int storage_element_type;
    typedef unsigned int fill_argument_type;
  };

  template<> struct helper_traits<experimental::precision::s4> {
    typedef experimental::precision::s4 element_type;
    typedef int storage_element_type;
    typedef int fill_argument_type;
  };
  
  template<> struct helper_traits<experimental::precision::b1> {
    typedef experimental::precision::b1 element_type;
    typedef unsigned int storage_element_type;
    typedef unsigned int fill_argument_type;
  };
#endif /* __CUDA_SUBBYTE_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  template<> struct helper_traits<precision::tf32> {
    typedef precision::tf32 element_type;
    typedef float storage_element_type;
    typedef float fill_argument_type;
  };
#endif  /* __CUDA_AMPERE_MMA__ */
  
  // 
  // The base fragment type
  // 
  /* note: alignment required for compiler implementation */
  template <typename T, int size, int packed_size = size> 
  struct __align__(8) __frag_base {

    /* Number of elements in the fragment */
    enum {num_elements = size};
    
    /* Number of storage elements in the fragment. 

       The elements of the fragment are packed together when the 
       fragment element type is experimental::precision::u4, 
       experimental::precision::s4 or experimental::precision::b1.
       When elements are packed, num_storage_elements 
       will be smaller than num_elements.
    */
    enum {num_storage_elements = packed_size};

    /* element type of the fragment */
    typedef T element_type;

    /* element type of the storage representation. 
    
       The mapping from element_type to storage_element_type is as follows:
       experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
       experimental::precision::s4 -> int (8 elements in 1 storage element)
       experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
       precision::tf32             -> float (1 element in 1 storage element)       
       all other types T           -> T
    */
    typedef typename helper_traits<T>::storage_element_type storage_element_type;

    /* Storage for the (possibly packed) fragment elements. */
    storage_element_type x[num_storage_elements];
  };

  template <typename FragEleType, typename StorageType, typename ArgType>
  static inline __device__ StorageType __get_storage_value(ArgType in) { return in; }

#ifdef __CUDA_SUBBYTE_IMMA__
  template<>
  __device__ inline unsigned 
  __get_storage_value<experimental::precision::u4, unsigned, unsigned>(unsigned in)
  {
    /* For experimental::precision::u4 fragment element type, pack 8 elements into a single 
       32-bit unsigned int storage element */
    unsigned val = in & 0xf;
    return (val | (val << 4) | (val << 8) | (val << 12) | (val << 16) |
            (val << 20) | (val << 24) | (val << 28));
  };

  template<>
  __device__ inline int
  __get_storage_value<experimental::precision::s4, int, int>(int in)
  {
    /* For experimental::precision::s4 fragment element type, pack 8 elements into a single 
       32-bit signed int storage element */
    int val = in & 0xf;
    return (val | (val << 4) | (val << 8) | (val << 12) | (val << 16) |
            (val << 20) | (val << 24) | (val << 28));
  };
  
  template<>
  __device__ inline unsigned 
  __get_storage_value<experimental::precision::b1, unsigned, unsigned>(unsigned in)
  {
    /* For experimental::precision::b1 fragment element type, pack 32 elements into a 
       single 32-bit unsigned int storage element */
    return (in & 0x1) ? 0xFFFFFFFFU : 0;
  }
#endif  /* __CUDA_SUBBYTE_IMMA__ */

  template <typename FragEleType, int size, int packed_size>
    __CUDA_MMA_DEVICE_DECL__ void fill_fragment(__frag_base<FragEleType, size, packed_size>& f, 
       /*  The mapping from fragment element type (FragEleType) to fill_argument_type is:
       experimental::precision::u4 -> unsigned (only lower 4 bits taken)
       experimental::precision::s4 -> int (only lower 4 bits taken)
       experimental::precision::b1 -> unsigned (only lowest 1 bit taken)
       precision::tf32             -> float
       all other types T           -> T
       */        
   const typename helper_traits<FragEleType>::fill_argument_type & in) {

   /* get the (possibly packed) storage element value. See the specializations above for fragment
      element types where the storage representation is packed */
   typedef typename helper_traits<FragEleType>::storage_element_type storage_type;
   storage_type v = __get_storage_value<FragEleType, storage_type>(in);
#pragma unroll
    for (int i=0; i< f.num_storage_elements; i++)
      f.x[i] = v; 
  }
  
  // 
  // Fragment template
  // 
  template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

  // 
  // Fragments for 16x16x16
  // 
  template<> class fragment<matrix_a, 16, 16, 16, __half, row_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_a, 16, 16, 16, __half, col_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_b, 16, 16, 16, __half, row_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_b, 16, 16, 16, __half, col_major> : public __frag_base<__half, 16> {};
  template<> class fragment<accumulator, 16, 16, 16, __half> : public __frag_base<__half, 8> {};
  template<> class fragment<accumulator, 16, 16, 16, float> : public __frag_base<float, 8> {};

#ifdef __CUDA_IMMA__
  template<> class fragment<matrix_a, 16, 16, 16, signed char, row_major> : public __frag_base<signed char, 8> {};
  template<> class fragment<matrix_a, 16, 16, 16, signed char, col_major> : public __frag_base<signed char, 8> {};
  template<> class fragment<matrix_a, 16, 16, 16, unsigned char, row_major> : public __frag_base<unsigned char, 8> {};
  template<> class fragment<matrix_a, 16, 16, 16, unsigned char, col_major> : public __frag_base<unsigned char, 8> {};
  template<> class fragment<matrix_b, 16, 16, 16, signed char, row_major> : public __frag_base<signed char, 8> {};
  template<> class fragment<matrix_b, 16, 16, 16, signed char, col_major> : public __frag_base<signed char, 8> {};  
  template<> class fragment<matrix_b, 16, 16, 16, unsigned char, row_major> : public __frag_base<unsigned char, 8> {};
  template<> class fragment<matrix_b, 16, 16, 16, unsigned char, col_major> : public __frag_base<unsigned char, 8> {};  
  template<> class fragment<accumulator, 16, 16, 16, int> : public __frag_base<int, 8> {};
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  template<> class fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> : public __frag_base<__nv_bfloat16, 8> {};
  template<> class fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> : public __frag_base<__nv_bfloat16, 8> {};
  template<> class fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> : public __frag_base<__nv_bfloat16, 8> {};
  template<> class fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> : public __frag_base<__nv_bfloat16, 8> {};
#endif  /* __CUDA_AMPERE_MMA__ */
  
  // 
  // Fragments for 32x8x16
  // 
  template<> class fragment<matrix_a, 32, 8, 16, __half, row_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_a, 32, 8, 16, __half, col_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_b, 32, 8, 16, __half, row_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_b, 32, 8, 16, __half, col_major> : public __frag_base<__half, 16> {};
  template<> class fragment<accumulator, 32, 8, 16, __half> : public __frag_base<__half, 8> {};
  template<> class fragment<accumulator, 32, 8, 16, float> : public __frag_base<float, 8> {};

#ifdef __CUDA_IMMA__
  template<> class fragment<matrix_a, 32, 8, 16, signed char, row_major> : public __frag_base<signed char, 16> {};
  template<> class fragment<matrix_a, 32, 8, 16, signed char, col_major> : public __frag_base<signed char, 16> {};
  template<> class fragment<matrix_a, 32, 8, 16, unsigned char, row_major> : public __frag_base<unsigned char, 16> {};
  template<> class fragment<matrix_a, 32, 8, 16, unsigned char, col_major> : public __frag_base<unsigned char, 16> {};
  template<> class fragment<matrix_b, 32, 8, 16, signed char, row_major> : public __frag_base<signed char, 4> {};
  template<> class fragment<matrix_b, 32, 8, 16, signed char, col_major> : public __frag_base<signed char, 4> {};
  template<> class fragment<matrix_b, 32, 8, 16, unsigned char, row_major> : public __frag_base<unsigned char, 4> {};
  template<> class fragment<matrix_b, 32, 8, 16, unsigned char, col_major> : public __frag_base<unsigned char, 4> {};
  template<> class fragment<accumulator, 32, 8, 16, int> : public __frag_base<int, 8> {};
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  template<> class fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major> : public __frag_base<__nv_bfloat16, 16> {};
  template<> class fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major> : public __frag_base<__nv_bfloat16, 16> {};
  template<> class fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major> : public __frag_base<__nv_bfloat16, 4> {};
  template<> class fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major> : public __frag_base<__nv_bfloat16, 4> {};
#endif  /* __CUDA_AMPERE_MMA__ */
  
  // 
  // Fragments for 8x32x16
  // 
  template<> class fragment<matrix_a, 8, 32, 16, __half, row_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_a, 8, 32, 16, __half, col_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_b, 8, 32, 16, __half, row_major> : public __frag_base<__half, 16> {};
  template<> class fragment<matrix_b, 8, 32, 16, __half, col_major> : public __frag_base<__half, 16> {};
  template<> class fragment<accumulator, 8, 32, 16, __half> : public __frag_base<__half, 8> {};
  template<> class fragment<accumulator, 8, 32, 16, float> : public __frag_base<float, 8> {};

#ifdef __CUDA_IMMA__
  template<> class fragment<matrix_a, 8, 32, 16, signed char, row_major> : public __frag_base<signed char, 4> {};
  template<> class fragment<matrix_a, 8, 32, 16, signed char, col_major> : public __frag_base<signed char, 4> {};
  template<> class fragment<matrix_a, 8, 32, 16, unsigned char, row_major> : public __frag_base<unsigned char, 4> {};
  template<> class fragment<matrix_a, 8, 32, 16, unsigned char, col_major> : public __frag_base<unsigned char, 4> {};
  template<> class fragment<matrix_b, 8, 32, 16, signed char, row_major> : public __frag_base<signed char, 16> {};
  template<> class fragment<matrix_b, 8, 32, 16, signed char, col_major> : public __frag_base<signed char, 16> {};
  template<> class fragment<matrix_b, 8, 32, 16, unsigned char, row_major> : public __frag_base<unsigned char, 16> {};
  template<> class fragment<matrix_b, 8, 32, 16, unsigned char, col_major> : public __frag_base<unsigned char, 16> {};
  template<> class fragment<accumulator, 8, 32, 16, int> : public __frag_base<int, 8> {};
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  template<> class fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major> : public __frag_base<__nv_bfloat16, 4> {};
  template<> class fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major> : public __frag_base<__nv_bfloat16, 4> {};
  template<> class fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major> : public __frag_base<__nv_bfloat16, 16> {};
  template<> class fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major> : public __frag_base<__nv_bfloat16, 16> {};
#endif  /* __CUDA_AMPERE_MMA__ */  
  
#ifdef __CUDA_SUBBYTE_IMMA__
  // 
  // Fragments for 8x8x32
  // 
  template<> class fragment<matrix_a, 8, 8, 32, experimental::precision::u4, row_major> : public __frag_base<experimental::precision::u4, 8, 1> {};
  template<> class fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> : public __frag_base<experimental::precision::s4, 8, 1> {};
  template<> class fragment<matrix_b, 8, 8, 32, experimental::precision::u4, col_major> : public __frag_base<experimental::precision::u4, 8, 1> {};
  template<> class fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> : public __frag_base<experimental::precision::s4, 8, 1> {};
  template<> class fragment<accumulator, 8, 8, 32, int> : public __frag_base<int, 2> {};

  // 
  // Fragments for 8x8x128
  // 
  template<> class fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major> : public __frag_base<experimental::precision::b1, 32, 1> {};
  template<> class fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major> : public __frag_base<experimental::precision::b1, 32, 1> {};
  template<> class fragment<accumulator, 8, 8, 128, int> : public __frag_base<int, 2> {};
#endif  /* __CUDA_SUBBYTE_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  //
  // Fragments for 16x16x8
  //
  template<> class fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> : public __frag_base<precision::tf32, 4> {};
  template<> class fragment<matrix_a, 16, 16, 8, precision::tf32, col_major> : public __frag_base<precision::tf32, 4> {};
  template<> class fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> : public __frag_base<precision::tf32, 4> {};
  template<> class fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> : public __frag_base<precision::tf32, 4> {};
  template<> class fragment<accumulator, 16, 16, 8, float> : public __frag_base<float, 8> {};
  
  //
  // Fragments for 8x8x4
  //
  template<> class fragment<matrix_a, 8, 8, 4, double, row_major> : public __frag_base<double, 1> {};
  template<> class fragment<matrix_a, 8, 8, 4, double, col_major> : public __frag_base<double, 1> {};
  template<> class fragment<matrix_b, 8, 8, 4, double, row_major> : public __frag_base<double, 1> {};
  template<> class fragment<matrix_b, 8, 8, 4, double, col_major> : public __frag_base<double, 1> {};
  template<> class fragment<accumulator, 8, 8, 4, double> : public __frag_base<double, 2> {};
#endif  /* __CUDA_AMPERE_MMA__ */  

  
  // 
  // Load functions for frags of shape m16n16k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, float>& a, const float* p, unsigned ldm, layout_t layout) __DEF_IF_HOST

#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, int>& a, const int* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */
  
#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

  //
  // Load functions for frags of shape m32n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 8, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 8, 16, float>& a, const float* p, unsigned ldm, layout_t layout) __DEF_IF_HOST

#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 8, 16, int>& a, const int* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

  //
  // Load functions for frags of shape m8n32k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 32, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 32, 16, float>& a, const float* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
  
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 32, 16, int>& a, const int* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major>& a, const __nv_bfloat16* p, unsigned ldm) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

#ifdef __CUDA_SUBBYTE_IMMA__
  //
  // Load functions for frags of shape m8n8k32
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major>& a, const void* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 32, experimental::precision::u4, row_major>& a, const void* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major>& a, const void* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 32, experimental::precision::u4, col_major>& a, const void* p, unsigned ldm) __DEF_IF_HOST
  
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 32, int>& a, const int* p, unsigned ldm, layout_t layout) __DEF_IF_HOST

  //
  // Load functions for frags of shape m8n8k128
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major>& a, const void* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major>& a, const void* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 128, int>& a, const int* p, unsigned ldm, layout_t layout) __DEF_IF_HOST

#endif  /* __CUDA_SUBBYTE_IMMA__ */


#ifdef __CUDA_AMPERE_MMA__
  //
  // Load functions for frags of shape m16n16k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const float* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const float* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& a, const float* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& a, const float* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 8, float>& a, const float* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
  
  //
  // Load functions for frags of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, double, row_major>& a, const double* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 8, 8, 4, double, col_major>& a, const double* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, double, row_major>& a, const double* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 8, 8, 4, double, col_major>& a, const double* p, unsigned ldm) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 8, 8, 4, double>& a, const double* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // Store functions for frags of shape m16n16k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 16, 16, 16, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 16, float>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 16, 16, 16, int>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */  

  // 
  // Store functions for frags of shape m32n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 32, 8, 16, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 32, 8, 16, float>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 32, 8, 16, int>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

  // 
  // Store functions for frags of shape m8n32k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 8, 32, 16, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 8, 32, 16, float>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#ifdef __CUDA_IMMA__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 8, 32, 16, int>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_SUBBYTE_IMMA__
  // 
  // Store functions for frags of shape m8n8k32
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 8, 8, 32, int>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST

  // 
  // Store functions for frags of shape m8n8k128
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(int *p, const fragment<accumulator, 8, 8, 128, int>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST

#endif  /* __CUDA_SUBBYTE_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  //
  // Store functions for frags of shape m16n16k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 8, float>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST

  //
  // Store functions for frags of shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(double *p, const fragment<accumulator, 8, 8, 4, double>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // MMA functions for shape m16n16k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST

#ifdef __CUDA_IMMA__  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, row_major>& a, const fragment<matrix_b,16, 16, 16, signed char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, col_major>& a, const fragment<matrix_b,16, 16, 16, signed char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, row_major>& a, const fragment<matrix_b,16, 16, 16, signed char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, signed char, col_major>& a, const fragment<matrix_b,16, 16, 16, signed char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, row_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, col_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, col_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, row_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, int>& d, const fragment<matrix_a, 16, 16, 16, unsigned char, col_major>& a, const fragment<matrix_b,16, 16, 16, unsigned char, row_major>& b, const fragment<accumulator,16, 16, 16, int>& c, bool satf=false) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b,16, 16, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // MMA functions for shape m32n8k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, row_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, __half>& d, const fragment<matrix_a, 32, 8, 16, __half, col_major>& a, const fragment<matrix_b, 32, 8, 16, __half, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST

#ifdef __CUDA_IMMA__  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, row_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, col_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, row_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, signed char, col_major>& a, const fragment<matrix_b, 32, 8, 16, signed char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, row_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, col_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, col_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, row_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, int>& d, const fragment<matrix_a, 32, 8, 16, unsigned char, col_major>& a, const fragment<matrix_b, 32, 8, 16, unsigned char, row_major>& b, const fragment<accumulator, 32, 8, 16, int>& c, bool satf=false) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 8, 16, float>& d, const fragment<matrix_a, 32, 8, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 32, 8, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 32, 8, 16, float>& c) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

  // 
  // MMA functions for shape m8n32k16
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, __half>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, row_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, __half>& d, const fragment<matrix_a, 8, 32, 16, __half, col_major>& a, const fragment<matrix_b, 8, 32, 16, __half, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  
#ifdef __CUDA_IMMA__  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, row_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, col_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, row_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, signed char, col_major>& a, const fragment<matrix_b, 8, 32, 16, signed char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, row_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, col_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, col_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, row_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, int>& d, const fragment<matrix_a, 8, 32, 16, unsigned char, col_major>& a, const fragment<matrix_b, 8, 32, 16, unsigned char, row_major>& b, const fragment<accumulator, 8, 32, 16, int>& c, bool satf=false) __DEF_IF_HOST
#endif  /* __CUDA_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, col_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, row_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 32, 16, float>& d, const fragment<matrix_a, 8, 32, 16, __nv_bfloat16, col_major>& a, const fragment<matrix_b, 8, 32, 16, __nv_bfloat16, row_major>& b, const fragment<accumulator, 8, 32, 16, float>& c) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */

#ifdef __CUDA_SUBBYTE_IMMA__  
  // 
  // MMA functions for shape m8n8k32
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 32, int>& d, const fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major>& a, const fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major>& b, const fragment<accumulator, 8, 8, 32, int>& c, bool satf=false) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 32, int>& d, const fragment<matrix_a, 8, 8, 32, experimental::precision::u4, row_major>& a, const fragment<matrix_b, 8, 8, 32, experimental::precision::u4, col_major>& b, const fragment<accumulator, 8, 8, 32, int>& c, bool satf=false) __DEF_IF_HOST
  

  // 
  // MMA functions for shape m8n8k128
  // 
  __CUDA_MMA_DEVICE_DECL__ void bmma_sync(fragment<accumulator, 8, 8, 128, int>& d, const fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major>& a, const fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major>& b, const fragment<accumulator, 8, 8, 128, int>& c,
                                          experimental::bmmaBitOp = experimental::bmmaBitOpXOR, 
                                          experimental::bmmaAccumulateOp = experimental::bmmaAccumulateOpPOPC) __DEF_IF_HOST

#endif  /* __CUDA_SUBBYTE_IMMA__ */

#ifdef __CUDA_AMPERE_MMA__
  // 
  // MMA functions for shape m16n16k8
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& b, const fragment<accumulator, 16, 16, 8, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& b, const fragment<accumulator, 16, 16, 8, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& b, const fragment<accumulator, 16, 16, 8, float>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 8, float>& d, const fragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const fragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& b, const fragment<accumulator, 16, 16, 8, float>& c) __DEF_IF_HOST

  // 
  // MMA functions for shape m8n8k4
  // 
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, row_major>& a, const fragment<matrix_b, 8, 8, 4, double, col_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, col_major>& a, const fragment<matrix_b, 8, 8, 4, double, col_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, row_major>& a, const fragment<matrix_b, 8, 8, 4, double, row_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) __DEF_IF_HOST
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 8, 8, 4, double>& d, const fragment<matrix_a, 8, 8, 4, double, col_major>& a, const fragment<matrix_b, 8, 8, 4, double, row_major>& b, const fragment<accumulator, 8, 8, 4, double>& c) __DEF_IF_HOST
#endif  /* __CUDA_AMPERE_MMA__ */
};
};

#undef __DEF_IF_HOST
#undef __CUDA_IMMA__
#undef __CUDA_SUBBYTE_IMMA__
#undef __CUDA_AMPERE_MMA__
#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 700 */

#endif /* __cplusplus && __CUDACC__ */

#undef __CUDA_MMA_DEVICE_DECL__

#if defined(__CUDA_ARCH__)
#include "mma.hpp"
#endif /* defined(__CUDA_ARCH__) */


#endif /* !__CUDA_MMA_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_H__
#endif
