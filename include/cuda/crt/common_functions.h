/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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
#pragma message("crt/common_functions.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/common_functions.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_COMMON_FUNCTIONS_H__
#endif

#if !defined(__COMMON_FUNCTIONS_H__)
#define __COMMON_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

#include "builtin_types.h"
#include "host_defines.h"

#define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported.  Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."

#ifndef __CUDA_API_VER_MAJOR__
#define __CUDA_API_VER_MAJOR__ __CUDACC_VER_MAJOR__
#endif /* __CUDA_API_VER_MAJOR__ */

#ifndef __CUDA_API_VER_MINOR__
#define __CUDA_API_VER_MINOR__ __CUDACC_VER_MINOR__
#endif /* __CUDA_API_VER_MINOR__ */

#if !defined(__CUDACC_RTC__)
#include <string.h>
#include <time.h>

extern "C"
{
#endif /* !__CUDACC_RTC__ */
extern _CRTIMP __host__ __device__ __device_builtin__ __cudart_builtin__ clock_t __cdecl clock(void)
#if defined(__QNX__)
asm("clock32")
#endif
__THROW;
extern         __host__ __device__ __device_builtin__ __cudart_builtin__ void*   __cdecl memset(void*, int, size_t) __THROW;
extern         __host__ __device__ __device_builtin__ __cudart_builtin__ void*   __cdecl memcpy(void*, const void*, size_t) __THROW;
#if !defined(__CUDACC_RTC__)
}
#endif /* !__CUDACC_RTC__ */

#if defined(__CUDA_ARCH__)

#if defined(__CUDACC_RTC__)
inline __host__ __device__ void* operator new(size_t, void *p) { return p; }
inline __host__ __device__ void* operator new[](size_t, void *p) { return p; }
inline __host__ __device__ void operator delete(void*, void*) { }
inline __host__ __device__ void operator delete[](void*, void*) { }
#else /* !__CUDACC_RTC__ */
#ifndef __CUDA_INTERNAL_SKIP_CPP_HEADERS__
#include <new>
#endif

#if defined (__GNUC__)

#define STD \
        std::
        
#else /* __GNUC__ */

#define STD

#endif /* __GNUC__ */

extern         __host__ __device__ __cudart_builtin__ void*   __cdecl operator new(STD size_t, void*) throw();
extern         __host__ __device__ __cudart_builtin__ void*   __cdecl operator new[](STD size_t, void*) throw();
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete(void*, void*) throw();
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete[](void*, void*) throw();
# if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900) || defined(__CUDA_XLC_CPP14__) || defined(__CUDA_ICC_CPP14__)
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete(void*, STD size_t) throw();
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete[](void*, STD size_t) throw();
#endif /* __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900) || defined(__CUDA_XLC_CPP14__)  || defined(__CUDA_ICC_CPP14__) */
#endif /* __CUDACC_RTC__ */

#if !defined(__CUDACC_RTC__)
#include <stdio.h>
#include <stdlib.h>
#endif /* !__CUDACC_RTC__ */

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
extern "C"
{
extern
#if !defined(_MSC_VER) || _MSC_VER < 1900
_CRTIMP
#endif
            
#if defined(__GLIBC__) && defined(__GLIBC_MINOR__) && ( (__GLIBC__ < 2) || ( (__GLIBC__ == 2) && (__GLIBC_MINOR__ < 3) ) ) 
__host__ __device__ __device_builtin__ __cudart_builtin__ int     __cdecl printf(const char*, ...) __THROW;
#else /* newer glibc */
__host__ __device__ __device_builtin__ __cudart_builtin__ int     __cdecl printf(const char*, ...);
#endif /* defined(__GLIBC__) && defined(__GLIBC_MINOR__) && ( (__GLIBC__ < 2) || ( (__GLIBC__ == 2) && (__GLIBC_MINOR__ < 3) ) ) */


extern _CRTIMP __host__ __device__ __cudart_builtin__ void*   __cdecl malloc(size_t) __THROW;
extern _CRTIMP __host__ __device__ __cudart_builtin__ void    __cdecl free(void*) __THROW;

#if defined(_MSC_VER)
extern  __host__ __device__ __cudart_builtin__ void*   __cdecl _alloca(size_t);
#endif

#if defined(__QNX__)
#undef alloca
#define alloca(__S) __builtin_alloca(__S)
#endif
}
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

#if !defined(__CUDACC_RTC__)
#include <assert.h>
#endif /* !__CUDACC_RTC__ */

extern "C"
{
#if defined(__CUDACC_RTC__)
extern __host__ __device__ void __assertfail(const char * __assertion, 
                                             const char *__file,
                                             unsigned int __line,
                                             const char *__function,
                                             size_t charsize);
#elif defined(__APPLE__)
#define __builtin_expect(exp,c) (exp)
extern __host__ __device__ __cudart_builtin__ void __assert_rtn(
  const char *, const char *, int, const char *);
#elif defined(__ANDROID__)
extern __host__ __device__ __cudart_builtin__ void __assert2(
  const char *, int, const char *, const char *);
#elif defined(__QNX__)
#if !defined(_LIBCPP_VERSION)
namespace std {
#endif
extern __host__ __device__ __cudart_builtin__ void __assert(
  const char *, const char *, unsigned int, const char *);
#if !defined(_LIBCPP_VERSION)
}
#endif
#elif defined(__HORIZON__)
extern __host__ __device__ __cudart_builtin__ void __assert_fail(
  const char *, const char *, int, const char *);
#elif defined(__GNUC__)
extern __host__ __device__ __cudart_builtin__ void __assert_fail(
  const char *, const char *, unsigned int, const char *)
  __THROW; 
#elif defined(_WIN32)
extern __host__ __device__ __cudart_builtin__ _CRTIMP void __cdecl _wassert(
  const wchar_t *, const wchar_t *, unsigned);
#endif
}

#if defined(__CUDACC_RTC__)
#ifdef NDEBUG
#define assert(e) (static_cast<void>(0))
#else /* !NDEBUG */
#define __ASSERT_STR_HELPER(x) #x
#define assert(e) ((e) ? static_cast<void>(0)\
                       : __assertfail(__ASSERT_STR_HELPER(e), __FILE__,\
                                      __LINE__, __PRETTY_FUNCTION__,\
                                      sizeof(char)))
#endif /* NDEBUG */
__host__ __device__  void* operator new(size_t);
__host__ __device__  void* operator new[](size_t);
__host__ __device__  void operator delete(void*);
__host__ __device__  void operator delete[](void*);
# if __cplusplus >= 201402L
__host__ __device__  void operator delete(void*, size_t);
__host__ __device__  void operator delete[](void*, size_t);
#endif /* __cplusplus >= 201402L */

#if __cplusplus >= 201703L
namespace std { enum class align_val_t : size_t {}; }
__host__ __device__ void*   __cdecl operator new(size_t sz, std::align_val_t) noexcept;
__host__ __device__ void*   __cdecl operator new[](size_t sz, std::align_val_t) noexcept;
__host__ __device__ void    __cdecl operator delete(void* ptr, std::align_val_t) noexcept;
__host__ __device__ void    __cdecl operator delete[](void* ptr, std::align_val_t) noexcept;
__host__ __device__ void    __cdecl operator delete(void* ptr, size_t, std::align_val_t) noexcept;
__host__ __device__ void    __cdecl operator delete[](void* ptr, size_t, std::align_val_t) noexcept;
#endif  /* __cplusplus >= 201703L */

#else /* !__CUDACC_RTC__ */
#if defined (__GNUC__)

#define __NV_GLIBCXX_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) 

#if (__cplusplus >= 201103L)  && ((!(defined(__QNX__) && defined(_LIBCPP_VERSION))) || (defined(__QNX__) && __NV_GLIBCXX_VERSION >= 80300))
#define THROWBADALLOC 
#else
#if defined(__ANDROID__) && !defined(_LIBCPP_VERSION) && (defined(__BIONIC__) || __NV_GLIBCXX_VERSION < 40900)
#define THROWBADALLOC
#else
#define THROWBADALLOC  throw(STD bad_alloc)
#endif
#endif
#define __DELETE_THROW throw()

#undef __NV_GLIBCXX_VERSION

#else /* __GNUC__ */

#define THROWBADALLOC  throw(...)

#endif /* __GNUC__ */

extern         __host__ __device__ __cudart_builtin__ void*   __cdecl operator new(STD size_t) THROWBADALLOC;
extern         __host__ __device__ __cudart_builtin__ void*   __cdecl operator new[](STD size_t) THROWBADALLOC;
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete(void*) throw();
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete[](void*) throw();
# if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900) || defined(__CUDA_XLC_CPP14__) || defined(__CUDA_ICC_CPP14__)
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete(void*, STD size_t) throw();
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete[](void*, STD size_t) throw();
#endif /* __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900) || defined(__CUDA_XLC_CPP14__) || defined(__CUDA_ICC_CPP14__)  */

#if __cpp_aligned_new
extern         __host__ __device__ __cudart_builtin__ void*   __cdecl operator new(STD size_t, std::align_val_t);
extern         __host__ __device__ __cudart_builtin__ void*   __cdecl operator new[](STD size_t, std::align_val_t);
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete(void*, std::align_val_t) noexcept;
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete[](void*, std::align_val_t) noexcept;
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete(void*, STD size_t, std::align_val_t) noexcept;
extern         __host__ __device__ __cudart_builtin__ void    __cdecl operator delete[](void*, STD size_t, std::align_val_t) noexcept;
#endif  /* __cpp_aligned_new */

#undef THROWBADALLOC
#undef STD
#endif /* __CUDACC_RTC__ */

#endif /* __CUDA_ARCH__ */

#endif /* __cplusplus && __CUDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__) && (__CUDA_ARCH__ >= 350)
#include "cuda_device_runtime_api.h"
#endif

#include "math_functions.h"

#endif /* !__COMMON_FUNCTIONS_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_COMMON_FUNCTIONS_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_COMMON_FUNCTIONS_H__
#endif
