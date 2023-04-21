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
#pragma message("crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/host_defines.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif

#if !defined(__HOST_DEFINES_H__)
#define __HOST_DEFINES_H__

/* CUDA JIT mode (__CUDACC_RTC__) also uses GNU style attributes */
#if defined(__GNUC__) || (defined(__PGIC__) && defined(__linux__)) || defined(__CUDA_LIBDEVICE__) || defined(__CUDACC_RTC__)

#if defined(__CUDACC_RTC__)
#define __volatile__ volatile
#endif /* __CUDACC_RTC__ */

#define __no_return__ \
        __attribute__((noreturn))
        
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)
/* gcc allows users to define attributes with underscores, 
   e.g., __attribute__((__noinline__)).
   Consider a non-CUDA source file (e.g. .cpp) that has the 
   above attribute specification, and includes this header file. In that case,
   defining __noinline__ as below  would cause a gcc compilation error.
   Hence, only define __noinline__ when the code is being processed
   by a  CUDA compiler component.
*/   
#define __noinline__ \
        __attribute__((noinline))
#endif /* __CUDACC__  || __CUDA_ARCH__ || __CUDA_LIBDEVICE__ */
        
#define __forceinline__ \
        __inline__ __attribute__((always_inline))
#define __align__(n) \
        __attribute__((aligned(n)))
#define __thread__ \
        __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a) \
        __attribute__((a))
#define __location__(a) \
        __annotate__(a)
#define CUDARTAPI
#define CUDARTAPI_CDECL

#elif defined(_MSC_VER)

#if _MSC_VER >= 1400

#define __restrict__ \
        __restrict

#else /* _MSC_VER >= 1400 */

#define __restrict__

#endif /* _MSC_VER >= 1400 */

#define __inline__ \
        __inline
#define __no_return__ \
        __declspec(noreturn)
#define __noinline__ \
        __declspec(noinline)
#define __forceinline__ \
        __forceinline
#define __align__(n) \
        __declspec(align(n))
#define __thread__ \
        __declspec(thread)
#define __import__ \
        __declspec(dllimport)
#define __export__ \
        __declspec(dllexport)
#define __annotate__(a) \
        __declspec(a)
#define __location__(a) \
        __annotate__(__##a##__)
#define CUDARTAPI \
        __stdcall
#define CUDARTAPI_CDECL \
        __cdecl

#else /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#define __inline__

#if !defined(__align__)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for '__align__' !!! ---

#endif /* !__align__ */

#if !defined(CUDARTAPI)

#error --- !!! UNKNOWN COMPILER: please provide a CUDA compatible definition for 'CUDARTAPI' !!! ---

#endif /* !CUDARTAPI */

#endif /* __GNUC__ || __CUDA_LIBDEVICE__ || __CUDACC_RTC__ */

#if (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !defined(__clang__)))) || \
    (defined(_MSC_VER) && _MSC_VER < 1900) || \
    (!defined(__GNUC__) && !defined(_MSC_VER))

#define __specialization_static \
        static

#else /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#define __specialization_static

#endif /* (__GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3 && !__clang__))) ||
         (_MSC_VER && _MSC_VER < 1900) ||
         (!__GNUC__ && !_MSC_VER) */

#if !defined(__CUDACC__) && !defined(__CUDA_LIBDEVICE__)

#undef __annotate__
#define __annotate__(a)

#else /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#define __launch_bounds__(...) \
        __annotate__(launch_bounds(__VA_ARGS__))

#endif /* !__CUDACC__ && !__CUDA_LIBDEVICE__ */

#if defined(__CUDACC__) || defined(__CUDA_LIBDEVICE__) || \
    defined(__GNUC__) || defined(_WIN64)

#define __builtin_align__(a) \
        __align__(a)

#else /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__ || _WIN64 */

#define __builtin_align__(a)

#endif /* __CUDACC__ || __CUDA_LIBDEVICE__ || __GNUC__  || _WIN64 */

#if defined(__CUDACC__) || !defined(__host__)
#define __host__ \
        __location__(host)
#endif /* defined(__CUDACC__) || !defined(__host__) */
#if defined(__CUDACC__) || !defined(__device__)
#define __device__ \
        __location__(device)
#endif /* defined(__CUDACC__) || !defined(__device__) */
#if defined(__CUDACC__) || !defined(__global__)
#define __global__ \
        __location__(global)
#endif /* defined(__CUDACC__) || !defined(__global__) */
#if defined(__CUDACC__) || !defined(__shared__)
#define __shared__ \
        __location__(shared)
#endif /* defined(__CUDACC__) || !defined(__shared__) */
#if defined(__CUDACC__) || !defined(__constant__)
#define __constant__ \
        __location__(constant)
#endif /* defined(__CUDACC__) || !defined(__constant__) */
#if defined(__CUDACC__) || !defined(__managed__)
#define __managed__ \
        __location__(managed)
#endif /* defined(__CUDACC__) || !defined(__managed__) */
        
#if !defined(__CUDACC__)
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#else /* defined(__CUDACC__) */
#define __device_builtin__ \
        __location__(device_builtin)
#define __device_builtin_texture_type__ \
        __location__(device_builtin_texture_type)
#define __device_builtin_surface_type__ \
        __location__(device_builtin_surface_type)
#define __cudart_builtin__ \
        __location__(cudart_builtin)
#endif /* !defined(__CUDACC__) */


#endif /* !__HOST_DEFINES_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_DEFINES_H__
#endif
