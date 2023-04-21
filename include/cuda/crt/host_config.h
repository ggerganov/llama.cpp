/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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
#pragma message("crt/host_config.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/host_config.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_CONFIG_H__
#endif

#if !defined(__HOST_CONFIG_H__)
#define __HOST_CONFIG_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC__)

#if defined(__CUDACC_RTC__)

#define _CRTIMP
#define __THROW

#else /* __CUDACC_RTC__ */

/* check for host compilers that are compatible with nvcc */
#if !defined(__GNUC__) && !defined(_WIN32)

#error --- !!! UNSUPPORTED COMPILER !!! ---

#endif /* !__GNUC__ && !_WIN32 */

/* check invalid configurations */
#if defined(__PGIC__)
#if !defined(__GNUC__) || !defined(__LP64__) || !defined(__linux__)
#error -- unsupported pgc++ configuration! pgc++ is supported only on Linux x86_64!
#endif /* !defined(__GNUC__) || !defined(__LP64__) || !defined(__linux__) */
#endif  /* defined(__PGIC__) */

#if defined(__powerpc__)
#if !defined(__powerpc64__) || !defined(__LITTLE_ENDIAN__)
#error -- unsupported PPC platform! Only 64-bit little endian PPC is supported!
#endif /* !__powerpc64__ || !__LITTLE_ENDIAN__ */
#endif /* __powerpc__ */

#if defined(__APPLE__) && defined(__MACH__) && !defined(__clang__)
#error -- clang and clang++ are the only supported host compilers on Mac OS X!
#endif /* __APPLE__ && __MACH__ && !__clang__ */


/* check host compiler version  */
#if !__NV_NO_HOST_COMPILER_CHECK

#if defined(__ICC)

#if (__ICC != 1500 && __ICC != 1600 && __ICC != 1700 && __ICC != 1800 && !(__ICC >= 1900 && __ICC <= 2021)) || !defined(__GNUC__) || !defined(__LP64__)

#error -- unsupported ICC configuration! Only ICC 15.0, ICC 16.0, ICC 17.0, ICC 18.0, ICC 19.x and 20.x on Linux x86_64 are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
 
#endif /* (__ICC != 1500 && __ICC != 1600 && __ICC != 1700 && __ICC != 1800 && __ICC != 1900) || !__GNUC__ || !__LP64__ */

#endif /* __ICC */

#if defined(__powerpc__)

#if defined(__ibmxl_vrm__) && !(__ibmxl_vrm__ >= 0x0d010000 && __ibmxl_vrm__ < 0x0d020000) && \
                              !(__ibmxl_vrm__ >= 0x10010000 && __ibmxl_vrm__ < 0x10020000)

#error -- unsupported xlC version! only xlC 13.1 and 16.1 are supported. The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

#endif /* __ibmxl_vrm__ && !(__ibmxl_vrm__ >= 0x0d010000 && __ibmxl_vrm__ < 0x0d020000) &&
                           !(__ibmxl_vrm__ >= 0x10010000 && __ibmxl_vrm__ < 0x10020000) */

#endif /* __powerpc__ */

#if defined(__GNUC__)

#if __GNUC__ > 11

#error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

#endif /* __GNUC__ > 11 */


#if defined(__clang__) && !defined(__ibmxl_vrm__) && !defined(__ICC) && !defined(__HORIZON__) && !defined(__APPLE__)

#if (__clang_major__ >= 14) || (__clang_major__ < 3) || ((__clang_major__ == 3) &&  (__clang_minor__ < 3))
#error -- unsupported clang version! clang version must be less than 14 and greater than 3.2 . The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

#endif  /* (__clang_major__ >=  14) || (__clang_major__ < 3) || ((__clang_major__ == 3) &&  (__clang_minor__ < 3)) */

#endif /* defined(__clang__) && !defined(__ibmxl_vrm__) && !defined(__ICC) && !defined(__HORIZON__) && !defined(__APPLE__) */


#endif /* __GNUC__ */

#if defined(_WIN32)

#if _MSC_VER < 1910 || _MSC_VER >= 1940

#error -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

#elif _MSC_VER >= 1910 && _MSC_VER < 1910

#pragma message("support for this version of Microsoft Visual Studio has been deprecated! Only the versions between 2017 and 2022 (inclusive) are supported!")

#endif /* (_MSC_VER < 1910 || _MSC_VER >= 1940) || (_MSC_VER >= 1910 && _MSC_VER < 1910) */

#endif /* _WIN32 */
#endif  /* !__NV_NO_HOST_COMPILER_CHECK */


/* configure host compiler */
#if defined(__APPLE__)

#define _CRTIMP
#define _ACRTIMP
#define __THROW

#if defined(__BLOCKS__) /* nvcc does not support closures */

#undef __BLOCKS__

#endif /* __BLOCKS__ */

#elif defined(__ANDROID__)

#define _CRTIMP
#define _ACRTIMP
#define __THROW

#elif defined(__QNX__)

#define _CRTIMP
#define _ACRTIMP
#define __THROW

#elif defined(__HORIZON__)

#define _CRTIMP
#define _ACRTIMP
#define __THROW

#elif defined(__GNUC__)

#define _CRTIMP
#define _ACRTIMP

#include <features.h> /* for __THROW */

#elif defined(_WIN32)

#if _MSC_VER >= 1500

#undef _USE_DECLSPECS_FOR_SAL
#define _USE_DECLSPECS_FOR_SAL \
        1

#endif /* _MSC_VER >= 1500 */

#if !defined(_CRT_NONSTDC_NO_WARNINGS)

#define _CRT_NONSTDC_NO_WARNINGS /* to suppress warnings */

#endif /* !_CRT_NONSTDC_NO_WARNINGS */

#if !defined(_CRT_SECURE_NO_WARNINGS)

#define _CRT_SECURE_NO_WARNINGS /* to suppress warnings */

#endif /* !_CRT_SECURE_NO_WARNINGS */

#if !defined(NOMINMAX)

#define NOMINMAX /* min and max are part of cuda runtime */

#endif /* !NOMINMAX */

#include <crtdefs.h> /* for _CRTIMP */
#if _MSC_VER >= 1900
#include <corecrt.h> /* for _ACRTIMP */
#endif /* _MSC_VER >= 1900 */

#define __THROW

#endif /* __APPLE__ */

#endif /* __CUDACC_RTC__ */


#if defined(__cplusplus) && defined(__CUDA_ARCH__) && (defined(__PGIC__) || defined(__CUDACC_RTC__) || (defined(_WIN32) && defined(_MSC_VER)))

#if __CUDACC_RTC__
typedef char *va_list;
#else /* !__CUDACC_RTC__ */
#include <cstdarg>
#endif /* __CUDACC_RTC__ */


#undef va_start
#undef va_end
#undef va_arg

#ifdef __PGIC__

#undef __builtin_va_end

#define va_start(v,l) __builtin_alt_va_start(v,l)
#define va_end(v) __builtin_va_end(v)
#define va_arg(v,l) __builtin_alt_va_arg(v,l)

#if (__cplusplus >= 201103L)
#undef va_copy
#define va_copy(d,s)  __builtin_va_copy(d,s)
#endif

#else /* !__PGIC__ */


#define va_start(ap, x) (__cu_va_start(&ap, x))
#define va_end(ap) (__cu_va_end(&ap))
#define va_arg(ap, t)  (*((t *)__cu_va_arg(&ap, (t *)0)))

#if (_MSC_VER >= 1800) || (defined(__CUDACC_RTC__) && (__cplusplus >= 201103L))
#undef va_copy
#define va_copy(apd, aps) (__cu_va_copy(&(apd), &(aps)))
#endif /* (_MSC_VER >= 1800)  || (defined(__CUDACC_RTC__) && (__cplusplus >= 201103L)) */
#endif /* __PGIC__ */

#endif /* defined(__cplusplus) && (defined(__CUDACC_RTC__) || (defined(_WIN32) && defined(_MSC_VER))) */



#endif /* __CUDACC__ */

#endif /* !__HOST_CONFIG_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_CONFIG_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_HOST_CONFIG_H__
#endif
