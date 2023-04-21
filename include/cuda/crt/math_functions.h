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
#pragma message("crt/math_functions.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/math_functions.h is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_MATH_FUNCTIONS_H__
#endif

#if !defined(__MATH_FUNCTIONS_H__)
#define __MATH_FUNCTIONS_H__

#if defined(__QNX__) && (__GNUC__ >= 5) && defined(__CUDACC__)
#if __has_include(<__config>)
#include <__config>
#endif
#endif

/**
 * \defgroup CUDA_MATH Mathematical Functions
 *
 * CUDA mathematical functions are always available in device code.
 *
 * Host implementations of the common mathematical functions are mapped
 * in a platform-specific way to standard math library functions, provided
 * by the host compiler and respective host libm where available.
 * Some functions, not available with the host compilers, are implemented
 * in crt/math_functions.hpp header file.
 * For example, see ::erfinv(). Other, less common functions,
 * like ::rhypot(), ::cyl_bessel_i0() are only available in device code.
 *
 * Note that many floating-point and integer functions names are
 * overloaded for different argument types. For example, the ::log()
 * function has the following prototypes:
 * \code
 * double log(double x);
 * float log(float x);
 * float logf(float x);
 * \endcode
 */

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

#include "builtin_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern "C"
{

/* Define math function DOXYGEN toplevel groups, functions will
   be added to these groups later.
*/
/**
 * \defgroup CUDA_MATH_SINGLE Single Precision Mathematical Functions
 * This section describes single precision mathematical functions.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */

/**
 * \defgroup CUDA_MATH_DOUBLE Double Precision Mathematical Functions
 * This section describes double precision mathematical functions.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */

/**
 * \defgroup CUDA_MATH_INT Integer Mathematical Functions
 * This section describes integer mathematical functions.
 * To use these functions you do not need to include any additional
 * header files in your program.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_SINGLE Single Precision Intrinsics
 * This section describes single precision intrinsic functions that are
 * only supported in device code.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_DOUBLE Double Precision Intrinsics
 * This section describes double precision intrinsic functions that are
 * only supported in device code.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_INT Integer Intrinsics
 * This section describes integer intrinsic functions that are
 * only supported in device code.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */

/**
 * \defgroup CUDA_MATH_INTRINSIC_CAST Type Casting Intrinsics
 * This section describes type casting intrinsic functions that are
 * only supported in device code.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */

/**
 *
 * \defgroup CUDA_MATH_INTRINSIC_SIMD SIMD Intrinsics
 * This section describes SIMD intrinsic functions that are
 * only supported in device code.
 * To use these functions you do not need to include any additional 
 * header files in your program.
 */


/**
 * @}
 */
#define __DEVICE_FUNCTIONS_DECL__ __host__ __device__
#if !defined(_MSC_VER)
#define __CUDA_MATH_CRTIMP
#else
#if _MSC_VER < 1900
#define __CUDA_MATH_CRTIMP _CRTIMP
#else
#define __CUDA_MATH_CRTIMP _ACRTIMP
#endif
#endif

#if defined(__ANDROID__) && (__ANDROID_API__ <= 20) && !defined(__aarch64__)
static __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ int                    abs(int);
static __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ long int               labs(long int);
static __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ long long int          llabs(long long int);
#else /* __ANDROID__ */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the absolute value of the input \p int argument.
 *
 * Calculate the absolute value of the input argument \p a.
 *
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ int            __cdecl abs(int a) __THROW;
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the absolute value of the input \p long \p int argument.
 *
 * Calculate the absolute value of the input argument \p a.
 *
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ long int       __cdecl labs(long int a) __THROW;
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the absolute value of the input \p long \p long \p int argument.
 *
 * Calculate the absolute value of the input argument \p a.
 *
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __cudart_builtin__ long long int          llabs(long long int a) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
}
#endif
#endif /* __ANDROID__ */

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
/* put all math functions in std */
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the absolute value of the input argument.
 *
 * Calculate the absolute value of the input argument \p x.
 *
 * \return
 * Returns the absolute value of the input argument.
 * - fabs(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fabs(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 0.
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl fabs(double x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the absolute value of its argument
 *
 * Calculate the absolute value of the input argument \p x.
 *
 * \return
 * Returns the absolute value of its argument.
 * - fabs(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fabs(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 0.
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fabsf(float x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    min(const int a, const int b);
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           umin(const unsigned int a, const unsigned int b);
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p long \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          llmin(const long long int a, const long long int b);
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p long \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int ullmin(const unsigned long long int a, const unsigned long long int b);

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine the minimum numeric value of the arguments.
 *
 * Determines the minimum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the minimum numeric value of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fminf(float x, float y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl fminf(float x, float y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the minimum numeric value of the arguments.
 *
 * Determines the minimum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the minimum numeric value of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 fmin(double x, double y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl fmin(double x, double y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    max(const int a, const int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned int           umax(const unsigned int a, const unsigned int b);
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p long \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          llmax(const long long int a, const long long int b);
/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p long \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ unsigned long long int ullmax(const unsigned long long int a, const unsigned long long int b);

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine the maximum numeric value of the arguments.
 *
 * Determines the maximum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the maximum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fmaxf(float x, float y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl fmaxf(float x, float y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Determine the maximum numeric value of the arguments.
 *
 * Determines the maximum numeric value of the arguments \p x and \p y. Treats NaN 
 * arguments as missing data. If one argument is a NaN and the other is legitimate numeric
 * value, the numeric value is chosen.
 *
 * \return
 * Returns the maximum numeric values of the arguments \p x and \p y.
 * - If both arguments are NaN, returns NaN.
 * - If one argument is NaN, returns the numeric argument.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 fmax(double, double) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl fmax(double, double);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine of the input argument.
 *
 * Calculate the sine of the input argument \p x (measured in radians).
 *
 * \return 
 * - sin(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sin(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl sin(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cosine of the input argument.
 *
 * Calculate the cosine of the input argument \p x (measured in radians).
 *
 * \return 
 * - cos(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - cos(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl cos(double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine and cosine of the first input argument.
 *
 * Calculate the sine and cosine of the first input argument \p x (measured 
 * in radians). The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sin() and ::cos().
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   sincos(double x, double *sptr, double *cptr) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine and cosine of the first input argument.
 *
 * Calculate the sine and cosine of the first input argument \p x (measured
 * in radians). The results for sine and cosine are written into the second 
 * argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sinf() and ::cosf().
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   sincosf(float x, float *sptr, float *cptr) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the tangent of the input argument.
 *
 * Calculate the tangent of the input argument \p x (measured in radians).
 *
 * \return 
 * - tan(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tan(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl tan(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the input argument.
 *
 * Calculate the nonnegative square root of \p x, 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrt(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrt(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl sqrt(double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal of the square root of the input argument.
 *
 * Calculate the reciprocal of the nonnegative square root of \p x, 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrt(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - rsqrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrt(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the reciprocal of the square root of the input argument.
 *
 * Calculate the reciprocal of the nonnegative square root of \p x, 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $1/\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>1</m:mn>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrtf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - rsqrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rsqrtf(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x);

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 logarithm of the input argument.
 *
 * Calculate the base 2 logarithm of the input argument \p x.
 *
 * \return 
 * - log2(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log2(1) returns +0.
 * - log2(\p x) returns NaN for \p x < 0.
 * - log2(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 log2(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl log2(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 2 exponential of the input argument.
 *
 * Calculate the base 2 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $2^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 exp2(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl exp2(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 2 exponential of the input argument.
 *
 * Calculate the base 2 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $2^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  exp2f(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl exp2f(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 10 exponential of the input argument.
 *
 * Calculate the base 10 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */         
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 exp10(double x) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 10 exponential of the input argument.
 *
 * Calculate the base 10 exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $10^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>10</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  exp10f(float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument, minus 1.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, minus 1.
 *
 * \return Returns 
 * \latexonly $e^x - 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 expm1(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl expm1(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument, minus 1.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, minus 1.
 *
 * \return  Returns 
 * \latexonly $e^x - 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  expm1f(float x) __THROW;        
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl expm1f(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 2 logarithm of the input argument.
 *
 * Calculate the base 2 logarithm of the input argument \p x.
 *
 * \return
 * - log2f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log2f(1) returns +0.
 * - log2f(\p x) returns NaN for \p x < 0.
 * - log2f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  log2f(float x) __THROW;         
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl log2f(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 10 logarithm of the input argument.
 *
 * Calculate the base 10 logarithm of the input argument \p x.
 *
 * \return 
 * - log10(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log10(1) returns +0.
 * - log10(\p x) returns NaN for \p x < 0.
 * - log10(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl log10(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  logarithm of the input argument \p x.
 *
 * \return 
 * - log(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log(1) returns +0.
 * - log(\p x) returns NaN for \p x < 0.
 * - log(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly

 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl log(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   of the input argument \p x.
 *
 * \return 
 * - log1p(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly.
 * - log1p(-1) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log1p(\p x) returns NaN for \p x < -1.
 * - log1p(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 log1p(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl log1p(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $log_{e}(1+x)$ \endlatexonly
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>+</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   of the input argument \p x.
 *
 * \return 
 * - log1pf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly.
 * - log1pf(-1) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log1pf(\p x) returns NaN for \p x < -1.
 * - log1pf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  log1pf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl log1pf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the largest integer less than or equal to \p x.
 * 
 * Calculates the largest integer value which is less than or equal to \p x.
 * 
 * \return
 * Returns 
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x230A;<!-- &Lfloor --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x230B;<!-- &Rfloor --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - floor(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - floor(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl floor(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x.
 *
 * \return Returns 
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl exp(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic cosine of the input argument.
 *
 * Calculate the hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * - cosh(0) returns 1.
 * - cosh(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl cosh(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic sine of the input argument.
 *
 * Calculate the hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - sinh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinh(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl sinh(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the hyperbolic tangent of the input argument.
 *
 * Calculate the hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - tanh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl tanh(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
 *
 * Calculate the nonnegative arc hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * Result will be in the interval [0, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - acosh(1) returns 0.
 * - acosh(\p x) returns NaN for \p x in the interval [
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 1).
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 acosh(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl acosh(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the nonnegative arc hyperbolic cosine of the input argument.
 *
 * Calculate the nonnegative arc hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * Result will be in the interval [0, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - acoshf(1) returns 0.
 * - acoshf(\p x) returns NaN for \p x in the interval [
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 1).
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  acoshf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl acoshf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc hyperbolic sine of the input argument.
 *
 * Calculate the arc hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - asinh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">0</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">0</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * - asinh(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 asinh(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl asinh(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc hyperbolic sine of the input argument.
 *
 * Calculate the arc hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - asinhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">0</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">0</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * - asinhf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  asinhf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl asinhf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc hyperbolic tangent of the input argument.
 *
 * Calculate the arc hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - atanh(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanh(
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanh(\p x) returns NaN for \p x outside interval [-1, 1].
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 atanh(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl atanh(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc hyperbolic tangent of the input argument.
 *
 * Calculate the arc hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - atanhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanhf(
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - atanhf(\p x) returns NaN for \p x outside interval [-1, 1].
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  atanhf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl atanhf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  of the input arguments \p x and \p exp.
 *
 * \return 
 * - ldexp(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating-point range.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl ldexp(double x, int exp) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the value of 
 * \latexonly $x\cdot 2^{exp}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *       <m:mi>x</m:mi>
 *       <m:mi>p</m:mi>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  of the input arguments \p x and \p exp.
 *
 * \return 
 * - ldexpf(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating-point range.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  ldexpf(float x, int exp) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the floating-point representation of the exponent of the input argument.
 *
 * Calculate the floating-point representation of the exponent of the input argument \p x.
 *
 * \return 
 * - logb
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - logb
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 logb(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl logb(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the floating-point representation of the exponent of the input argument.
 *
 * Calculate the floating-point representation of the exponent of the input argument \p x.
 *
 * \return 
 * - logbf
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - logbf
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  logbf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl logbf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute the unbiased integer exponent of the argument.
 *
 * Calculates the unbiased integer exponent of the input argument \p x.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - ilogb(0) returns <tt>INT_MIN</tt>.
 * - ilogb(NaN) returns <tt>INT_MIN</tt>.
 * - ilogb(\p x) returns <tt>INT_MAX</tt> if \p x is 
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 
 *     or the correct value is greater than <tt>INT_MAX</tt>.
 * - ilogb(\p x) returns <tt>INT_MIN</tt> if the correct value is less 
 *     than <tt>INT_MIN</tt>.
 * - Note: above behavior does not take into account <tt>FP_ILOGB0</tt> nor <tt>FP_ILOGBNAN</tt>.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    ilogb(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP int    __cdecl ilogb(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute the unbiased integer exponent of the argument.
 *
 * Calculates the unbiased integer exponent of the input argument \p x.
 *
 * \return
 * - If successful, returns the unbiased exponent of the argument.
 * - ilogbf(0) returns <tt>INT_MIN</tt>.
 * - ilogbf(NaN) returns <tt>INT_MIN</tt>.
 * - ilogbf(\p x) returns <tt>INT_MAX</tt> if \p x is 
 * \latexonly $\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 
 *     or the correct value is greater than <tt>INT_MAX</tt>.
 * - ilogbf(\p x) returns <tt>INT_MIN</tt> if the correct value is less 
 *     than <tt>INT_MIN</tt>.
 * - Note: above behavior does not take into account <tt>FP_ILOGB0</tt> nor <tt>FP_ILOGBNAN</tt>.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    ilogbf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP int    __cdecl ilogbf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbn(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbn(\p x, 0) returns \p x.
 * - scalbn(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 scalbn(double x, int n) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl scalbn(double x, int n);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbnf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbnf(\p x, 0) returns \p x.
 * - scalbnf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  scalbnf(float x, int n) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl scalbnf(float x, int n);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbln(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalbln(\p x, 0) returns \p x.
 * - scalbln(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 scalbln(double x, long int n) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl scalbln(double x, long int n);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Scale floating-point input by integer power of two.
 *
 * Scale \p x by 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  by efficient manipulation of the floating-point
 * exponent.
 *
 * \return 
 * Returns \p x * 
 * \latexonly $2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalblnf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - scalblnf(\p x, 0) returns \p x.
 * - scalblnf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p n) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  scalblnf(float x, long int n) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl scalblnf(float x, long int n);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Extract mantissa and exponent of a floating-point value
 * 
 * Decompose the floating-point value \p x into a component \p m for the 
 * normalized fraction element and another term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to  0.5 and 
 * less than 1.0 or it will be equal to 0; 
 * \latexonly $x = m\cdot 2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>m</m:mi>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * Returns the fractional component \p m.
 * - frexp(0, \p nptr) returns 0 for the fractional component and zero for the integer component.
 * - frexp(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores zero in the location pointed to by \p nptr.
 * - frexp(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores an unspecified value in the 
 * location to which \p nptr points.
 * - frexp(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl frexp(double x, int *nptr) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Extract mantissa and exponent of a floating-point value
 * 
 * Decomposes the floating-point value \p x into a component \p m for the 
 * normalized fraction element and another term \p n for the exponent.
 * The absolute value of \p m will be greater than or equal to  0.5 and 
 * less than 1.0 or it will be equal to 0; 
 * \latexonly $x = m\cdot 2^n$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>m</m:mi>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:msup>
 *     <m:mn>2</m:mn>
 *     <m:mi>n</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The integer exponent \p n will be stored in the location to which \p nptr points.
 *
 * \return
 * Returns the fractional component \p m.
 * - frexp(0, \p nptr) returns 0 for the fractional component and zero for the integer component.
 * - frexp(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores zero in the location pointed to by \p nptr.
 * - frexp(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p nptr) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores an unspecified value in the 
 * location to which \p nptr points.
 * - frexp(NaN, \p y) returns a NaN and stores an unspecified value in the location to which \p nptr points.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  frexpf(float x, int *nptr) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::rint().
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 round(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl round(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded away from zero.
 *
 * \return
 * Returns rounded integer value.
 *
 * \note_slow_round See ::rintf().
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  roundf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl roundf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::lrint().
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long int               lround(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long int __cdecl lround(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::lrintf().
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long int               lroundf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long int __cdecl lroundf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::llrint().
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          llround(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long long int __cdecl llround(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round to nearest integer value.
 *
 * Round \p x to the nearest integer value, with halfway cases rounded 
 * away from zero.  If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 *
 * \note_slow_round See ::llrintf().
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          llroundf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long long int __cdecl llroundf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded to the nearest even integer value.
 *
 * \return 
 * Returns rounded integer value.
 */
#if defined(__CUDA_ARCH__) || defined(__DOXYGEN_ONLY__)
/*
 * We don't generate the declaration of rint for host compilation.
 * This is acaully a workaround to compile the boost header file when
 * Clang 3.8 is used as the host compiler. The boost header file has
 * the following example code:
 *   namespace NS { extern "C" { double rint(double); }
 *   }
 *
 * After preprocessing, we get something like below:
 *
 * extern "C" { double rint(double x) throw(); }
 * # 30 "/usr/include/math.h" 3
 * extern "C" { double rint(double x) throw(); }
 * namespace NS { extern "C" { double rint(double); } }
 *
 * Although GCC accepts this output, Clang 3.8 doesn't.
 * Furthermore, we cannot change the boost header file by adding "throw()"
 * to rint's declaration there. So, as a workaround, we just don't generate
 * our re-declaration for the host compilation.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rint(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl rint(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#endif /* __CUDA_ARCH__ || __DOXYGEN_ONLY__ */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value in floating-point.
 *
 * Round \p x to the nearest integer value in floating-point format,
 * with halfway cases rounded to the nearest even integer value.
 *
 * \return 
 * Returns rounded integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rintf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl rintf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, 
 * with halfway cases rounded to the nearest even integer value.
 * If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long int               lrint(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long int __cdecl lrint(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, 
 * with halfway cases rounded to the nearest even integer value.
 * If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long int               lrintf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long int __cdecl lrintf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, 
 * with halfway cases rounded to the nearest even integer value.
 * If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          llrint(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long long int __cdecl llrint(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round input to nearest integer value.
 *
 * Round \p x to the nearest integer value, 
 * with halfway cases rounded to the nearest even integer value.
 * If the result is outside the range of the return type,
 * the result is undefined.
 *
 * \return 
 * Returns rounded integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ long long int          llrintf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP long long int __cdecl llrintf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Round the input argument to the nearest integer.
 *
 * Round argument \p x to an integer value in double precision floating-point format. Uses round to nearest rounding, with ties rounding to even.
 *
 * \return 
 * - nearbyint(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - nearbyint(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 nearbyint(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl nearbyint(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Round the input argument to the nearest integer.
 *
 * Round argument \p x to an integer value in single precision floating-point format. Uses round to nearest rounding, with ties rounding to even.
 *
 * \return 
 * - nearbyintf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - nearbyintf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  nearbyintf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl nearbyintf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p x.
 *
 * \return
 * Returns 
 * \latexonly $\lceil x \rceil$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x2308;<!-- &Lceil --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x2309;<!-- &Rceil --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 expressed as a floating-point number.
 * - ceil(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - ceil(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl ceil(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Truncate input argument to the integral part.
 *
 * Round \p x to the nearest integer value that does not exceed \p x in 
 * magnitude.
 *
 * \return 
 * Returns truncated integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 trunc(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl trunc(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Truncate input argument to the integral part.
 *
 * Round \p x to the nearest integer value that does not exceed \p x in 
 * magnitude.
 *
 * \return 
 * Returns truncated integer value.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  truncf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl truncf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute the positive difference between \p x and \p y.
 *
 * Compute the positive difference between \p x and \p y.  The positive
 * difference is \p x - \p y when \p x > \p y and +0 otherwise.
 *
 * \return 
 * Returns the positive difference between \p x and \p y.
 * - fdim(\p x, \p y) returns \p x - \p y if \p x > \p y.
 * - fdim(\p x, \p y) returns +0 if \p x 
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly \p y.
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 fdim(double x, double y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl fdim(double x, double y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute the positive difference between \p x and \p y.
 *
 * Compute the positive difference between \p x and \p y.  The positive
 * difference is \p x - \p y when \p x > \p y and +0 otherwise.
 *
 * \return 
 * Returns the positive difference between \p x and \p y.
 * - fdimf(\p x, \p y) returns \p x - \p y if \p x > \p y.
 * - fdimf(\p x, \p y) returns +0 if \p x 
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly \p y.
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fdimf(float x, float y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl fdimf(float x, float y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc tangent of the ratio of first and second input arguments.
 *
 * Calculate the principal value of the arc tangent of the ratio of first
 * and second input arguments \p y / \p x. The quadrant of the result is
 * determined by the signs of inputs \p y and \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - atan2(0, 1) returns +0.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl atan2(double y, double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc tangent of the input argument.
 *
 * Calculate the principal value of the arc tangent of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2].
 * - atan(0) returns +0.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl atan(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc cosine of the input argument.
 *
 * Calculate the principal value of the arc cosine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [0, 
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - acos(1) returns +0.
 * - acos(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl acos(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the arc sine of the input argument.
 *
 * Calculate the principal value of the arc sine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2, +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * /2] for \p x inside [-1, +1].
 * - asin(0) returns +0.
 * - asin(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl asin(double x) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of two arguments.
 *
 * Calculate the length of the hypotenuse of a right triangle whose two sides have lengths 
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse 
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
#if defined(_WIN32)
#if defined(_MSC_VER) && _MSC_VER < 1900
static __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double __CRTDECL hypot(double x, double y);
#else
extern _ACRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double __cdecl hypot(double x, double y);
#endif
#else /* _WIN32 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double           hypot(double x, double y) __THROW;
#endif /* _WIN32 */

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate one over the square root of the sum of squares of two arguments.
 *
 * Calculate one over the length of the hypotenuse of a right triangle whose two sides have 
 * lengths \p x and \p y without undue overflow or underflow.
 *
 * \return Returns one over the length of the hypotenuse 
 * \latexonly $\frac{1}{\sqrt{x^2+y^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double                rhypot(double x, double y) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of two arguments.
 *
 * Calculates the length of the hypotenuse of a right triangle whose two sides have lengths 
 * \p x and \p y without undue overflow or underflow.
 *
 * \return Returns the length of the hypotenuse 
 * \latexonly $\sqrt{x^2+y^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
#if defined(_WIN32)
static __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __CRTDECL hypotf(float x, float y);
#else /* _WIN32 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float           hypotf(float x, float y) __THROW;
#endif /* _WIN32 */

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate one over the square root of the sum of squares of two arguments.
 *
 * Calculates one over the length of the hypotenuse of a right triangle whose two sides have 
 * lengths \p x and \p y without undue overflow or underflow.
 *
 * \return Returns one over the length of the hypotenuse 
 * \latexonly $\frac{1}{\sqrt{x^2+y^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __device__ __device_builtin__ float                 rhypotf(float x, float y) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculate the length of three dimensional vector \p p in Euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of 3D vector
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl norm3d(double a, double b, double c) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate one over the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculate one over the length of three dimensional vector \p p in Euclidean space undue overflow or underflow.
 *
 * \return Returns one over the length of the 3D vector 
 * \latexonly $\frac{1}{\sqrt{p.x^2+p.y^2+p.z^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double                rnorm3d(double a, double b, double c) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of four coordinates of the argument.
 *
 * Calculate the length of four dimensional vector \p p in Euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of 4D vector
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2+p.t^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.t</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl norm4d(double a, double b, double c, double d) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate one over the square root of the sum of squares of four coordinates of the argument.
 *
 * Calculate one over the length of four dimensional vector \p p in Euclidean space undue overflow or underflow.
 *
 * \return Returns one over the length of the 3D vector 
 * \latexonly $\frac{1}{\sqrt{p.x^2+p.y^2+p.z^2+p.t^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *	   <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.t</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double rnorm4d(double a, double b, double c, double d) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the square root of the sum of squares of any number of coordinates.
 *
 * Calculate the length of a vector p, dimension of which is passed as an argument \p without undue overflow or underflow.
 *
 * \return Returns the length of the dim-D vector 
 * \latexonly $\sqrt{\sum_{i=1}^{dim} p_i^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.1</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.2</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+ ... +</m:mo>
 *     <m:msup>
 *       <m:mi>p.dim</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 * If two of the input arguments is 0, returns remaining argument
 *
 * \note_accuracy_double
 */
extern "C++"  __device__ __device_builtin__  double norm(int dim, double const * t) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the reciprocal of square root of the sum of squares of any number of coordinates.
 *
 * Calculates one over the length of vector \p p, dimension of which is passed as an argument, in Euclidean space without undue overflow or underflow.
 *
 * \return Returns one over the length of the vector
 * \latexonly $\frac{1}{\sqrt{\sum_{i=1}^{dim} p_i^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.1</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.2</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+ ... +</m:mo>
 *         <m:msup>
 *           <m:mi>p.dim</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double rnorm(int dim, double const * t) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the reciprocal of square root of the sum of squares of any number of coordinates.
 *
 * Calculates one over the length of vector \p p, dimension of which is passed as an argument, in Euclidean space without undue overflow or underflow.
 *
 * \return Returns one over the length of the vector
 * \latexonly $\frac{1}{\sqrt{\sum_{i=1}^{dim} p_i^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.1</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.2</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+ ... +</m:mo>
 *         <m:msup>
 *           <m:mi>p.dim</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */

extern __device__ __device_builtin__ float rnormf(int dim, float const * a) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of any number of coordinates.
 *
 * Calculates the length of a vector \p p, dimension of which is passed as an argument without undue overflow or underflow.
 *
 * \return Returns the length of the vector
 * \latexonly $\sqrt{\sum_{i=1}^{dim} p_i^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.1</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.2</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+ ... +</m:mo>
 *     <m:msup>
 *       <m:mi>p.dim</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
extern "C++"  __device__ __device_builtin__  float normf(int dim, float const * a) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculates the length of three dimensional vector \p p in Euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of the 3D
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */

extern __device__ __device_builtin__ float norm3df(float a, float b, float c) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate one over the square root of the sum of squares of three coordinates of the argument.
 *
 * Calculates one over the length of three dimension vector \p p in Euclidean space without undue overflow or underflow.
 *
 * \return Returns one over the length of the 3D vector
 * \latexonly $\frac{1}{\sqrt{p.x^2+p.y^2+p.z^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __device__ __device_builtin__ float rnorm3df(float a, float b, float c) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the sum of squares of four coordinates of the argument.
 *
 * Calculates the length of four dimensional vector \p p in Euclidean space without undue overflow or underflow.
 *
 * \return Returns the length of the 4D vector
 * \latexonly $\sqrt{p.x^2+p.y^2+p.z^2+p.t^2}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:msup>
 *       <m:mi>p.x</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.y</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.z</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *     <m:mo>+</m:mo>
 *     <m:msup>
 *       <m:mi>p.t</m:mi>
 *       <m:mn>2</m:mn>
 *     </m:msup>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the correct value would overflow, returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * If the correct value would underflow, returns 0.
 *
 * \note_accuracy_single
 */
extern __device__ __device_builtin__ float norm4df(float a, float b, float c, float d) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate one over the square root of the sum of squares of four coordinates of the argument.
 *
 * Calculates one over the length of four dimension vector \p p in Euclidean space without undue overflow or underflow.
 *
 * \return Returns one over the length of the 3D vector
 * \latexonly $\frac{1}{\sqrt{p.x^2+p.y^2+p.z^2+p.t^2}}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mrow>
 *       <m:mi>1</m:mi>
 *     </m:mrow>
 *     <m:mrow>
 *       <m:msqrt>
 *         <m:msup>
 *           <m:mi>p.x</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.y</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *         <m:mo>+</m:mo>
 *         <m:msup>
 *           <m:mi>p.z</m:mi>
 *           <m:mn>2</m:mn>
 *         </m:msup>
 *       </m:msqrt>
 *     </m:mrow>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * If the square root would overflow, returns 0.
 * If the square root would underflow, returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __device__ __device_builtin__ float rnorm4df(float a, float b, float c, float d) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cube root of the input argument.
 *
 * Calculate the cube root of \p x, 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cbrt(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl cbrt(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cube root of the input argument.
 *
 * Calculate the cube root of \p x, 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $x^{1/3}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>x</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>1</m:mn>
 *       <m:mrow class="MJX-TeXAtom-ORD">
 *         <m:mo>/</m:mo>
 *       </m:mrow>
 *       <m:mn>3</m:mn>
 *     </m:mrow>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - cbrtf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cbrtf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl cbrtf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate reciprocal cube root function.
 *
 * Calculate reciprocal cube root function of \p x
 *
 * \return 
 * - rcbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rcbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rcbrt(double x);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate reciprocal cube root function.
 *
 * Calculate reciprocal cube root function of \p x
 *
 * \return 
 * - rcbrt(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - rcbrt(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rcbrtf(float x);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the sine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians), 
 * where \p x is the input argument.
 *
 * \return 
 * - sinpi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinpi(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians), 
 * where \p x is the input argument.
 *
 * \return 
 * - sinpif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinpif(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the cosine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the cosine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians), 
 * where \p x is the input argument.
 *
 * \return 
 * - cospi(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - cospi(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cosine of the input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the cosine of \p x
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  (measured in radians),
 * where \p x is the input argument.
 *
 * \return 
 * - cospif(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - cospif(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief  Calculate the sine and cosine of the first input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine and cosine of the first input argument, \p x (measured in radians), 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.  The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sinpi() and ::cospi().
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   sincospi(double x, double *sptr, double *cptr);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief  Calculate the sine and cosine of the first input argument 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * Calculate the sine and cosine of the first input argument, \p x (measured in radians), 
 * \latexonly $\times \pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.  The results for sine and cosine are written into the
 * second argument, \p sptr, and, respectively, third argument, \p cptr.
 *
 * \return 
 * - none
 *
 * \see ::sinpif() and ::cospif().
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ void                   sincospif(float x, float *sptr, float *cptr);

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y
 *
 * \return 
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer less than 0.
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
*   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y less than 0 and not an odd integer.
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - pow(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not an odd integer.
 * - pow(-1, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - pow(+1, \p y) returns 1 for any \p y, even a NaN.
 * - pow(\p x, 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - pow(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - pow(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - pow(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - pow(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - pow(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl pow(double x, double y) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The 
 * integral part is stored in the argument \p iptr.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return 
 * - modf(
 * \latexonly $\pm x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *  <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi>x</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns a result with the same sign as \p x.
 * - modf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   in the object pointed to by \p iptr.
 * - modf(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl modf(double x, double *iptr) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the double-precision floating-point remainder of \p x / \p y.
 *
 * Calculate the double-precision floating-point remainder of \p x / \p y.
 * The floating-point remainder of the division operation \p x / \p y calculated
 * by this function is exactly the value <tt>x - n*y</tt>, where \p n is \p x / \p y with its fractional part truncated.
 * The computed value will have the same sign as \p x, and its magnitude will be less than the magnitude of \p y.
 *
 * \return
 * - Returns the floating-point remainder of \p x / \p y.
 * - fmod(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p y is not zero.
 * - fmod(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x if \p x is finite.
 * - fmod(\p x, \p y) returns NaN if \p x is 
 * \latexonly $\pm\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  or \p y is zero.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double         __cdecl fmod(double x, double y) __THROW;
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute double-precision floating-point remainder.
 *
 * Compute double-precision floating-point remainder \p r of dividing 
 * \p x by \p y for nonzero \p y. Thus 
 * \latexonly $ r = x - n y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>r</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi>n</m:mi>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The value \p n is the integer value nearest 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * In the case when 
 * \latexonly $ | n -\frac{x}{y} | = \frac{1}{2} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>n</m:mi>
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>=</m:mo>
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mn>2</m:mn>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the
 * even \p n value is chosen.
 *
 * \return 
 * - remainder(\p x, 0) returns NaN.
 * - remainder(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN.
 * - remainder(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x for finite \p x.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 remainder(double x, double y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl remainder(double x, double y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute single-precision floating-point remainder.
 *
 * Compute single-precision floating-point remainder \p r of dividing 
 * \p x by \p y for nonzero \p y. Thus 
 * \latexonly $ r = x - n y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>r</m:mi>
 *   <m:mo>=</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi>n</m:mi>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * The value \p n is the integer value nearest 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly. 
 * In the case when 
 * \latexonly $ | n -\frac{x}{y} | = \frac{1}{2} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>n</m:mi>
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>=</m:mo>
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mn>2</m:mn>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , the
 * even \p n value is chosen.
 *
 * \return 
 * \return 
 * - remainderf(\p x, 0) returns NaN.
 * - remainderf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns NaN.
 * - remainderf(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x for finite \p x.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  remainderf(float x, float y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl remainderf(float x, float y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute double-precision floating-point remainder and part of quotient.
 *
 * Compute a double-precision floating-point remainder in the same way as the
 * ::remainder() function. Argument \p quo returns part of quotient upon 
 * division of \p x by \p y. Value \p quo has the same sign as 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * and may not be the exact quotient but agrees with the exact quotient
 * in the low order 3 bits.
 *
 * \return 
 * Returns the remainder.
 * - remquo(\p x, 0, \p quo) returns NaN.
 * - remquo(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y, \p quo) returns NaN.
 * - remquo(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p quo) returns \p x.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 remquo(double x, double y, int *quo) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl remquo(double x, double y, int *quo);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute single-precision floating-point remainder and part of quotient.
 *
 * Compute a double-precision floating-point remainder in the same way as the 
 * ::remainderf() function. Argument \p quo returns part of quotient upon 
 * division of \p x by \p y. Value \p quo has the same sign as 
 * \latexonly $ \frac{x}{y} $ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mi>x</m:mi>
 *     <m:mi>y</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * and may not be the exact quotient but agrees with the exact quotient
 * in the low order 3 bits.
 *
 * \return 
 * Returns the remainder.
 * - remquof(\p x, 0, \p quo) returns NaN.
 * - remquof(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y, \p quo) returns NaN.
 * - remquof(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p quo) returns \p x.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  remquof(float x, float y, int *quo) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl remquof(float x, float y, int *quo);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 0 for
 * the input argument \p x, 
 * \latexonly $J_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 0.
 * - j0(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - j0(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl j0(double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 0 for
 * the input argument \p x, 
 * \latexonly $J_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 0.
 * - j0f(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - j0f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  j0f(float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 1 for
 * the input argument \p x, 
 * \latexonly $J_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 1.
 * - j1(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - j1(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - j1(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl j1(double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order 1 for
 * the input argument \p x, 
 * \latexonly $J_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order 1.
 * - j1f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - j1f(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - j1f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  j1f(float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order \p n for
 * the input argument \p x, 
 * \latexonly $J_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order \p n.
 * - jn(\p n, NaN) returns NaN.
 * - jn(\p n, \p x) returns NaN for \p n < 0.
 * - jn(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl jn(int n, double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the first kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the first kind of order \p n for
 * the input argument \p x, 
 * \latexonly $J_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>J</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the first kind of order \p n.
 * - jnf(\p n, NaN) returns NaN.
 * - jnf(\p n, \p x) returns NaN for \p n < 0.
 * - jnf(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  jnf(int n, float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 0 for
 * the input argument \p x, 
 * \latexonly $Y_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 0.
 * - y0(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y0(\p x) returns NaN for \p x < 0.
 * - y0(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y0(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl y0(double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 0 for
 * the input argument \p x, 
 * \latexonly $Y_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 0.
 * - y0f(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y0f(\p x) returns NaN for \p x < 0.
 * - y0f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y0f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  y0f(float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 1 for
 * the input argument \p x, 
 * \latexonly $Y_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 1.
 * - y1(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y1(\p x) returns NaN for \p x < 0.
 * - y1(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y1(NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl y1(double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order 1 for
 * the input argument \p x, 
 * \latexonly $Y_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order 1.
 * - y1f(0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - y1f(\p x) returns NaN for \p x < 0.
 * - y1f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - y1f(NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  y1f(float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order \p n for
 * the input argument \p x, 
 * \latexonly $Y_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order \p n.
 * - yn(\p n, \p x) returns NaN for \p n < 0.
 * - yn(\p n, 0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - yn(\p n, \p x) returns NaN for \p x < 0.
 * - yn(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - yn(\p n, NaN) returns NaN.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl yn(int n, double x) __THROW;
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the Bessel function of the second kind of order n for the input argument.
 *
 * Calculate the value of the Bessel function of the second kind of order \p n for
 * the input argument \p x, 
 * \latexonly $Y_n(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>Y</m:mi>
 *     <m:mi>n</m:mi>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the Bessel function of the second kind of order \p n.
 * - ynf(\p n, \p x) returns NaN for \p n < 0.
 * - ynf(\p n, 0) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - ynf(\p n, \p x) returns NaN for \p x < 0.
 * - ynf(\p n, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 * - ynf(\p n, NaN) returns NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  ynf(int n, float x) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 0 for
 * the input argument \p x, 
 * \latexonly $I_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 0.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl cyl_bessel_i0(double x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 0 for
 * the input argument \p x, 
 * \latexonly $I_0(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>0</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 0.
 *
 * \note_accuracy_single
 */
extern __device__ __device_builtin__ float                  cyl_bessel_i0f(float x) __THROW;

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 1 for
 * the input argument \p x, 
 * \latexonly $I_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 1.
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl cyl_bessel_i1(double x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument.
 *
 * Calculate the value of the regular modified cylindrical Bessel function of order 1 for
 * the input argument \p x, 
 * \latexonly $I_1(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>I</m:mi>
 *     <m:mn>1</m:mn>
 *   </m:msub>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return
 * Returns the value of the regular modified cylindrical Bessel function of order 1.
 *
 * \note_accuracy_single
 */
extern __device__ __device_builtin__ float                  cyl_bessel_i1f(float x) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the error function of the input argument.
 *
 * Calculate the value of the error function for the input argument \p x,
 * \latexonly $\frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>2</m:mn>
 *     <m:msqrt>
 *       <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 *     </m:msqrt>
 *   </m:mfrac>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- &Int --></m:mo>
 *     <m:mn>0</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 erf(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl erf(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the error function of the input argument.
 *
 * Calculate the value of the error function for the input argument \p x,
 * \latexonly $\frac{2}{\sqrt \pi} \int_0^x e^{-t^2} dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>2</m:mn>
 *     <m:msqrt>
 *       <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 *     </m:msqrt>
 *   </m:mfrac>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- &Int --></m:mo>
 *     <m:mn>0</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return  
 * - erff(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erff(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  erff(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl erff(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse error function of the input argument.
 *
 * Calculate the inverse error function of the input argument \p y, for \p y in the interval [-1, 1].
 * The inverse error function finds the value \p x that satisfies the equation \p y = erf(\p x),
 * for 
 * \latexonly $-1 \le y \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfinv(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfinv(-1) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 erfinv(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse error function of the input argument.
 *
 * Calculate the inverse error function of the input argument \p y, for \p y in the interval [-1, 1].
 * The inverse error function finds the value \p x that satisfies the equation \p y = erf(\p x),
 * for 
 * \latexonly $-1 \le y \le 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfinvf(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfinvf(-1) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  erfinvf(float y);

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the complementary error function of the input argument.
 *
 * Calculate the complementary error function of the input argument \p x,
 * 1 - erf(\p x).
 *
 * \return 
 * - erfc(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 2.
 * - erfc(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 erfc(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl erfc(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the complementary error function of the input argument.
 *
 * Calculate the complementary error function of the input argument \p x,
 * 1 - erf(\p x).
 *
 * \return 
 * - erfcf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 2.
 * - erfcf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  erfcf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl erfcf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
 *
 * Calculate the natural logarithm of the absolute value of the gamma function of the input argument \p x, namely the value of
 * \latexonly $\log_{e}\left|\int_{0}^{\infty} e^{-t}t^{x-1}dt\right|$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msub>
 *     <m:mi>log</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mfenced open="|" close="|">
 *     <m:mrow>
 *       <m:msubsup>
 *         <m:mo>&#x222B;<!-- &Int --></m:mo>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mn>0</m:mn>
 *         </m:mrow>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *         </m:mrow>
 *       </m:msubsup>
 *       <m:msup>
 *         <m:mi>e</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *           <m:mi>t</m:mi>
 *         </m:mrow>
 *       </m:msup>
 *       <m:msup>
 *         <m:mi>t</m:mi>
 *         <m:mrow class="MJX-TeXAtom-ORD">
 *           <m:mi>x</m:mi>
 *           <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *           <m:mn>1</m:mn>
 *         </m:mrow>
 *       </m:msup>
 *       <m:mi>d</m:mi>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:mfenced>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *
 * \return 
 * - lgamma(1) returns +0.
 * - lgamma(2) returns +0.
 * - lgamma(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating-point range.
 * - lgamma(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p x 
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly 0 and \p x is an integer.
 * - lgamma(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - lgamma(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 lgamma(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl lgamma(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse complementary error function of the input argument.
 *
 * Calculate the inverse complementary error function of the input argument \p y, for \p y in the interval [0, 2].
 * The inverse complementary error function find the value \p x that satisfies the equation \p y = erfc(\p x),
 * for 
 * \latexonly $0 \le y \le 2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcinv(0) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfcinv(2) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 erfcinv(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse complementary error function of the input argument.
 *
 * Calculate the inverse complementary error function of the input argument \p y, for \p y in the interval [0, 2].
 * The inverse complementary error function find the value \p x that satisfies the equation \p y = erfc(\p x),
 * for 
 * \latexonly $0 \le y \le 2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mn>0</m:mn>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , and 
 * \latexonly $-\infty \le x \le \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcinvf(0) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - erfcinvf(2) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  erfcinvf(float y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the inverse of the standard normal cumulative distribution function.
 *
 * Calculate the inverse of the standard normal cumulative distribution function for input argument \p y,
 * \latexonly $\Phi^{-1}(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi mathvariant="normal">&#x03A6;<!-- &Phi --></m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly. The function is defined for input values in the interval 
 * \latexonly $(0, 1)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>0</m:mn>
 *   <m:mo>,</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdfinv(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinv(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinv(\p x) returns NaN
 *  if \p x is not in the interval [0,1].
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 normcdfinv(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the inverse of the standard normal cumulative distribution function.
 *
 * Calculate the inverse of the standard normal cumulative distribution function for input argument \p y,
 * \latexonly $\Phi^{-1}(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi mathvariant="normal">&#x03A6;<!-- &Phi --></m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly. The function is defined for input values in the interval 
 * \latexonly $(0, 1)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mn>0</m:mn>
 *   <m:mo>,</m:mo>
 *   <m:mn>1</m:mn>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdfinvf(0) returns
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinvf(1) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - normcdfinvf(\p x) returns NaN
 *  if \p x is not in the interval [0,1].
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  normcdfinvf(float y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the standard normal cumulative distribution function.
 *
 * Calculate the cumulative distribution function of the standard normal distribution for input argument \p y,
 * \latexonly $\Phi(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x03A6;<!-- &Phi --></m:mi>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1
 * - normcdf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 normcdf(double y);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the standard normal cumulative distribution function.
 *
 * Calculate the cumulative distribution function of the standard normal distribution for input argument \p y,
 * \latexonly $\Phi(y)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x03A6;<!-- &Phi --></m:mi>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - normcdff(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1
 * - normcdff(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0

 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  normcdff(float y);
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the scaled complementary error function of the input argument.
 *
 * Calculate the scaled complementary error function of the input argument \p x,
 * \latexonly $e^{x^2}\cdot \textrm{erfc}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:msup>
 *         <m:mi>x</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mtext>erfc</m:mtext>
 *   </m:mrow>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcx(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - erfcx(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 * - erfcx(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating-point range.
 *
 * \note_accuracy_double
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 erfcx(double x);
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the scaled complementary error function of the input argument.
 *
 * Calculate the scaled complementary error function of the input argument \p x,
 * \latexonly $e^{x^2}\cdot \textrm{erfc}(x)$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:msup>
 *         <m:mi>x</m:mi>
 *         <m:mn>2</m:mn>
 *       </m:msup>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mo>&#x22C5;<!-- &Sdot --></m:mo>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mtext>erfc</m:mtext>
 *   </m:mrow>
 *   <m:mo stretchy="false">(</m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo stretchy="false">)</m:mo>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - erfcxf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * - erfcxf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML"> 
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0
 * - erfcxf(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating-point range.

 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  erfcxf(float x);

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
 *
 * Calculate the natural logarithm of the absolute value of the gamma function of the input argument \p x, namely the value of
 * \latexonly $log_{e}|\ \int_{0}^{\infty} e^{-t}t^{x-1}dt|$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>l</m:mi>
 *   <m:mi>o</m:mi>
 *   <m:msub>
 *     <m:mi>g</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>e</m:mi>
 *     </m:mrow>
 *   </m:msub>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mtext>&#xA0;</m:mtext>
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- &Int --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - lgammaf(1) returns +0.
 * - lgammaf(2) returns +0.
 * - lgammaf(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating-point range.
 * - lgammaf(\p x) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p x
 * \latexonly $\leq$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2264;<!-- &Le --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  0 and \p x is an integer.
 * - lgammaf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - lgammaf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  lgammaf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl lgammaf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the gamma function of the input argument.
 *
 * Calculate the gamma function of the input argument \p x, namely the value of
 * \latexonly $\int_{0}^{\infty} e^{-t}t^{x-1}dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- &Int --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - tgamma(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tgamma(2) returns +1.
 * - tgamma(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the double floating-point range.
 * - tgamma(\p x) returns NaN if \p x < 0 and \p x is an integer.
 * - tgamma(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 * - tgamma(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 tgamma(double x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl tgamma(double x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the gamma function of the input argument.
 *
 * Calculate the gamma function of the input argument \p x, namely the value of
 * \latexonly $\int_{0}^{\infty} e^{-t}t^{x-1}dt$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msubsup>
 *     <m:mo>&#x222B;<!-- &Int --></m:mo>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mn>0</m:mn>
 *     </m:mrow>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 *     </m:mrow>
 *   </m:msubsup>
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mi>t</m:mi>
 *     </m:mrow>
 *   </m:msup>
 *   <m:msup>
 *     <m:mi>t</m:mi>
 *     <m:mrow class="MJX-TeXAtom-ORD">
 *       <m:mi>x</m:mi>
 *       <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *       <m:mn>1</m:mn>
 *     </m:mrow>
 *   </m:msup>
 *   <m:mi>d</m:mi>
 *   <m:mi>t</m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * - tgammaf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tgammaf(2) returns +1.
 * - tgammaf(\p x) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if the correctly calculated value is outside the single floating-point range.
 * - tgammaf(\p x) returns NaN if \p x < 0  and \p x is an integer.
 * - tgammaf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 * - tgammaf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  tgammaf(float x) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl tgammaf(float x);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/** \ingroup CUDA_MATH_DOUBLE
 * \brief Create value with given magnitude, copying sign of second value.
 *
 * Create a floating-point value with the magnitude \p x and the sign of \p y.
 *
 * \return
 * Returns a value with the magnitude of \p x and the sign of \p y.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 copysign(double x, double y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl copysign(double x, double y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/** \ingroup CUDA_MATH_SINGLE
 * \brief Create value with given magnitude, copying sign of second value.
 *
 * Create a floating-point value with the magnitude \p x and the sign of \p y.
 *
 * \return
 * Returns a value with the magnitude of \p x and the sign of \p y.
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  copysignf(float x, float y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl copysignf(float x, float y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Return next representable double-precision floating-point value after argument \p x in the direction of \p y.
 *
 * Calculate the next representable double-precision floating-point value
 * following \p x in the direction of \p y. For example, if \p y is greater than \p x, ::nextafter()
 * returns the smallest representable number greater than \p x
 *
 * \return 
 * - nextafter(\p x, \p y) = \p y if \p x equals \p y
 * - nextafter(\p x, \p y) = \p NaN if either \p x or \p y are \p NaN
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 nextafter(double x, double y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl nextafter(double x, double y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Return next representable single-precision floating-point value after argument \p x in the direction of \p y.
 *
 * Calculate the next representable single-precision floating-point value
 * following \p x in the direction of \p y. For example, if \p y is greater than \p x, ::nextafterf()
 * returns the smallest representable number greater than \p x
 *
 * \return 
 * - nextafterf(\p x, \p y) = \p y if \p x equals \p y
 * - nextafterf(\p x, \p y) = \p NaN if either \p x or \p y are \p NaN
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  nextafterf(float x, float y) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl nextafterf(float x, float y);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Returns "Not a Number" value.
 *
 * Return a representation of a quiet NaN. Argument \p tagp selects one of the possible representations.
 *
 * \return 
 * - nan(\p tagp) returns NaN.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 nan(const char *tagp) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl nan(const char *tagp);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Returns "Not a Number" value
 *
 * Return a representation of a quiet NaN. Argument \p tagp selects one of the possible representations.
 *
 * \return 
 * - nanf(\p tagp) returns NaN.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  nanf(const char *tagp) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl nanf(const char *tagp);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* namespace std */
#endif
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isinff(float) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isnanf(float) __THROW;


#if defined(__APPLE__)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isfinited(double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isfinitef(float) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __signbitd(double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isnand(double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isinfd(double) __THROW;
#else /* __APPLE__ */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __finite(double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __finitef(float) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __signbit(double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isnan(double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isinf(double) __THROW;
#endif /* __APPLE__ */

extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __signbitf(float) __THROW;

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 *
 * Compute the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation. After computing the value
 * to infinite precision, the value is rounded once.
 *
 * \return
 * Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - fma(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fma(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fma(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fma(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 fma(double x, double y, double z) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP double __cdecl fma(double x, double y, double z);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 *
 * Compute the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation. After computing the value
 * to infinite precision, the value is rounded once.
 *
 * \return
 * Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation.
 * - fmaf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - fmaf(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if 
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Times --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
#if (!defined(_MSC_VER) || _MSC_VER < 1800)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fmaf(float x, float y, float z) __THROW;
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ __CUDA_MATH_CRTIMP float  __cdecl fmaf(float x, float y, float z);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif


/* these are here to avoid warnings on the call graph.
   long double is not supported on the device */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __signbitl(long double) __THROW;
#if defined(__APPLE__)
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isfinite(long double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isinf(long double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isnan(long double) __THROW;
#else /* __APPLE__ */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __finitel(long double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isinfl(long double) __THROW;
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int                    __isnanl(long double) __THROW;
#endif /* __APPLE__ */

#if defined(_WIN32) && defined(_M_AMD64)
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl acosf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl asinf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl atanf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl atan2f(float, float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl cosf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl sinf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl tanf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl coshf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl sinhf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl tanhf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl expf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl logf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl log10f(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl modff(float, float*) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl powf(float, float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl sqrtf(float) __THROW;         
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl ceilf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl floorf(float) __THROW;
extern __CUDA_MATH_CRTIMP __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float __cdecl fmodf(float, float) __THROW;
#else /* _WIN32 && _M_AMD64 */

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc cosine of the input argument.
 *
 * Calculate the principal value of the arc cosine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [0, 
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - acosf(1) returns +0.
 * - acosf(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  acosf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc sine of the input argument.
 *
 * Calculate the principal value of the arc sine of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , +
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ] for \p x inside [-1, +1].
 * - asinf(0) returns +0.
 * - asinf(\p x) returns NaN for \p x outside [-1, +1].
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  asinf(float x) __THROW;

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc tangent of the input argument.
 *
 * Calculate the principal value of the arc tangent of the input argument \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , +
 * \latexonly $\pi/2$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo>/</m:mo>
 *   </m:mrow>
 *   <m:mn>2</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - atanf(0) returns +0.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  atanf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the arc tangent of the ratio of first and second input arguments.
 *
 * Calculate the principal value of the arc tangent of the ratio of first
 * and second input arguments \p y / \p x. The quadrant of the result is 
 * determined by the signs of inputs \p y and \p x.
 *
 * \return 
 * Result will be in radians, in the interval [-
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , +
 * \latexonly $\pi$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>&#x03C0;<!-- &Pi --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ].
 * - atan2f(0, 1) returns +0.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  atan2f(float y, float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the cosine of the input argument.
 *
 * Calculate the cosine of the input argument \p x (measured in radians).
 *
 * \return 
 * - cosf(0) returns 1.
 * - cosf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cosf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the sine of the input argument.
 *
 * Calculate the sine of the input argument \p x (measured in radians).
 *
 * \return 
 * - sinf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the tangent of the input argument.
 *
 * Calculate the tangent of the input argument \p x (measured in radians).
 *
 * \return 
 * - tanf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - tanf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  tanf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic cosine of the input argument.
 *
 * Calculate the hyperbolic cosine of the input argument \p x.
 *
 * \return 
 * - coshf(0) returns 1.
 * - coshf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  coshf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic sine of the input argument.
 *
 * Calculate the hyperbolic sine of the input argument \p x.
 *
 * \return 
 * - sinhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sinhf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinhf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the hyperbolic tangent of the input argument.
 *
 * Calculate the hyperbolic tangent of the input argument \p x.
 *
 * \return 
 * - tanhf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  tanhf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the natural logarithm of the input argument.
 *
 * Calculate the natural logarithm of the input argument \p x.
 *
 * \return 
 * - logf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - logf(1) returns +0.
 * - logf(\p x) returns NaN for \p x < 0.
 * - logf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  logf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument.
 *
 * Calculate the base 
 * \latexonly $e$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>e</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  exponential of the input argument \p x, 
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return Returns 
 * \latexonly $e^x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msup>
 *     <m:mi>e</m:mi>
 *     <m:mi>x</m:mi>
 *   </m:msup>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  expf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the base 10 logarithm of the input argument.
 *
 * Calculate the base 10 logarithm of the input argument \p x.
 *
 * \return 
 * - log10f(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - log10f(1) returns +0.
 * - log10f(\p x) returns NaN for \p x < 0.
 * - log10f(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  log10f(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Break down the input argument into fractional and integral parts.
 *
 * Break down the argument \p x into fractional and integral parts. The integral part is stored in the argument \p iptr.
 * Fractional and integral parts are given the same sign as the argument \p x.
 *
 * \return 
 * - modff(
 * \latexonly $\pm x$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *  <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi>x</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns a result with the same sign as \p x.
 * - modff(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p iptr) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  and stores 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *   in the object pointed to by \p iptr.
 * - modff(NaN, \p iptr) stores a NaN in the object pointed to by \p iptr and returns a NaN.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  modff(float x, float *iptr) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the value of first argument to the power of second argument.
 *
 * Calculate the value of \p x to the power of \p y.
 *
 * \return 
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer less than 0.
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y less than 0 and not an odd integer.
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - powf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y > 0 and not an odd integer.
 * - powf(-1, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1.
 * - powf(+1, \p y) returns 1 for any \p y, even a NaN.
 * - powf(\p x, 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 1 for any \p x, even a NaN.
 * - powf(\p x, \p y) returns a NaN for finite \p x < 0 and finite non-integer \p y.
 * - powf(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(\p x, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns +0 for 
 * \latexonly $| x | < 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&lt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(\p x, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for 
 * \latexonly $| x | > 1$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mi>x</m:mi>
 *   <m:mrow class="MJX-TeXAtom-ORD">
 *     <m:mo stretchy="false">|</m:mo>
 *   </m:mrow>
 *   <m:mo>&gt;</m:mo>
 *   <m:mn>1</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns -0 for \p y an odd integer less than 0.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0 and not an odd integer.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y an odd integer greater than 0.
 * - powf(
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x2212;<!-- &Minus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0 and not an odd integer.
 * - powf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns +0 for \p y < 0.
 * - powf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  for \p y > 0.
 *
 * \note_accuracy_single
 * \note_fastmath
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  powf(float x, float y) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the square root of the input argument.
 *
 * Calculate the nonnegative square root of \p x, 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \return 
 * Returns 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrtf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrtf(
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - sqrtf(\p x) returns NaN if \p x is less than 0.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sqrtf(float x) __THROW;         
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate ceiling of the input argument.
 *
 * Compute the smallest integer value not less than \p x.
 *
 * \return
 * Returns 
 * \latexonly $\lceil x \rceil$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x2308;<!-- &Lceil --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x2309;<!-- &Rceil --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - ceilf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - ceilf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  ceilf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the largest integer less than or equal to \p x.
 * 
 * Calculate the largest integer value which is less than or equal to \p x.
 * 
 * \return
 * Returns 
 * \latexonly $\lfloor x \rfloor$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo fence="false" stretchy="false">&#x230A;<!-- &Lfloor --></m:mo>
 *   <m:mi>x</m:mi>
 *   <m:mo fence="false" stretchy="false">&#x230B;<!-- &Rfloor --></m:mo>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  expressed as a floating-point number.
 * - floorf(
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 * - floorf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  floorf(float x) __THROW;
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the floating-point remainder of \p x / \p y.
 *
 * Calculate the floating-point remainder of \p x / \p y.
 * The floating-point remainder of the division operation \p x / \p y calculated
 * by this function is exactly the value <tt>x - n*y</tt>, where \p n is \p x / \p y with its fractional part truncated.
 * The computed value will have the same sign as \p x, and its magnitude will be less than the magnitude of \p y.
 * \return
 * - Returns the floating-point remainder of \p x / \p y.
 * - fmodf(
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p y) returns 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  if \p y is not zero.
 * - fmodf(\p x, 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns \p x if \p x is finite.
 * - fmodf(\p x, \p y) returns NaN if \p x is 
 * \latexonly $\pm\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus --></m:mo>
 *   <m:mi mathvariant="normal">&#x221E;<!-- &Infinity --></m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  or \p y is zero.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_single
 */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  fmodf(float x, float y) __THROW;
#if defined(__QNX__)
/* redeclare some builtins that QNX uses */
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float _FLog(float, int);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float _FCosh(float, float);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float _FSinh(float, float);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float _FSinx(float, unsigned int, int);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int _FDsign(float);
extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ int _Dsign(double);
#endif
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif
#endif /* _WIN32 && _M_AMD64 */

}

#if !defined(__CUDACC_RTC__)
#include <math.h>
#include <stdlib.h>

#ifndef __CUDA_INTERNAL_SKIP_CPP_HEADERS__
#include <cmath>
#include <cstdlib>
#endif /* __CUDA_INTERNAL_SKIP_CPP_HEADERS__ */
#endif /* __CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__CUDACC_RTC__)

__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(long double x);

__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(long double x);

__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(long double x);

__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(long double x);

#elif defined(__GNUC__)

#undef signbit
#undef isfinite
#undef isnan
#undef isinf

#if defined(__APPLE__)

__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(float x);
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(double x);
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(long double x);

__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(float x); 
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(double x);
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(long double x);

__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(double x) throw();
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 7000
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(float x);
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(long double x);
#else /* !(!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 7000) */
template <typename T>
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool __libcpp_isnan(T) _NOEXCEPT;
inline _LIBCPP_INLINE_VISIBILITY __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isnan(float x) _NOEXCEPT;
inline _LIBCPP_INLINE_VISIBILITY  __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isnan(long double x) _NOEXCEPT;
#endif /* !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 7000 */

__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(double x) throw();
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 7000
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(float x);
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(long double x);
#else /* !(!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 7000) */
template <typename T>
__cudart_builtin__ __DEVICE_FUNCTIONS_DECL__ bool __libcpp_isinf(T) _NOEXCEPT;
inline _LIBCPP_INLINE_VISIBILITY __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isinf(float x) _NOEXCEPT;
inline _LIBCPP_INLINE_VISIBILITY __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isinf(long double x) _NOEXCEPT;
#endif /* !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 7000 */

#else /* __APPLE__ */

#if ((defined _GLIBCXX_MATH_H) && _GLIBCXX_MATH_H) && (__cplusplus >= 201103L)
namespace std {
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool signbit(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool signbit(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool signbit(long double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isfinite(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isfinite(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isfinite(long double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isnan(float x);
/* GCC 6.1 uses ::isnan(double x) for isnan(double x) if the condition is true */
#if _GLIBCXX_HAVE_OBSOLETE_ISNAN && !_GLIBCXX_NO_OBSOLETE_ISINF_ISNAN_DYNAMIC
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(double x) throw();
#else /* !(_GLIBCXX_HAVE_OBSOLETE_ISNAN && !_GLIBCXX_NO_OBSOLETE_ISINF_ISNAN_DYNAMIC) */
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isnan(double x);
#endif /* _GLIBCXX_HAVE_OBSOLETE_ISNAN && !_GLIBCXX_NO_OBSOLETE_ISINF_ISNAN_DYNAMIC */
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isnan(long double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isinf(float x);
/* GCC 6.1 uses ::isinf(double x) for isinf(double x) if the condition is true. */
#if _GLIBCXX_HAVE_OBSOLETE_ISINF && !_GLIBCXX_NO_OBSOLETE_ISINF_ISNAN_DYNAMIC
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(double x) throw();
#else /* !(_GLIBCXX_HAVE_OBSOLETE_ISINF && !_GLIBCXX_NO_OBSOLETE_ISINF_ISNAN_DYNAMIC) */
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isinf(double x);
#endif /* _GLIBCXX_HAVE_OBSOLETE_ISINF && !_GLIBCXX_NO_OBSOLETE_ISINF_ISNAN_DYNAMIC */
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ constexpr bool isinf(long double x);
}

#else /* !(((defined _GLIBCXX_MATH_H) && _GLIBCXX_MATH_H) && (__cplusplus >= 201103L)) */

#if defined(__QNX__)
#if (__QNX__) && !defined(_LIBCPP_VERSION)
/* QNX defines functions in std, need to declare them here */
namespace std {
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool signbit(float x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool signbit(double x);
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool signbit(long double x);
}
#else
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool signbit(const float x);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool signbit(const double x);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool signbit(const long double x);
#endif
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isfinite(const float a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isfinite(const double a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isfinite(const long double a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isnan(const float a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isnan(const double a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isnan(const long double a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isinf(const float a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isinf(const double a);
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isinf(const long double a);
#else /* ! __QNX__ */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(const float x);
#if defined(__ICC)
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(const double x) throw();
#else /* !__ICC */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(const double x);
#endif /* __ICC */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int signbit(const long double x);

__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(const float x);
#if defined(__ICC)
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(const double x) throw();
#else /* !__ICC */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(const double x);
#endif /* __ICC */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isfinite(const long double x);

#if (defined(__ANDROID__) || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000
template <typename T>
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool __libcpp_isnan(T) _NOEXCEPT;
inline _LIBCPP_INLINE_VISIBILITY __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isnan(float x) _NOEXCEPT;
#else /* !((defined(__ANDROID__)  || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000) */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(float x);
#endif /* (defined(__ANDROID__)  || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000 */
#if defined(__ANDROID__) || defined(__HORIZON__)
#if !defined(_LIBCPP_VERSION)
__forceinline__
#endif  /* !defined(_LIBCPP_VERSION) */
#if _LIBCPP_VERSION >= 7000
#ifdef _LIBCPP_PREFERRED_OVERLOAD
_LIBCPP_INLINE_VISIBILITY _LIBCPP_PREFERRED_OVERLOAD __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isnan(double x) _NOEXCEPT;
#endif /* _LIBCPP_PREFERRED_OVERLOAD */
#else /* _LIBCPP_VERSION < 7000 */
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(double x);
#endif /* _LIBCPP_VERSION >= 7000 */
#else /* !(__ANDROID__ || __HORIZON__) */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(double x) throw();
#endif /* __ANDROID__ */
#if (defined(__ANDROID__) || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000
inline _LIBCPP_INLINE_VISIBILITY  __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isnan(long double x) _NOEXCEPT;
#else /* !( (defined(__ANDROID__) || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000) */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isnan(long double x);
#endif /* (defined(__ANDROID__) || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000 */

#if (defined(__ANDROID__) || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000
template <typename T>
__cudart_builtin__ __DEVICE_FUNCTIONS_DECL__ bool __libcpp_isinf(T) _NOEXCEPT;
inline _LIBCPP_INLINE_VISIBILITY __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isinf(float x) _NOEXCEPT;
#else /* !( (defined(__ANDROID__)  || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000) */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(float x);
#endif /* (defined(__ANDROID__) || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000 */

#if defined(__ANDROID__) || defined(__HORIZON__)
#if !defined(_LIBCPP_VERSION)
__forceinline__
#endif  /* !defined(_LIBCPP_VERSION) */
#if _LIBCPP_VERSION >= 7000
#ifdef _LIBCPP_PREFERRED_OVERLOAD
_LIBCPP_INLINE_VISIBILITY _LIBCPP_PREFERRED_OVERLOAD __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isinf(double x) _NOEXCEPT;
#endif /* _LIBCPP_PREFERRED_OVERLOAD */
#else /* _LIBCPP_VERSION < 7000 */
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(double x);
#endif /* _LIBCPP_VERSION >= 7000 */
#else /* ! (__ANDROID__  || __HORIZON__) */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(double x) throw();
#endif /* __ANDROID__ || __HORIZON__ */
#if (defined(__ANDROID__)  || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000
inline _LIBCPP_INLINE_VISIBILITY __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool isinf(long double x) _NOEXCEPT;
#else /* !( (defined(__ANDROID__)  || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000) */
__forceinline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ int isinf(long double x);
#endif  /* (defined(__ANDROID__)  || defined(__HORIZON__)) && _LIBCPP_VERSION >= 8000 */
#endif /* __QNX__  */

#endif /* ((defined _GLIBCXX_MATH_H) && _GLIBCXX_MATH_H) && (__cplusplus >= 201103L) */
#endif /* __APPLE__ */

#if !defined(_LIBCPP_VERSION)
#if defined(__clang__)
#if __has_include(<ext/random>)
#define __NV_GLIBCXX_VERSION 40800
#endif /* __has_include(<random>) */
#endif /* __clang__ */

#if !defined(__NV_GLIBCXX_VERSION)
#define __NV_GLIBCXX_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) 
#endif /* !__NV_GLIBCXX_VERSION */
#endif /* !defined(_LIBCPP_VERSION) */

#if !defined(__HORIZON__) || !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 3800
#if defined(__arm__) && !defined(_STLPORT_VERSION) && !_GLIBCXX_USE_C99
#if !defined(__ANDROID__) || (defined(__NV_GLIBCXX_VERSION) && __NV_GLIBCXX_VERSION < 40800)

#if defined(__QNX__)
/* QNX defines functions in std, need to declare them here */
namespace std {
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs (long long int a);
}
#elif defined(__HORIZON__)
#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif
_LIBCPP_BEGIN_NAMESPACE_STD
__DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs (long long int a) throw();
_LIBCPP_END_NAMESPACE_STD
#else
static __inline__ __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs(long long int a);
#endif /* __QNX__ || __HORIZON__*/

#endif /* !__ANDROID__ || (defined(__NV_GLIBCXX_VERSION) && __NV_GLIBCXX_VERSION < 40800) */
#endif /* __arm__ && !_STLPORT_VERSION && !_GLIBCXX_USE_C99 */
#endif /* !defined(__HORIZON__) || !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 3800 */

#if defined(__NV_GLIBCXX_VERSION) && __NV_GLIBCXX_VERSION < 40800 && !defined(__ibmxl__)

#if !defined(_STLPORT_VERSION)
namespace __gnu_cxx
{
#endif /* !_STLPORT_VERSION */

extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs(long long int a);

#if !defined(_STLPORT_VERSION)
}
#endif /* !_STLPORT_VERSION */

#endif /* defined(__NV_GLIBCXX_VERSION) && __NV_GLIBCXX_VERSION < 40800 && !__ibmxl__ */

namespace std
{
  template<typename T> extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ T __pow_helper(T, int);
  template<typename T> extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ T __cmath_power(T, unsigned int);
}

using std::abs;
using std::fabs;
using std::ceil;
using std::floor;
using std::sqrt;
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 3800
using std::pow;
#endif /* !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 3800 */
using std::log;
using std::log10;
using std::fmod;
using std::modf;
using std::exp;
using std::frexp;
using std::ldexp;
using std::asin;
using std::sin;
using std::sinh;
using std::acos;
using std::cos;
using std::cosh;
using std::atan;
using std::atan2;
using std::tan;
using std::tanh;

#elif defined(_WIN32)

extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __CUDA_MATH_CRTIMP double __cdecl _hypot(double x, double y);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __CUDA_MATH_CRTIMP float  __cdecl _hypotf(float x, float y);

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
static __inline__ __DEVICE_FUNCTIONS_DECL__ int signbit(long double a);
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#if _MSC_VER >= 1900
#define __SIGNBIT_THROW throw()
#else
#define __SIGNBIT_THROW
#endif
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ bool signbit(long double) __SIGNBIT_THROW;
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __device_builtin__ __CUDA_MATH_CRTIMP int _ldsign(long double);
#undef __SIGNBIT_THROW
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is negative. 
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE signbit(double a);
#undef __RETURN_TYPE 
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
#if _MSC_VER >= 1900
#define __SIGNBIT_THROW throw()
#else
#define __SIGNBIT_THROW
#endif
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is negative. 
 */
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __RETURN_TYPE signbit(double) __SIGNBIT_THROW;
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __device_builtin__ __CUDA_MATH_CRTIMP int _dsign(double);
#undef __RETURN_TYPE 
#undef __SIGNBIT_THROW
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'.  Returns a nonzero value 
 * if and only if \p a is negative.  
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE signbit(float a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
#if _MSC_VER >= 1900
#define __SIGNBIT_THROW throw()
#else
#define __SIGNBIT_THROW
#endif
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Return the sign bit of the input.
 *
 * Determine whether the floating-point value \p a is negative.
 *
 * \return
 * Reports the sign bit of all values including infinities, zeros, and NaNs.
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is negative.
 * - With other host compilers: __RETURN_TYPE is 'int'.  Returns a nonzero value 
 * if and only if \p a is negative.  
 */
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __RETURN_TYPE signbit(float) __SIGNBIT_THROW;
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ __device_builtin__ __CUDA_MATH_CRTIMP int _fdsign(float);
#undef __RETURN_TYPE
#undef __SIGNBIT_THROW
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
static __inline__ __DEVICE_FUNCTIONS_DECL__ int isinf(long double a);
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isinf(long double a);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 * \return
 * - With Visual Studio 2013 host compiler: Returns true if and only 
 * if \p a is a infinite value.
 * - With other host compilers: Returns a nonzero value if and only 
 * if \p a is a infinite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isinf(double a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 * \return
 * - With Visual Studio 2013 host compiler: Returns true if and only 
 * if \p a is a infinite value.
 * - With other host compilers: Returns a nonzero value if and only 
 * if \p a is a infinite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isinf(double a);
#undef __RETURN_TYPE
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is a infinite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a nonzero 
 * value if and only if \p a is a infinite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isinf(float a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * \brief Determine whether argument is infinite.
 *
 * Determine whether the floating-point value \p a is an infinite value
 * (positive or negative).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns 
 * true if and only if \p a is a infinite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a nonzero 
 * value if and only if \p a is a infinite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isinf(float a);
#undef __RETURN_TYPE
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
static __inline__ __DEVICE_FUNCTIONS_DECL__ int isnan(long double a);
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isnan(long double a);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isnan(double a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isnan(double a);
#undef __RETURN_TYPE
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isnan(float a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_SINGLE
 * 
 * 
 * \brief Determine whether argument is a NaN.
 *
 * Determine whether the floating-point value \p a is a NaN.
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. 
 * Returns true if and only if \p a is a NaN value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns a 
 * nonzero value if and only if \p a is a NaN value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isnan(float a);
#undef __RETURN_TYPE
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
static __inline__ __DEVICE_FUNCTIONS_DECL__ int isfinite(long double a);
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
static __inline__ __DEVICE_FUNCTIONS_DECL__ bool isfinite(long double a);
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isfinite(double a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_DOUBLE
 * 
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isfinite(double a);
#undef __RETURN_TYPE
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#define __RETURN_TYPE int
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isfinite(float a);
#undef __RETURN_TYPE
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
#define __RETURN_TYPE bool
/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Determine whether argument is finite.
 *
 * Determine whether the floating-point value \p a is a finite value
 * (zero, subnormal, or normal and not infinity or NaN).
 *
 * \return
 * - With Visual Studio 2013 host compiler: __RETURN_TYPE is 'bool'. Returns
 * true if and only if \p a is a finite value.
 * - With other host compilers: __RETURN_TYPE is 'int'. Returns 
 * a nonzero value if and only if \p a is a finite value.
 */
static __inline__ __DEVICE_FUNCTIONS_DECL__ __RETURN_TYPE isfinite(float a);
#undef __RETURN_TYPE
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
template<class T> extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ T _Pow_int(T, int);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs(long long int);
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
template<class T> extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ T _Pow_int(T, int) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs(long long int) throw();
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

#endif /* __CUDACC_RTC__ */

#if __cplusplus >= 201103L
#define __NV_NOEXCEPT noexcept
#else /* !__cplusplus >= 201103L */
#define __NV_NOEXCEPT throw()
#endif /* __cplusplus >= 201103L */

#if defined(_LIBCPP_VERSION) && defined(_LIBCPP_BEGIN_NAMESPACE_STD) && !defined(_STLPORT_VERSION)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-extensions"
#endif /* __clang__ */
#if _LIBCPP_VERSION < 3800
_LIBCPP_BEGIN_NAMESPACE_STD
#endif /* _LIBCPP_VERSION < 3800 */
#elif defined(__GNUC__) && !defined(_STLPORT_VERSION)
namespace std {
#endif /* defined(_LIBCPP_VERSION) && defined(_LIBCPP_BEGIN_NAMESPACE_STD) && !defined(_STLPORT_VERSION) ||
          __GNUC__ && !_STLPORT_VERSION */

#if defined(__CUDACC_RTC__) || defined(__GNUC__)

#if defined(__CUDACC_RTC__) || \
    (defined(__NV_GLIBCXX_VERSION) && __NV_GLIBCXX_VERSION >= 40800) || \
    defined(__ibmxl__)
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int abs(long long int);
#endif /* __CUDACC__RTC__ ||
          (defined(__NV_GLIBCXX_VERSION) && __NV_GLIBCXX_VERSION >= 40800) ||
          __ibmxl__ */

#endif /* __CUDACC_RTC__ || __GNUC__ */

#if defined(__CUDACC_RTC__) || \
    (!defined(_MSC_VER) || _MSC_VER < 1800) && \
    (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION < 1101))
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long int __cdecl abs(long int);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl abs(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ double   __cdecl abs(double);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl fabs(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl ceil(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl floor(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl sqrt(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl pow(float, float);

#if !defined(__QNX__)
     
#if defined(__GNUC__) && __cplusplus >= 201103L && !defined(_LIBCPP_VERSION)
template<typename _Tp, typename _Up>
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__
typename __gnu_cxx::__promote_2<_Tp, _Up>::__type pow(_Tp, _Up);
#else  /* !(defined(__GNUC__) && __cplusplus >= 201103L && !defined(_LIBCPP_VERSION)) */
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl pow(float, int);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ double   __cdecl pow(double, int);
#endif  /* defined(__GNUC__) && __cplusplus >= 201103L && !defined(_LIBCPP_VERSION) */
     
#endif  /* !defined(__QNX__) */

extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl log(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl log10(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl fmod(float, float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl modf(float, float*);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl exp(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl frexp(float, int*);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl ldexp(float, int);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl asin(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl sin(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl sinh(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl acos(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl cos(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl cosh(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl atan(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl atan2(float, float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl tan(float);
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl tanh(float);
#else /* __CUDACC_RTC__ ||
         (!defined(_MSC_VER) || _MSC_VER < 1800) &&
         (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION < 1101)) */
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long int __cdecl abs(long int) throw();
#if defined(_LIBCPP_VERSION)
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ long long int __cdecl abs(long long int) throw();
#endif /* defined(_LIBCPP_VERSION) */
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl abs(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ double   __cdecl abs(double) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl fabs(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl ceil(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl floor(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl sqrt(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl pow(float, float) throw();
#if defined(_LIBCPP_VERSION)
#if (defined (__ANDROID__) || defined(__HORIZON__)) && (_LIBCPP_VERSION >= 9000)
template <class _A1, class _A2>
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__
typename std::_EnableIf
<
    std::is_arithmetic<_A1>::value &&
    std::is_arithmetic<_A2>::value,
    std::__promote<_A1, _A2>
>::type pow(_A1 __lcpp_x, _A2 __lcpp_y) __NV_NOEXCEPT;
#elif (defined(__APPLE__) && __clang_major__ >= 7) || _LIBCPP_VERSION >= 3800 || defined(__QNX__)
template <class _Tp, class _Up>
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__
typename std::__lazy_enable_if <
  std::is_arithmetic<_Tp>::value && std::is_arithmetic<_Up>::value,
  std::__promote<_Tp, _Up>
>::type pow(_Tp __x, _Up __y) __NV_NOEXCEPT;
#else /* !((__APPLE__ && __clang_major__ >= 7) || _LIBCPP_VERSION >= 3800) */
template <class _Tp, class _Up>
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__
typename enable_if <
  std::is_arithmetic<_Tp>::value && std::is_arithmetic<_Up>::value,
  typename std::__promote<_Tp, _Up>::type
>::type pow(_Tp __x, _Up __y) __NV_NOEXCEPT;
#endif /* (__APPLE__ && __clang_major__ >= 7) || _LIBCPP_VERSION >= 3800 */
#else /* !defined(_LIBCPP_VERSION) */
#if !(defined(__GNUC__) && __cplusplus >= 201103L)
#if (defined(_MSC_VER) && (_MSC_VER >= 1928)) && !(defined __CUDA_INTERNAL_SKIP_CPP_HEADERS__)
template <class _Ty1, class _Ty2, ::std:: enable_if_t< ::std:: is_arithmetic_v<_Ty1> && ::std:: is_arithmetic_v<_Ty2>, int> > [[nodiscard]] __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ ::std:: _Common_float_type_t<_Ty1, _Ty2> __cdecl pow(_Ty1 _Left, _Ty2 _Right) noexcept;
#else
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl pow(float, int) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ double   __cdecl pow(double, int) throw();
#endif /* (defined(_MSC_VER) && (_MSC_VER >= 1928)) && !(defined __CUDA_INTERNAL_SKIP_CPP_HEADERS__) */
#endif /* !(defined(__GNUC__) && __cplusplus >= 201103L) */
#endif /* defined(_LIBCPP_VERSION) */
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl log(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl log10(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl fmod(float, float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl modf(float, float*) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl exp(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl frexp(float, int*) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl ldexp(float, int) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl asin(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl sin(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl sinh(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl acos(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl cos(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl cosh(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl atan(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl atan2(float, float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl tan(float) throw();
extern __DEVICE_FUNCTIONS_DECL__ __cudart_builtin__ float    __cdecl tanh(float) throw();
#endif /* __CUDACC_RTC__ ||
          (!defined(_MSC_VER) || _MSC_VER < 1800) &&
          (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION < 1101)) */

#if defined(_LIBCPP_VERSION) && defined(_LIBCPP_END_NAMESPACE_STD) && !defined(_STLPORT_VERSION)
#if _LIBCPP_VERSION < 3800
_LIBCPP_END_NAMESPACE_STD
#endif /* _LIBCPP_VERSION < 3800 */
#if defined(__clang__)
#pragma clang diagnostic pop
#endif /* __clang__ */
#elif defined(__GNUC__) && !defined(_STLPORT_VERSION)
}
#endif /* defined(_LIBCPP_VERSION) && defined(_LIBCPP_BEGIN_NAMESPACE_STD) && !defined(_STLPORT_VERSION) ||
          __GNUC__ && !_STLPORT_VERSION */

#undef __DEVICE_FUNCTIONS_DECL__
#undef __NV_NOEXCEPT

#if defined(__CUDACC_RTC__)
#define __MATH_FUNCTIONS_DECL__ __host__ __device__
#define __MATH_FUNCTIONS_DEVICE_DECL__ __device__
#else /* __CUDACC_RTC__ */
#define __MATH_FUNCTIONS_DECL__ static inline __host__ __device__ __cudart_builtin__
#define __MATH_FUNCTIONS_DEVICE_DECL__ static inline __device__ __cudart_builtin__
#endif /* __CUDACC_RTC__ */

#if (!defined(_MSC_VER) || _MSC_VER < 1800)
#if defined(__QNX__) || (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 3800)
#if defined(__QNX__) && (!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 8000)
#if defined(_LIBCPP_VERSION)
#define __NV_NOEXCEPT _NOEXCEPT
_LIBCPP_BEGIN_NAMESPACE_STD
#else
#define __NV_NOEXCEPT
namespace std {
__host__ __device__ __cudart_builtin__ int ilogbf(float a);
#endif
#else /* !(defined(__QNX__) && (!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 8000)) */
#define __NV_NOEXCEPT _NOEXCEPT
#endif /* defined(__QNX__) && (!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 8000) */
__host__ __device__ __cudart_builtin__ float logb(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ int ilogb(float a) __NV_NOEXCEPT;

__host__ __device__ __cudart_builtin__ float scalbn(float a, int b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float scalbln(float a, long int b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float exp2(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float expm1(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float log2(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float log1p(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float acosh(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float asinh(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float atanh(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float hypot(float a, float b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float cbrt(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float erf(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float erfc(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float lgamma(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float tgamma(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float copysign(float a, float b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float nextafter(float a, float b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float remainder(float a, float b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float remquo(float a, float b, int *quo) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float round(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ long int lround(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ long long int llround(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float trunc(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float rint(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ long int lrint(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ long long int llrint(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float nearbyint(float a) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float fdim(float a, float b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float fma(float a, float b, float c) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float fmax(float a, float b) __NV_NOEXCEPT;
__host__ __device__ __cudart_builtin__ float fmin(float a, float b) __NV_NOEXCEPT;
#if defined(__QNX__) && (!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 8000)
#if defined(_LIBCPP_VERSION)
_LIBCPP_END_NAMESPACE_STD
using _VSTD::logb;
using _VSTD::ilogb;
using _VSTD::scalbn;
using _VSTD::scalbln;
using _VSTD::exp2;
using _VSTD::expm1;
using _VSTD::log2;
using _VSTD::log1p;
using _VSTD::acosh;
using _VSTD::asinh;
using _VSTD::atanh;
using _VSTD::hypot;
using _VSTD::cbrt;
using _VSTD::erf;
using _VSTD::erfc;
using _VSTD::lgamma;
using _VSTD::tgamma;
using _VSTD::copysign;
using _VSTD::nextafter;
using _VSTD::remainder;
using _VSTD::remquo;
using _VSTD::round;
using _VSTD::lround;
using _VSTD::llround;
using _VSTD::trunc;
using _VSTD::rint;
using _VSTD::lrint;
using _VSTD::llrint;
using _VSTD::nearbyint;
using _VSTD::fdim;
using _VSTD::fma;
using _VSTD::fmax;
using _VSTD::fmin;
#else
}
#endif
#endif /* defined(__QNX__) && (!defined(_LIBCPP_VERSION) || _LIBCPP_VERSION < 8000) */
#undef __NV_NOEXCEPT
#else /* !(defined(__QNX__ ) || (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 3800)) */
#if ((defined _GLIBCXX_MATH_H) && _GLIBCXX_MATH_H) && (__cplusplus >= 201103L)
namespace std {
__host__ __device__ __cudart_builtin__ constexpr float logb(float a);
__host__ __device__ __cudart_builtin__ constexpr int ilogb(float a);
__host__ __device__ __cudart_builtin__ constexpr float scalbn(float a, int b);
__host__ __device__ __cudart_builtin__ constexpr float scalbln(float a, long int b);
__host__ __device__ __cudart_builtin__ constexpr float exp2(float a);
__host__ __device__ __cudart_builtin__ constexpr float expm1(float a);
__host__ __device__ __cudart_builtin__ constexpr float log2(float a);
__host__ __device__ __cudart_builtin__ constexpr float log1p(float a);
__host__ __device__ __cudart_builtin__ constexpr float acosh(float a);
__host__ __device__ __cudart_builtin__ constexpr float asinh(float a);
__host__ __device__ __cudart_builtin__ constexpr float atanh(float a);
__host__ __device__ __cudart_builtin__ constexpr float hypot(float a, float b);
__host__ __device__ __cudart_builtin__ constexpr float cbrt(float a);
__host__ __device__ __cudart_builtin__ constexpr float erf(float a);
__host__ __device__ __cudart_builtin__ constexpr float erfc(float a);
__host__ __device__ __cudart_builtin__ constexpr float lgamma(float a);
__host__ __device__ __cudart_builtin__ constexpr float tgamma(float a);
__host__ __device__ __cudart_builtin__ constexpr float copysign(float a, float b);
__host__ __device__ __cudart_builtin__ constexpr float nextafter(float a, float b);
__host__ __device__ __cudart_builtin__ constexpr float remainder(float a, float b);
__host__ __device__ __cudart_builtin__ float remquo(float a, float b, int *quo);
__host__ __device__ __cudart_builtin__ constexpr float round(float a);
__host__ __device__ __cudart_builtin__ constexpr long int lround(float a);
__host__ __device__ __cudart_builtin__ constexpr long long int llround(float a);
__host__ __device__ __cudart_builtin__ constexpr float trunc(float a);
__host__ __device__ __cudart_builtin__ constexpr float rint(float a);
__host__ __device__ __cudart_builtin__ constexpr long int lrint(float a);
__host__ __device__ __cudart_builtin__ constexpr long long int llrint(float a);
__host__ __device__ __cudart_builtin__ constexpr float nearbyint(float a);
__host__ __device__ __cudart_builtin__ constexpr float fdim(float a, float b);
__host__ __device__ __cudart_builtin__ constexpr float fma(float a, float b, float c);
__host__ __device__ __cudart_builtin__ constexpr float fmax(float a, float b);
__host__ __device__ __cudart_builtin__ constexpr float fmin(float a, float b);
}
#else /* !(((defined _GLIBCXX_MATH_H) && _GLIBCXX_MATH_H) && (__cplusplus >= 201103L)) */
__MATH_FUNCTIONS_DECL__ float logb(float a);

__MATH_FUNCTIONS_DECL__ int ilogb(float a);

__MATH_FUNCTIONS_DECL__ float scalbn(float a, int b);

__MATH_FUNCTIONS_DECL__ float scalbln(float a, long int b);

__MATH_FUNCTIONS_DECL__ float exp2(float a);

__MATH_FUNCTIONS_DECL__ float expm1(float a);

__MATH_FUNCTIONS_DECL__ float log2(float a);

__MATH_FUNCTIONS_DECL__ float log1p(float a);

__MATH_FUNCTIONS_DECL__ float acosh(float a);

__MATH_FUNCTIONS_DECL__ float asinh(float a);

__MATH_FUNCTIONS_DECL__ float atanh(float a);

__MATH_FUNCTIONS_DECL__ float hypot(float a, float b);

__MATH_FUNCTIONS_DECL__ float cbrt(float a);

__MATH_FUNCTIONS_DECL__ float erf(float a);

__MATH_FUNCTIONS_DECL__ float erfc(float a);

__MATH_FUNCTIONS_DECL__ float lgamma(float a);

__MATH_FUNCTIONS_DECL__ float tgamma(float a);

__MATH_FUNCTIONS_DECL__ float copysign(float a, float b);

__MATH_FUNCTIONS_DECL__ float nextafter(float a, float b);

__MATH_FUNCTIONS_DECL__ float remainder(float a, float b);

__MATH_FUNCTIONS_DECL__ float remquo(float a, float b, int *quo);

__MATH_FUNCTIONS_DECL__ float round(float a);

__MATH_FUNCTIONS_DECL__ long int lround(float a);

__MATH_FUNCTIONS_DECL__ long long int llround(float a);

__MATH_FUNCTIONS_DECL__ float trunc(float a);

__MATH_FUNCTIONS_DECL__ float rint(float a);

__MATH_FUNCTIONS_DECL__ long int lrint(float a);

__MATH_FUNCTIONS_DECL__ long long int llrint(float a);

__MATH_FUNCTIONS_DECL__ float nearbyint(float a);

__MATH_FUNCTIONS_DECL__ float fdim(float a, float b);

__MATH_FUNCTIONS_DECL__ float fma(float a, float b, float c);

__MATH_FUNCTIONS_DECL__ float fmax(float a, float b);

__MATH_FUNCTIONS_DECL__ float fmin(float a, float b);
#endif /* ((defined _GLIBCXX_MATH_H) && _GLIBCXX_MATH_H) && (__cplusplus >= 201103L) */
#endif /* defined(__QNX__) || (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 3800) */
#else /* (!defined(_MSC_VER) || _MSC_VER < 1800) */
extern __host__ __device__ __cudart_builtin__ float __cdecl logb(float) throw();
extern __host__ __device__ __cudart_builtin__ int   __cdecl ilogb(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl scalbn(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl scalbln(float, long int) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl exp2(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl expm1(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl log2(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl log1p(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl acosh(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl asinh(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl atanh(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl hypot(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl cbrt(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl erf(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl erfc(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl lgamma(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl tgamma(float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl copysign(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl nextafter(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl remainder(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl remquo(float, float, int *) throw();
extern __host__ __device__ __cudart_builtin__ float __cdecl round(float) throw();
extern __host__ __device__ __cudart_builtin__ long int      __cdecl lround(float) throw();
extern __host__ __device__ __cudart_builtin__ long long int __cdecl llround(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl trunc(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl rint(float) throw();
extern __host__ __device__ __cudart_builtin__ long int      __cdecl lrint(float) throw();
extern __host__ __device__ __cudart_builtin__ long long int __cdecl llrint(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl nearbyint(float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fdim(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fma(float, float, float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fmax(float, float) throw();
extern __host__ __device__ __cudart_builtin__ float         __cdecl fmin(float, float) throw();
#endif /* (!defined(_MSC_VER) || _MSC_VER < 1800) */

__MATH_FUNCTIONS_DECL__ float exp10(const float a);

__MATH_FUNCTIONS_DECL__ float rsqrt(const float a);

__MATH_FUNCTIONS_DECL__ float rcbrt(const float a);

__MATH_FUNCTIONS_DECL__ float sinpi(const float a);

__MATH_FUNCTIONS_DECL__ float cospi(const float a);

__MATH_FUNCTIONS_DECL__ void sincospi(const float a, float *const sptr, float *const cptr);

__MATH_FUNCTIONS_DECL__ void sincos(const float a, float *const sptr, float *const cptr);

__MATH_FUNCTIONS_DECL__ float j0(const float a);

__MATH_FUNCTIONS_DECL__ float j1(const float a);

__MATH_FUNCTIONS_DECL__ float jn(const int n, const float a);

__MATH_FUNCTIONS_DECL__ float y0(const float a);

__MATH_FUNCTIONS_DECL__ float y1(const float a);

__MATH_FUNCTIONS_DECL__ float yn(const int n, const float a);

__MATH_FUNCTIONS_DEVICE_DECL__ float cyl_bessel_i0(const float a);

__MATH_FUNCTIONS_DEVICE_DECL__ float cyl_bessel_i1(const float a);

__MATH_FUNCTIONS_DECL__ float erfinv(const float a);

__MATH_FUNCTIONS_DECL__ float erfcinv(const float a);

__MATH_FUNCTIONS_DECL__ float normcdfinv(const float a);

__MATH_FUNCTIONS_DECL__ float normcdf(const float a);

__MATH_FUNCTIONS_DECL__ float erfcx(const float a);

__MATH_FUNCTIONS_DECL__ double copysign(const double a, const float b);

__MATH_FUNCTIONS_DECL__ double copysign(const float a, const double b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ unsigned int min(const unsigned int a, const unsigned int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p int and \p unsigned \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned int min(const int a, const unsigned int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p int and \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned int min(const unsigned int a, const int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ long int min(const long int a, const long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ unsigned long int min(const unsigned long int a, const unsigned long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p long \p int and \p unsigned \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long int min(const long int a, const unsigned long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p long \p int and \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long int min(const unsigned long int a, const long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p long \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ long long int min(const long long int a, const long long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p long \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ unsigned long long int min(const unsigned long long int a, const unsigned long long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p long \p long \p int and \p unsigned \p long \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long long int min(const long long int a, const unsigned long long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the minimum value of the input \p unsigned \p long \p long \p int and \p long \p long \p int arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long long int min(const unsigned long long int a, const long long int b);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the minimum value of the input \p float arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 * Behavior is equivalent to ::fminf() function.
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ float min(const float a, const float b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the minimum value of the input \p float arguments.
 *
 * Calculate the minimum value of the arguments \p a and \p b.
 * Behavior is equivalent to ::fmin() function.
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ double min(const double a, const double b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the minimum value of the input \p float and \p double arguments.
 *
 * Convert \p float argument \p a to \p double, followed by ::fmin().
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ double min(const float a, const double b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the minimum value of the input \p double and \p float arguments.
 *
 * Convert \p float argument \p b to \p double, followed by ::fmin().
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ double min(const double a, const float b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ unsigned int max(const unsigned int a, const unsigned int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p int and \p unsigned \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned int max(const int a, const unsigned int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p int and \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned int max(const unsigned int a, const int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ long int max(const long int a, const long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ unsigned long int max(const unsigned long int a, const unsigned long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p long \p int and \p unsigned \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long int max(const long int a, const unsigned long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p long \p int and \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long int max(const unsigned long int a, const long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p long \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ long long int max(const long long int a, const long long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p long \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 */
__MATH_FUNCTIONS_DECL__ unsigned long long int max(const unsigned long long int a, const unsigned long long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p long \p long \p int and \p unsigned \p long \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long long int max(const long long int a, const unsigned long long int b);

/**
 * \ingroup CUDA_MATH_INT
 * \brief Calculate the maximum value of the input \p unsigned \p long \p long \p int and \p long \p long \p int arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b, perform integer promotion first.
 */
__MATH_FUNCTIONS_DECL__ unsigned long long int max(const unsigned long long int a, const long long int b);

/**
 * \ingroup CUDA_MATH_SINGLE
 * \brief Calculate the maximum value of the input \p float arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 * Behavior is equivalent to ::fmaxf() function.
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ float max(const float a, const float b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the maximum value of the input \p float arguments.
 *
 * Calculate the maximum value of the arguments \p a and \p b.
 * Behavior is equivalent to ::fmax() function.
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ double max(const double a, const double b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the maximum value of the input \p float and \p double arguments.
 *
 * Convert \p float argument \p a to \p double, followed by ::fmax().
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ double max(const float a, const double b);

/**
 * \ingroup CUDA_MATH_DOUBLE
 * \brief Calculate the maximum value of the input \p double and \p float arguments.
 *
 * Convert \p float argument \p b to \p double, followed by ::fmax().
 *
 * Note, this is different from \p std:: specification
 */
__MATH_FUNCTIONS_DECL__ double max(const double a, const float b);

#undef __MATH_FUNCTIONS_DECL__
#undef __MATH_FUNCTIONS_DEVICE_DECL__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern "C"{
inline __device__ void *__nv_aligned_device_malloc(size_t size, size_t align)
{
  __device__ void *__nv_aligned_device_malloc_impl(size_t, size_t);
  return __nv_aligned_device_malloc_impl(size, align);
}
}
#endif /* __cplusplus && __CUDACC__ */
#if !defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
* ONLY FOR HOST CODE! NOT FOR DEVICE EXECUTION                                 *
*                                                                              *
*******************************************************************************/

#include <crt/func_macro.h>

#if defined(_WIN32)

#pragma warning(disable : 4211)

#endif /* _WIN32 */

__func__(double rsqrt(double a));

__func__(double rcbrt(double a));

__func__(double sinpi(double a));

__func__(double cospi(double a));

__func__(void sincospi(double a, double *sptr, double *cptr));

__func__(double erfinv(double a));

__func__(double erfcinv(double a));

__func__(double normcdfinv(double a));

__func__(double normcdf(double a));

__func__(double erfcx(double a));

__func__(float rsqrtf(float a));

__func__(float rcbrtf(float a));

__func__(float sinpif(float a));

__func__(float cospif(float a));

__func__(void sincospif(float a, float *sptr, float *cptr));

__func__(float erfinvf(float a));

__func__(float erfcinvf(float a));

__func__(float normcdfinvf(float a));

__func__(float normcdff(float a));

__func__(float erfcxf(float a));

__func__(int min(int a, int b));

__func__(unsigned int umin(unsigned int a, unsigned int b));

__func__(long long int llmin(long long int a, long long int b));

__func__(unsigned long long int ullmin(unsigned long long int a, unsigned long long int b));

__func__(int max(int a, int b));

__func__(unsigned int umax(unsigned int a, unsigned int b));

__func__(long long int llmax(long long int a, long long int b));

__func__(unsigned long long int ullmax(unsigned long long int a, unsigned long long int b));

#if defined(_WIN32) || defined(__APPLE__) || defined (__ANDROID__)

__func__(int __isnan(double a));

#endif /* _WIN32 || __APPLE__ || __ANDROID__ */

#if defined(_WIN32) || defined(__APPLE__) || defined (__QNX__)

__func__(void sincos(double a, double *sptr, double *cptr));

#endif /* _WIN32 || __APPLE__ || __QNX__ */

#if defined(_WIN32) || defined(__APPLE__)

__func__(double exp10(double a));

__func__(float exp10f(float a));

__func__(void sincosf(float a, float *sptr, float *cptr));

__func__(int __isinf(double a));

#endif /* _WIN32 || __APPLE__ */

#if (defined(_WIN32) && (!defined(_MSC_VER) || _MSC_VER < 1800)) || defined (__ANDROID__)

__func__(double log2(double a));

#endif /* (_WIN32 && (!defined(_MSC_VER) || _MSC_VER < 1800)) || __ANDROID__ */

#if defined(_WIN32)

__func__(int __signbit(double a));

__func__(int __finite(double a));

__func__(int __signbitl(long double a));

__func__(int __signbitf(float a));

__func__(int __finitel(long double a));

__func__(int __finitef(float a));

__func__(int __isinfl(long double a));

__func__(int __isinff(float a));

__func__(int __isnanl(long double a));

__func__(int __isnanf(float a));

#endif /* _WIN32 */

#if defined(_WIN32) && (!defined(_MSC_VER) || _MSC_VER < 1800)

__func__(double copysign(double a, double b));

__func__(double fmax(double a, double b));

__func__(double fmin(double a, double b));

__func__(double trunc(double a));

__func__(double round(double a));

__func__(long int lround(double a));

__func__(long long int llround(double a));

__func__(double rint(double a));

__func__(double nearbyint(double a));

__func__(long int lrint(double a));

__func__(long long int llrint(double a));

__func__(double fdim(double a, double b));

__func__(double scalbn(double a, int b));

__func__(double scalbln(double a, long int b));

__func__(double exp2(double a));

__func__(double log1p(double a));

__func__(double expm1(double a));

__func__(double cbrt(double a));

__func__(double acosh(double a));

__func__(double asinh(double a));

__func__(double atanh(double a));

__func__(int ilogb(double a));

__func__(double logb(double a));

__func__(double remquo(double a, double b, int *quo));

__func__(double remainder(double a, double b));

__func__(double fma (double a, double b, double c));

__func__(double nextafter(double a, double b));

__func__(double erf(double a));

__func__(double erfc(double a));

__func__(double lgamma(double a));

__func__(unsigned long long int __internal_host_nan_kernel(const char *s));

__func__(double nan(const char *tagp));

__func__(double __host_tgamma_kernel(double a));

__func__(double __host_stirling_poly(double a));

__func__(double __host_tgamma_stirling(double a));

__func__(double tgamma(double a));

__func__(float fmaxf(float a, float b));

__func__(float fminf(float a, float b));

__func__(float roundf(float a));

__func__(long int lroundf(float a));

__func__(long long int llroundf(float a));

__func__(float truncf(float a));

__func__(float rintf(float a));

__func__(float nearbyintf(float a));

__func__(long int lrintf(float a));

__func__(long long int llrintf(float a));

__func__(float logbf(float a));

__func__(float scalblnf(float a, long int b));

__func__(float log2f(float a));

__func__(float exp2f(float a));

__func__(float acoshf(float a));

__func__(float asinhf(float a));

__func__(float atanhf(float a));

__func__(float cbrtf(float a));

__func__(float expm1f(float a));

__func__(float fdimf(float a, float b));

__func__(float log1pf(float a));

__func__(float scalbnf(float a, int b));

__func__(float fmaf(float a, float b, float c));

__func__(int ilogbf(float a));

__func__(float erff(float a));

__func__(float erfcf(float a));

__func__(float lgammaf(float a));

__func__(float tgammaf(float a));

__func__(float remquof(float a, float b, int *quo));

__func__(float remainderf(float a, float b));

__func__(float copysignf(float a, float b));

__func__(float nextafterf(float a, float b));

__func__(float nanf(const char *tagp));

#endif /* _WIN32 && (!defined(_MSC_VER) || _MSC_VER < 1800) */

#if defined(_WIN32)

#pragma warning(default: 4211)

#endif /* _WIN32 */

#endif /* !__CUDACC__ */

#if !defined(__CUDACC_RTC__)

#include "math_functions.hpp"

#endif /* !__CUDACC_RTC__ */

#endif /* !__MATH_FUNCTIONS_H__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_MATH_FUNCTIONS_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_MATH_FUNCTIONS_H__
#endif
