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

#if !defined(__SM_20_INTRINSICS_H__)
#define __SM_20_INTRINSICS_H__

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

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

#if defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is not valid on compute_70 and above, and should be replaced with "#x"_sync()."\
    "To continue using "#x"(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."
#else
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is deprecated in favor of "#x"_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."
#endif

extern "C"
{
extern __device__ __device_builtin__ void                   __threadfence_system(void);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-to-nearest-even mode.
 *
 * Divides two floating-point values \p x by \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-towards-zero mode.
 *
 * Divides two floating-point values \p x by \p y in round-towards-zero mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-up mode.
 * 
 * Divides two floating-point values \p x by \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-down mode.
 *
 * Divides two floating-point values \p x by \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x / \p y.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 * 
 * Compute the reciprocal of \p x in round-to-nearest-even mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 *
 * Compute the reciprocal of \p x in round-towards-zero mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 * 
 * Compute the reciprocal of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 * 
 * Compute the reciprocal of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns 
 * \latexonly $\frac{1}{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mfrac>
 *     <m:mn>1</m:mn>
 *     <m:mi>x</m:mi>
 *   </m:mfrac>
 * </m:math>
 * </d4p_MathML>\endxmlonly.
 *
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-to-nearest-even mode.
 * 
 * Compute the square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns 
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-towards-zero mode.
 * 
 * Compute the square root of \p x in round-towards-zero mode.
 *
 * \return Returns 
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-up mode.
 * 
 * Compute the square root of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns 
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $\sqrt{x}$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:msqrt>
 *     <m:mi>x</m:mi>
 *   </m:msqrt>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  in round-down mode.
 * 
 * Compute the square root of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns 
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
 * \note_accuracy_double
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rd(double x);
extern __device__ __device_builtin__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__ballot)) unsigned int __ballot(int);
extern __device__ __device_builtin__ int                   __syncthreads_count(int);
extern __device__ __device_builtin__ int                   __syncthreads_and(int);
extern __device__ __device_builtin__ int                   __syncthreads_or(int);
extern __device__ __device_builtin__ long long int         clock64(void);


/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-to-nearest-even mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_rn(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_rn(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-down mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_rd(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_rd(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-up mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_ru(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_ru(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-towards-zero mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_rz(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_rz(float x, float y, float z);


// SM_13 intrinsics

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a double as a 64-bit signed integer.
 *
 * Reinterpret the bits in the double-precision floating-point value \p x
 * as a signed 64-bit integer.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ long long int         __double_as_longlong(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a 64-bit signed integer as a double.
 *
 * Reinterpret the bits in the 64-bit signed integer value \p x as
 * a double-precision floating-point value.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ double                __longlong_as_double(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-to-nearest-even mode.
 *
 * Computes the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-to-nearest-even mode.
 *
 * \return Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - fmaf(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double                __fma_rn(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-towards-zero mode.
 *
 * Computes the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-towards-zero mode.
 *
 * \return Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - fmaf(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double                __fma_rz(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-up mode.
 *
 * Computes the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-up (to positive infinity) mode.
 *
 * \return Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - fmaf(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double                __fma_ru(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single operation in round-down mode.
 *
 * Computes the value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 *   <m:mo>+</m:mo>
 *   <m:mi>z</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  as a single ternary operation, rounding the
 * result once in round-down (to negative infinity) mode.
 *
 * \return Returns the rounded value of 
 * \latexonly $x \times y + z$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm 0$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
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
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>0</m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , 
 * \latexonly $\pm \infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>&#x00B1;<!-- &PlusMinus; --></m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * , \p z) returns NaN.
 * - fmaf(\p x, \p y, 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
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
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 * - fmaf(\p x, \p y, 
 * \latexonly $+\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>+</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * ) returns NaN if
 * \latexonly $x \times y$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mi>x</m:mi>
 *   <m:mo>&#x00D7;<!-- &Multiply; --></m:mo>
 *   <m:mi>y</m:mi>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 *  is an exact 
 * \latexonly $-\infty$ \endlatexonly
 * \xmlonly
 * <d4p_MathML outputclass="xmlonly">
 * <m:math xmlns:m="http://www.w3.org/1998/Math/MathML">
 *   <m:mo>-</m:mo>
 *   <m:mn>&#x221E;<!-- &Infinity; --></m:mn>
 * </m:math>
 * </d4p_MathML>
 * \endxmlonly
 * .
 *
 * \note_accuracy_double
 */
extern __device__ __device_builtin__ double                __fma_rd(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-to-nearest-even mode.
 *
 * Adds two floating-point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dadd_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-towards-zero mode.
 *
 * Adds two floating-point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dadd_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-up mode.
 * 
 * Adds two floating-point values \p x and \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */ 
extern __device__ __device_builtin__ double                __dadd_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-down mode.
 *
 * Adds two floating-point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x + \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dadd_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-to-nearest-even mode.
 *
 * Subtracts two floating-point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dsub_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-towards-zero mode.
 *
 * Subtracts two floating-point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dsub_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-up mode.
 * 
 * Subtracts two floating-point values \p x and \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */ 
extern __device__ __device_builtin__ double                __dsub_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-down mode.
 *
 * Subtracts two floating-point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x - \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dsub_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-to-nearest-even mode.
 *
 * Multiplies two floating-point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-towards-zero mode.
 *
 * Multiplies two floating-point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-up mode.
 * 
 * Multiplies two floating-point values \p x and \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-down mode.
 *
 * Multiplies two floating-point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x * \p y.
 *
 * \note_accuracy_double
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-towards-zero mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-towards-zero mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ int                   __double2int_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ int                   __double2int_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ int                   __double2int_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ unsigned int          __double2uint_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ unsigned int          __double2uint_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ unsigned int          __double2uint_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed 64-bit integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ long long int          __double2ll_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed 64-bit integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ long long int          __double2ll_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed 64-bit integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ long long int          __double2ll_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned 64-bit integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ unsigned long long int __double2ull_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned 64-bit integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ unsigned long long int __double2ull_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned 64-bit integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ unsigned long long int __double2ull_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed int to a double.
 *
 * Convert the signed integer value \p x to a double-precision floating-point value.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __int2double_rn(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned int to a double.
 *
 * Convert the unsigned integer value \p x to a double-precision floating-point value.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __uint2double_rn(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-to-nearest-even mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_rn(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-towards-zero mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-towards-zero mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_rz(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-up mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_ru(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-down mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_rd(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-to-nearest-even mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_rn(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-towards-zero mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-towards-zero mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_rz(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-up mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_ru(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-down mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_rd(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret high 32 bits in a double as a signed integer.
 *
 * Reinterpret the high 32 bits in the double-precision floating-point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ int                    __double2hiint(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret low 32 bits in a double as a signed integer.
 *
 * Reinterpret the low 32 bits in the double-precision floating-point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ int                    __double2loint(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret high and low 32-bit integer values as a double.
 *
 * Reinterpret the integer value of \p hi as the high 32 bits of a 
 * double-precision floating-point value and the integer value of \p lo
 * as the low 32 bits of the same double-precision floating-point value.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ double                 __hiloint2double(int hi, int lo);


}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
__SM_20_INTRINSICS_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__ballot)) unsigned int ballot(bool pred) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ int syncthreads_count(bool pred) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ bool syncthreads_and(bool pred) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ bool syncthreads_or(bool pred) __DEF_IF_HOST

#undef __DEPRECATED__
#undef __WSB_DEPRECATION_MESSAGE

__SM_20_INTRINSICS_DECL__ unsigned int __isGlobal(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ unsigned int __isShared(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ unsigned int __isConstant(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ unsigned int __isLocal(const void *ptr) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_global(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_shared(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_constant(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_local(const void *ptr) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ void * __cvta_global_to_generic(size_t rawbits) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ void * __cvta_shared_to_generic(size_t rawbits) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ void * __cvta_constant_to_generic(size_t rawbits) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ void * __cvta_local_to_generic(size_t rawbits) __DEF_IF_HOST

#endif /* __cplusplus && __CUDACC__ */

#undef __DEF_IF_HOST
#undef __SM_20_INTRINSICS_DECL__

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
#include "sm_20_intrinsics.hpp"
#endif /* !__CUDACC_RTC__ */
#endif /* !__SM_20_INTRINSICS_H__ && defined(__CUDA_ARCH__) */
