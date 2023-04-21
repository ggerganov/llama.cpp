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

/**
* \defgroup CUDA_MATH_INTRINSIC_HALF Half Precision Intrinsics
* This section describes half precision intrinsic functions that are
* only supported in device code.
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_ARITHMETIC Half Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF2_ARITHMETIC Half2 Arithmetic Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_COMPARISON Half Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF2_COMPARISON Half2 Comparison Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_MISC Half Precision Conversion and Data Movement
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF_FUNCTIONS Half Math Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

/**
* \defgroup CUDA_MATH__HALF2_FUNCTIONS Half2 Math Functions
* \ingroup CUDA_MATH_INTRINSIC_HALF
* To use these functions, include the header file \p cuda_fp16.h in your program.
*/

#ifndef __CUDA_FP16_H__
#define __CUDA_FP16_H__

#define ___CUDA_FP16_STRINGIFY_INNERMOST(x) #x
#define __CUDA_FP16_STRINGIFY(x) ___CUDA_FP16_STRINGIFY_INNERMOST(x)

#if defined(__cplusplus)
#if defined(__CUDACC__)
#define __CUDA_FP16_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __host__ __device__ __inline__
#else
#define __CUDA_HOSTDEVICE_FP16_DECL__ static
#endif /* defined(__CUDACC__) */

#define __CUDA_FP16_TYPES_EXIST__

/* Forward-declaration of structures defined in "cuda_fp16.hpp" */

/**
 * \brief half datatype 
 * 
 * \details This structure implements the datatype for storing 
 * half-precision floating-point numbers. The structure implements 
 * assignment operators and type conversions. 
 * 16 bits are being used in total: 1 sign bit, 5 bits for the exponent, 
 * and the significand is being stored in 10 bits. 
 * The total precision is 11 bits. There are 15361 representable 
 * numbers within the interval [0.0, 1.0], endpoints included. 
 * On average we have log10(2**11) ~ 3.311 decimal digits. 
 * 
 * \internal
 * \req IEEE 754-2008 compliant implementation of half-precision 
 * floating-point numbers. 
 * \endinternal
 */
struct __half;

/**
 * \brief half2 datatype
 * 
 * \details This structure implements the datatype for storing two 
 * half-precision floating-point numbers. 
 * The structure implements assignment operators and type conversions. 
 * 
 * \internal
 * \req Vectorified version of half. 
 * \endinternal
 */
struct __half2;

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts double number to half precision in round-to-nearest-even mode
* and returns \p half with converted value.
*
* \details Converts double number \p a to half precision in round-to-nearest-even mode.
* \param[in] a - double. Is only being read.
* \returns half
* \retval a converted to half.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __double2half(const double a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-to-nearest-even mode
* and returns \p half with converted value. 
* 
* \details Converts float number \p a to half precision in round-to-nearest-even mode. 
* \param[in] a - float. Is only being read. 
* \returns half
* \retval a converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-to-nearest-even mode
* and returns \p half with converted value.
*
* \details Converts float number \p a to half precision in round-to-nearest-even mode.
* \param[in] a - float. Is only being read. 
* \returns half
* \retval a converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rn(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-towards-zero mode
* and returns \p half with converted value.
* 
* \details Converts float number \p a to half precision in round-towards-zero mode.
* \param[in] a - float. Is only being read. 
* \returns half
* \retval a converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rz(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-down mode
* and returns \p half with converted value.
* 
* \details Converts float number \p a to half precision in round-down mode.
* \param[in] a - float. Is only being read. 
* 
* \returns half
* \retval a converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rd(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts float number to half precision in round-up mode
* and returns \p half with converted value.
* 
* \details Converts float number \p a to half precision in round-up mode.
* \param[in] a - float. Is only being read. 
* 
* \returns half
* \retval a converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_ru(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts \p half number to float.
* 
* \details Converts half number \p a to float.
* \param[in] a - float. Is only being read. 
* 
* \returns float
* \retval a converted to float. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts input to half precision in round-to-nearest-even mode and
* populates both halves of \p half2 with converted value.
*
* \details Converts input \p a to half precision in round-to-nearest-even mode and
* populates both halves of \p half2 with converted value.
* \param[in] a - float. Is only being read. 
*
* \returns half2
* \retval The \p half2 value with both halves equal to the converted half
* precision number.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float2half2_rn(const float a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both input floats to half precision in round-to-nearest-even
* mode and returns \p half2 with converted values.
*
* \details Converts both input floats to half precision in round-to-nearest-even mode
* and combines the results into one \p half2 number. Low 16 bits of the return
* value correspond to the input \p a, high 16 bits correspond to the input \p
* b.
* \param[in] a - float. Is only being read. 
* \param[in] b - float. Is only being read. 
* 
* \returns half2
* \retval The \p half2 value with corresponding halves equal to the
* converted input floats.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts low 16 bits of \p half2 to float and returns the result
* 
* \details Converts low 16 bits of \p half2 input \p a to 32-bit floating-point number
* and returns the result.
* \param[in] a - half2. Is only being read. 
* 
* \returns float
* \retval The low 16 bits of \p a converted to float.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float __low2float(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts high 16 bits of \p half2 to float and returns the result
* 
* \details Converts high 16 bits of \p half2 input \p a to 32-bit floating-point number
* and returns the result.
* \param[in] a - half2. Is only being read. 
* 
* \returns float
* \retval The high 16 bits of \p a converted to float.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float __high2float(const __half2 a);

#if defined(__CUDACC__)
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both components of float2 number to half precision in
* round-to-nearest-even mode and returns \p half2 with converted values.
* 
* \details Converts both components of float2 to half precision in round-to-nearest
* mode and combines the results into one \p half2 number. Low 16 bits of the
* return value correspond to \p a.x and high 16 bits of the return value
* correspond to \p a.y.
* \param[in] a - float2. Is only being read. 
*  
* \returns half2
* \retval The \p half2 which has corresponding halves equal to the
* converted float2 components.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float22half2_rn(const float2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Converts both halves of \p half2 to float2 and returns the result.
* 
* \details Converts both halves of \p half2 input \p a to float2 and returns the
* result.
* \param[in] a - half2. Is only being read. 
* 
* \returns float2
* \retval a converted to float2.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ float2 __half22float2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-to-nearest-even mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* \retval h converted to a signed integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __half2int_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-towards-zero mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* \retval h converted to a signed integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* \retval h converted to a signed integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __half2int_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed integer in
* round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns int
* \retval h converted to a signed integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __half2int_ru(const __half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-to-nearest-even mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rn(const int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-towards-zero mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __int2half_rz(const int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-down mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __int2half_rd(const int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed integer to a half in round-up mode.
* 
* \details Convert the signed integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __int2half_ru(const int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* \retval h converted to a signed short integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half2short_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-towards-zero mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* \retval h converted to a signed short integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* \retval h converted to a signed short integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half2short_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed short integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed short
* integer in round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* \retval h converted to a signed short integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half2short_ru(const __half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rn(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-towards-zero mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __short2half_rz(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-down mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __short2half_rd(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed short integer to a half in round-up mode.
* 
* \details Convert the signed short integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __short2half_ru(const short int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-to-nearest-even mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned int
* \retval h converted to an unsigned integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-towards-zero mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned int
* \retval h converted to an unsigned integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-down mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
*
* \returns unsigned int
* \retval h converted to an unsigned integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned integer in round-up mode.
*
* \details Convert the half-precision floating-point value \p h to an unsigned integer
* in round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
*
* \returns unsigned int
* \retval h converted to an unsigned integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned int __half2uint_ru(const __half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-to-nearest-even mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn(const unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-towards-zero mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* \retval i converted to half.  
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __uint2half_rz(const unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-down mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __uint2half_rd(const unsigned int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned integer to a half in round-up mode.
* 
* \details Convert the unsigned integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - unsigned int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __uint2half_ru(const unsigned int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-to-nearest-even mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* \retval h converted to an unsigned short integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-towards-zero
* mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-towards-zero mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* \retval h converted to an unsigned short integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-down mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* \retval h converted to an unsigned short integer. 
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned short integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned short
* integer in round-up mode. NaN inputs are converted to 0.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* \retval h converted to an unsigned short integer. 
*/
__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(const __half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-towards-zero
* mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ushort2half_rz(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-down mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ushort2half_rd(const unsigned short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned short integer to a half in round-up mode.
* 
* \details Convert the unsigned short integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ushort2half_ru(const unsigned short int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-to-nearest-even mode. NaN inputs return 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* \retval h converted to an unsigned 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-towards-zero
* mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-towards-zero mode. NaN inputs return 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* \retval h converted to an unsigned 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-down mode. NaN inputs return 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* \retval h converted to an unsigned 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to an unsigned 64-bit integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to an unsigned 64-bit
* integer in round-up mode. NaN inputs return 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned long long int
* \retval h converted to an unsigned 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(const __half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-towards-zero
* mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ull2half_rz(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-down mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half.  
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ull2half_rd(const unsigned long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert an unsigned 64-bit integer to a half in round-up mode.
* 
* \details Convert the unsigned 64-bit integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - unsigned long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ull2half_ru(const unsigned long long int i);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-to-nearest-even
* mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-to-nearest-even mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* \retval h converted to a signed 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ long long int __half2ll_rn(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-towards-zero mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-towards-zero mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* \retval h converted to a signed 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-down mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-down mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* \retval h converted to a signed 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ long long int __half2ll_rd(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a half to a signed 64-bit integer in round-up mode.
* 
* \details Convert the half-precision floating-point value \p h to a signed 64-bit
* integer in round-up mode. NaN inputs return a long long int with hex value of 0x8000000000000000.
* \param[in] h - half. Is only being read. 
* 
* \returns long long int
* \retval h converted to a signed 64-bit integer. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ long long int __half2ll_ru(const __half h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-to-nearest-even
* mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-to-nearest-even mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-towards-zero mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-towards-zero mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
*/
__CUDA_FP16_DECL__ __half __ll2half_rz(const long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-down mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-down mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ll2half_rd(const long long int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Convert a signed 64-bit integer to a half in round-up mode.
* 
* \details Convert the signed 64-bit integer value \p i to a half-precision floating-point
* value in round-up mode.
* \param[in] i - long long int. Is only being read. 
* 
* \returns half
* \retval i converted to half. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ll2half_ru(const long long int i);

/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Truncate input argument to the integral part.
* 
* \details Round \p h to the nearest integer value that does not exceed \p h in
* magnitude.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* \retval The truncated integer value. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half htrunc(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculate ceiling of the input argument.
* 
* \details Compute the smallest integer value not less than \p h.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* \retval The smallest integer value not less than \p h. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hceil(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
* 
* \details Calculate the largest integer value which is less than or equal to \p h.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* \retval The largest integer value which is less than or equal to \p h. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hfloor(const __half h);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Round input to nearest integer value in half-precision floating-point
* number.
* 
* \details Round \p h to the nearest integer value in half-precision floating-point
* format, with halfway cases rounded to the nearest even integer value.
* \param[in] h - half. Is only being read. 
* 
* \returns half
* \retval The nearest integer to \p h. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hrint(const __half h);

/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Truncate \p half2 vector input argument to the integral part.
* 
* \details Round each component of vector \p h to the nearest integer value that does
* not exceed \p h in magnitude.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* \retval The truncated \p h. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculate \p half2 vector ceiling of the input argument.
* 
* \details For each component of vector \p h compute the smallest integer value not less
* than \p h.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* \retval The vector of smallest integers not less than \p h. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculate the largest integer less than or equal to \p h.
* 
* \details For each component of vector \p h calculate the largest integer value which
* is less than or equal to \p h.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* \retval The vector of largest integers which is less than or equal to \p h. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Round input to nearest integer value in half-precision floating-point
* number.
* 
* \details Round each component of \p half2 vector \p h to the nearest integer value in
* half-precision floating-point format, with halfway cases rounded to the
* nearest even integer value.
* \param[in] h - half2. Is only being read. 
* 
* \returns half2
* \retval The vector of rounded integer values. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns \p half2 with both halves equal to the input value.
* 
* \details Returns \p half2 number with both halves equal to the input \p a \p half
* number.
* \param[in] a - half. Is only being read. 
* 
* \returns half2
* \retval The vector which has both its halves equal to the input \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __half2half2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Swaps both halves of the \p half2 input.
* 
* \details Swaps both halves of the \p half2 input and returns a new \p half2 number
* with swapped halves.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* \retval a with its halves being swapped. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts low 16 bits from each of the two \p half2 inputs and combines
* into one \p half2 number. 
* 
* \details Extracts low 16 bits from each of the two \p half2 inputs and combines into
* one \p half2 number. Low 16 bits from input \p a is stored in low 16 bits of
* the return value, low 16 bits from input \p b is stored in high 16 bits of
* the return value. 
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* \retval The low 16 bits of \p a and of \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts high 16 bits from each of the two \p half2 inputs and
* combines into one \p half2 number.
* 
* \details Extracts high 16 bits from each of the two \p half2 inputs and combines into
* one \p half2 number. High 16 bits from input \p a is stored in low 16 bits of
* the return value, high 16 bits from input \p b is stored in high 16 bits of
* the return value.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* \retval The high 16 bits of \p a and of \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns high 16 bits of \p half2 input.
*
* \details Returns high 16 bits of \p half2 input \p a.
* \param[in] a - half2. Is only being read. 
*
* \returns half
* \retval The high 16 bits of the input. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __high2half(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Returns low 16 bits of \p half2 input.
*
* \details Returns low 16 bits of \p half2 input \p a.
* \param[in] a - half2. Is only being read. 
*
* \returns half
* \retval Returns \p half which contains low 16 bits of the input \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __low2half(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Checks if the input \p half number is infinite.
* 
* \details Checks if the input \p half number \p a is infinite. 
* \param[in] a - half. Is only being read. 
* 
* \returns int 
* \retval -1 iff \p a is equal to negative infinity, 
* \retval 1 iff \p a is equal to positive infinity, 
* \retval 0 otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ int __hisinf(const __half a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Combines two \p half numbers into one \p half2 number.
* 
* \details Combines two input \p half number \p a and \p b into one \p half2 number.
* Input \p a is stored in low 16 bits of the return value, input \p b is stored
* in high 16 bits of the return value.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* 
* \returns half2
* \retval The half2 with one half equal to \p a and the other to \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts low 16 bits from \p half2 input.
* 
* \details Extracts low 16 bits from \p half2 input \p a and returns a new \p half2
* number which has both halves equal to the extracted bits.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* \retval The half2 with both halves equal to the low 16 bits of the input. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Extracts high 16 bits from \p half2 input.
* 
* \details Extracts high 16 bits from \p half2 input \p a and returns a new \p half2
* number which has both halves equal to the extracted bits.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* \retval The half2 with both halves equal to the high 16 bits of the input. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 a);

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a \p half as a signed short integer.
* 
* \details Reinterprets the bits in the half-precision floating-point number \p h
* as a signed short integer. 
* \param[in] h - half. Is only being read. 
* 
* \returns short int
* \retval The reinterpreted value. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ short int __half_as_short(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a \p half as an unsigned short integer.
* 
* \details Reinterprets the bits in the half-precision floating-point \p h
* as an unsigned short number.
* \param[in] h - half. Is only being read. 
* 
* \returns unsigned short int
* \retval The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in a signed short integer as a \p half.
* 
* \details Reinterprets the bits in the signed short integer \p i as a
* half-precision floating-point number.
* \param[in] i - short int. Is only being read. 
* 
* \returns half
* \retval The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __short_as_half(const short int i);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Reinterprets bits in an unsigned short integer as a \p half.
* 
* \details Reinterprets the bits in the unsigned short integer \p i as a
* half-precision floating-point number.
* \param[in] i - unsigned short int. Is only being read. 
* 
* \returns half
* \retval The reinterpreted value.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300)
#if !defined warpSize && !defined __local_warpSize
#define warpSize    32
#define __local_warpSize
#endif

#if defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
#define __WSB_DEPRECATION_MESSAGE(x) __CUDA_FP16_STRINGIFY(x) "() is deprecated in favor of " __CUDA_FP16_STRINGIFY(x) "_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."

__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl)) __half2 __shfl(const __half2 var, const int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_up)) __half2 __shfl_up(const __half2 var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_down))__half2 __shfl_down(const __half2 var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half2 __shfl_xor(const __half2 var, const int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl)) __half __shfl(const __half var, const int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_up)) __half __shfl_up(const __half var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_down)) __half __shfl_down(const __half var, const unsigned int delta, const int width = warpSize);
__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half __shfl_xor(const __half var, const int delta, const int width = warpSize);
#endif

/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Direct copy from indexed thread. 
* 
* \details Returns the value of var held by the thread whose ID is given by delta. 
* If width is less than warpSize then each subsection of the warp behaves as a separate 
* entity with a starting logical thread ID of 0. If delta is outside the range [0:width-1], 
* the value returned corresponds to the value of var held by the delta modulo width (i.e. 
* within the same subsection). width must have a value which is a power of 2; 
* results are undefined if width is not a power of 2, or is a number greater than 
* warpSize. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half2. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by var from the source thread ID as half2. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_sync(const unsigned mask, const __half2 var, const int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller. 
* 
* \details Calculates a source thread ID by subtracting delta from the caller's lane ID. 
* The value of var held by the resulting lane ID is returned: in effect, var is shifted up 
* the warp by delta threads. If width is less than warpSize then each subsection of the warp 
* behaves as a separate entity with a starting logical thread ID of 0. The source thread index 
* will not wrap around the value of width, so effectively the lower delta threads will be unchanged. 
* width must have a value which is a power of 2; results are undefined if width is not a power of 2, 
* or is a number greater than warpSize. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half2. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by var from the source thread ID as half2. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_up_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller. 
* 
* \details Calculates a source thread ID by adding delta to the caller's thread ID. 
* The value of var held by the resulting thread ID is returned: this has the effect 
* of shifting var down the warp by delta threads. If width is less than warpSize then 
* each subsection of the warp behaves as a separate entity with a starting logical 
* thread ID of 0. As for __shfl_up_sync(), the ID number of the source thread 
* will not wrap around the value of width and so the upper delta threads 
* will remain unchanged. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half2. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by var from the source thread ID as half2. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_down_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID. 
* 
* \details Calculates a source thread ID by performing a bitwise XOR of the caller's thread ID with mask: 
* the value of var held by the resulting thread ID is returned. If width is less than warpSize then each 
* group of width consecutive threads are able to access elements from earlier groups of threads, 
* however if they attempt to access elements from later groups of threads their own value of var 
* will be returned. This mode implements a butterfly addressing pattern such as is used in tree 
* reduction and broadcast. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half2. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 4-byte word referenced by var from the source thread ID as half2. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(const unsigned mask, const __half2 var, const int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Direct copy from indexed thread. 
* 
* \details Returns the value of var held by the thread whose ID is given by delta. 
* If width is less than warpSize then each subsection of the warp behaves as a separate 
* entity with a starting logical thread ID of 0. If delta is outside the range [0:width-1], 
* the value returned corresponds to the value of var held by the delta modulo width (i.e. 
* within the same subsection). width must have a value which is a power of 2; 
* results are undefined if width is not a power of 2, or is a number greater than 
* warpSize. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by var from the source thread ID as half. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_sync(const unsigned mask, const __half var, const int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with lower ID relative to the caller. 
* \details Calculates a source thread ID by subtracting delta from the caller's lane ID. 
* The value of var held by the resulting lane ID is returned: in effect, var is shifted up 
* the warp by delta threads. If width is less than warpSize then each subsection of the warp 
* behaves as a separate entity with a starting logical thread ID of 0. The source thread index 
* will not wrap around the value of width, so effectively the lower delta threads will be unchanged. 
* width must have a value which is a power of 2; results are undefined if width is not a power of 2, 
* or is a number greater than warpSize. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by var from the source thread ID as half. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_up_sync(const unsigned mask, const __half var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread with higher ID relative to the caller. 
* 
* \details Calculates a source thread ID by adding delta to the caller's thread ID. 
* The value of var held by the resulting thread ID is returned: this has the effect 
* of shifting var down the warp by delta threads. If width is less than warpSize then 
* each subsection of the warp behaves as a separate entity with a starting logical 
* thread ID of 0. As for __shfl_up_sync(), the ID number of the source thread 
* will not wrap around the value of width and so the upper delta threads 
* will remain unchanged. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by var from the source thread ID as half. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_down_sync(const unsigned mask, const __half var, const unsigned int delta, const int width = warpSize);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Exchange a variable between threads within a warp. Copy from a thread based on bitwise XOR of own thread ID. 
* 
* \details Calculates a source thread ID by performing a bitwise XOR of the caller's thread ID with mask: 
* the value of var held by the resulting thread ID is returned. If width is less than warpSize then each 
* group of width consecutive threads are able to access elements from earlier groups of threads, 
* however if they attempt to access elements from later groups of threads their own value of var 
* will be returned. This mode implements a butterfly addressing pattern such as is used in tree 
* reduction and broadcast. 
* \param[in] mask - unsigned int. Is only being read. 
* \param[in] var - half. Is only being read. 
* \param[in] delta - int. Is only being read. 
* \param[in] width - int. Is only being read. 
* 
* \returns Returns the 2-byte word referenced by var from the source thread ID as half. 
* If the source thread ID is out of range or the source thread has exited, the calling thread's own var is returned. 
* \note_ref_guide_warp_shuffle
* \internal
* \exception-guarantee no-throw guarantee
* \behavior not reentrant, not thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __shfl_xor_sync(const unsigned mask, const __half var, const int delta, const int width = warpSize);

#if defined(__local_warpSize)
#undef warpSize
#undef __local_warpSize
#endif
#endif /*!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 300) */

#if defined(__cplusplus) && ( !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320) )
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.nc` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.nc` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldg(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cg` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cg` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldcg(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.ca` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.ca` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldca(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cs` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cs` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldcs(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.lu` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldlu(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.lu` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldlu(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cv` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half2 __ldcv(const  __half2 *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `ld.global.cv` load instruction.
* \param[in] ptr - memory location
* \returns The value pointed by `ptr`
*/
__CUDA_FP16_DECL__ __half __ldcv(const __half *const ptr);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wb` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwb(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wb` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwb(__half *const ptr, const __half value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cg` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcg(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cg` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcg(__half *const ptr, const __half value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cs` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcs(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.cs` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stcs(__half *const ptr, const __half value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wt` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwt(__half2 *const ptr, const __half2 value);
/**
* \ingroup CUDA_MATH__HALF_MISC
* \brief Generates a `st.global.wt` store instruction.
* \param[out] ptr - memory location
* \param[in] value - the value to be stored
*/
__CUDA_FP16_DECL__ void __stwt(__half *const ptr, const __half value);
#endif /*defined(__cplusplus) && ( !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320) )*/

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs half2 vector if-equal comparison.
* 
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* \retval The vector result of if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison.
* 
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* \retval The vector result of not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The \p half2 result of less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The vector result of greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The half2 vector result of less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison.
* 
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* \retval The vector result of greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison.
* 
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns half2
* \retval The vector result of unordered if-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The vector result of unordered not-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison.
*
* Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The vector result of unordered less-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The \p half2 vector result of unordered greater-equal comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The vector result of unordered less-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The corresponding \p half results are set to 1.0 for true, or 0.0 for false.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The \p half2 vector result of unordered greater-than comparison of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Determine whether \p half2 argument is a NaN.
*
* \details Determine whether each half of input \p half2 number \p a is a NaN.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The half2 with the corresponding \p half results set to
* 1.0 for NaN, 0.0 otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode.
*
* \details Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-95
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The sum of vectors \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-104
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The subtraction of vector \p b from \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
*
* \details Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-102
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The result of elementwise multiplying the vectors \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode.
*
* \details Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
* mode. Prevents floating-point contractions of mul+add into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-95
* \endinternal
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The sum of vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hadd2_rn(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode. Prevents floating-point contractions of mul+sub
* into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-104
* \endinternal
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The subtraction of vector \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hsub2_rn(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode.
*
* \details Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode. Prevents floating-point contractions of
* mul+add or sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-102
* \endinternal
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The result of elementwise multiplying the vectors \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmul2_rn(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector division in round-to-nearest-even mode.
*
* \details Divides \p half2 input vector \p a by input vector \p b in round-to-nearest
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-103
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise division of \p a with \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __h2div(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Calculates the absolute value of both halves of the input \p half2 number and
* returns the result.
*
* \details Calculates the absolute value of both halves of the input \p half2 number and
* returns the result.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval Returns \p a with the absolute value of both halves. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __habs2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p half2 vector add of inputs \p a and \p b, in round-to-nearest
* mode, and clamps the results to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The sum of \p a and \p b, with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector subtraction in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Subtracts \p half2 input vector \p b from input vector \p a in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The subtraction of vector \p b from \p a, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector multiplication in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Performs \p half2 vector multiplication of inputs \p a and \p b, in
* round-to-nearest-even mode, and clamps the results to range [0.0, 1.0]. NaN
* results are flushed to +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns half2
* \retval The result of elementwise multiplication of vectors \p a and \p b, 
* with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode.
*
* \details Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-105
* \endinternal
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* \param[in] c - half2. Is only being read. 
*
* \returns half2
* \retval The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode, with saturation to [0.0, 1.0].
*
* \details Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the
* results to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* \param[in] c - half2. Is only being read. 
*
* \returns half2
* \retval The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c, 
* with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Negates both halves of the input \p half2 number and returns the
* result.
*
* \details Negates both halves of the input \p half2 number \p a and returns the result.
* \internal
* \req DEEPLEARN-SRM_REQ-101
* \endinternal
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval Returns \p a with both halves negated. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Calculates the absolute value of input \p half number and returns the result.
*
* \details Calculates the absolute value of input \p half number and returns the result.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The absolute value of a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __habs(const __half a);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode.
*
* \details Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-94
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* \retval The sum of \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half input \p b from input \p a in round-to-nearest
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-97
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* \retval The result of subtracting \p b from \p a. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode.
*
* \details Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-99
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* \retval The result of multiplying \p a and \p b. 
*/
__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode.
*
* \details Performs \p half addition of inputs \p a and \p b, in round-to-nearest-even
* mode. Prevents floating-point contractions of mul+add into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-94
* \endinternal
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \retval The sum of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hadd_rn(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode.
*
* \details Subtracts \p half input \p b from input \p a in round-to-nearest
* mode. Prevents floating-point contractions of mul+sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-97
* \endinternal
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \retval The result of subtracting \p b from \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hsub_rn(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode.
*
* \details Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
* mode. Prevents floating-point contractions of mul+add or sub into fma.
* \internal
* \req DEEPLEARN-SRM_REQ-99
* \endinternal
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \retval The result of multiplying \p a and \p b.
*/
__CUDA_FP16_DECL__ __half __hmul_rn(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half division in round-to-nearest-even mode.
* 
* \details Divides \p half input \p a by input \p b in round-to-nearest
* mode.
* \internal
* \req DEEPLEARN-SRM_REQ-98
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* 
* \returns half
* \retval The result of dividing \p a by \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__  __half __hdiv(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half addition in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p half add of inputs \p a and \p b, in round-to-nearest-even mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* \retval The sum of \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half subtraction in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Subtracts \p half input \p b from input \p a in round-to-nearest
* mode,
* and clamps the result to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* \retval The result of subtraction of \p b from \p a, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half multiplication in round-to-nearest-even mode, with
* saturation to [0.0, 1.0].
*
* \details Performs \p half multiplication of inputs \p a and \p b, in round-to-nearest
* mode, and clamps the result to range [0.0, 1.0]. NaN results are flushed to
* +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns half
* \retval The result of multiplying \p a and \p b, with respect to saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode.
*
* \details Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* \internal
* \req DEEPLEARN-SRM_REQ-96
* \endinternal
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* \param[in] c - half. Is only being read. 
*
* \returns half
* \retval The result of fused multiply-add operation on \p
* a, \p b, and \p c. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode,
* with saturation to [0.0, 1.0].
*
* \details Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode, and clamps the result
* to range [0.0, 1.0]. NaN results are flushed to +0.0.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
* \param[in] c - half. Is only being read. 
*
* \returns half
* \retval The result of fused multiply-add operation on \p
* a, \p b, and \p c, with respect to saturation. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Negates input \p half number and returns the result.
*
* \details Negates input \p half number and returns the result.
* \internal
* \req DEEPLEARN-SRM_REQ-100
* \endinternal
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval minus a
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hneg(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector if-equal comparison and returns boolean true
* iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of if-equal comparison
* of vectors \p a and \p b are true;
* \retval false otherwise.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector not-equal comparison and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of not-equal comparison
* of vectors \p a and \p b are true, 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-equal comparison and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of less-equal comparison
* of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-equal comparison and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of greater-equal
* comparison of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector less-than comparison and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of less-than comparison
* of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector greater-than comparison and returns boolean
* true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate false results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
* 
* \returns bool 
* \retval true if both \p half results of greater-than
* comparison of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered if-equal comparison and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector if-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half if-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of unordered if-equal
* comparison of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered not-equal comparison and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector not-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half not-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of unordered not-equal
* comparison of vectors \p a and \p b are true;
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-equal comparison and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of unordered less-equal
* comparison of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-equal comparison and
* returns boolean true iff both \p half results are true, boolean false
* otherwise.
*
* \details Performs \p half2 vector greater-equal comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-equal comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of unordered
* greater-equal comparison of vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered less-than comparison and returns
* boolean true iff both \p half results are true, boolean false otherwise.
*
* \details Performs \p half2 vector less-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half less-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of unordered less-than comparison of 
* vectors \p a and \p b are true; 
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Performs \p half2 vector unordered greater-than comparison and
* returns boolean true iff both \p half results are true, boolean false
* otherwise.
*
* \details Performs \p half2 vector greater-than comparison of inputs \p a and \p b.
* The bool result is set to true only if both \p half greater-than comparisons
* evaluate to true, or false otherwise.
* NaN inputs generate true results.
* \param[in] a - half2. Is only being read. 
* \param[in] b - half2. Is only being read. 
*
* \returns bool
* \retval true if both \p half results of unordered
* greater-than comparison of vectors \p a and \p b are true;
* \retval false otherwise. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half if-equal comparison.
*
* \details Performs \p half if-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of if-equal comparison of \p a and \p b. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half not-equal comparison.
*
* \details Performs \p half not-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of not-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half less-equal comparison.
*
* \details Performs \p half less-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of less-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half greater-equal comparison.
*
* \details Performs \p half greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of greater-equal comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half less-than comparison.
*
* \details Performs \p half less-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of less-than comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half greater-than comparison.
*
* \details Performs \p half greater-than comparison of inputs \p a and \p b.
* NaN inputs generate false results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of greater-than comparison of \p a and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered if-equal comparison.
*
* \details Performs \p half if-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of unordered if-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered not-equal comparison.
*
* \details Performs \p half not-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of unordered not-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered less-equal comparison.
*
* \details Performs \p half less-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of unordered less-equal comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered greater-equal comparison.
*
* \details Performs \p half greater-equal comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of unordered greater-equal comparison of \p a
* and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered less-than comparison.
*
* \details Performs \p half less-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of unordered less-than comparison of \p a and
* \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Performs \p half unordered greater-than comparison.
*
* \details Performs \p half greater-than comparison of inputs \p a and \p b.
* NaN inputs generate true results.
* \param[in] a - half. Is only being read. 
* \param[in] b - half. Is only being read. 
*
* \returns bool
* \retval The boolean result of unordered greater-than comparison of \p a
* and \p b.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Determine whether \p half argument is a NaN.
*
* \details Determine whether \p half value \p a is a NaN.
* \param[in] a - half. Is only being read. 
*
* \returns bool
* \retval true iff argument is NaN. 
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ bool __hisnan(const __half a);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half maximum of two input values.
*
* \details Calculates \p half max(\p a, \p b)
* defined as (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hmax(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half minimum of two input values.
*
* \details Calculates \p half min(\p a, \p b)
* defined as (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hmin(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half maximum of two input values, NaNs pass through.
*
* \details Calculates \p half max(\p a, \p b)
* defined as (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hmax_nan(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_COMPARISON
* \brief Calculates \p half minimum of two input values, NaNs pass through.
*
* \details Calculates \p half min(\p a, \p b)
* defined as (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
*
* \returns half
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hmin_nan(const __half a, const __half b);
/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Performs \p half fused multiply-add in round-to-nearest-even mode with relu saturation.
*
* \details Performs \p half multiply on inputs \p a and \p b,
* then performs a \p half add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* Then negative result is clamped to 0.
* NaN result is converted to canonical NaN.
* \param[in] a - half. Is only being read.
* \param[in] b - half. Is only being read.
* \param[in] c - half. Is only being read.
*
* \returns half
* \retval The result of fused multiply-add operation on \p
* a, \p b, and \p c with relu saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half __hfma_relu(const __half a, const __half b, const __half c);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector maximum of two inputs.
*
* \details Calculates \p half2 vector max(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The result of elementwise maximum of vectors \p a  and \p b
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmax2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector minimum of two inputs.
*
* \details Calculates \p half2 vector min(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, the other input is returned.
* - If both inputs are NaNs, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The result of elementwise minimum of vectors \p a  and \p b
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmin2(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector maximum of two inputs, NaNs pass through.
*
* \details Calculates \p half2 vector max(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a > \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The result of elementwise maximum of vectors \p a  and \p b, with NaNs pass through
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmax2_nan(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_COMPARISON
* \brief Calculates \p half2 vector minimum of two inputs, NaNs pass through.
*
* \details Calculates \p half2 vector min(\p a, \p b).
* Elementwise \p half operation is defined as
* (\p a < \p b) ? \p a : \p b.
* - If either of inputs is NaN, then canonical NaN is returned.
* - If values of both inputs are 0.0, then +0.0 > -0.0
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
*
* \returns half2
* \retval The result of elementwise minimum of vectors \p a  and \p b, with NaNs pass through
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hmin2_nan(const __half2 a, const __half2 b);
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs \p half2 vector fused multiply-add in round-to-nearest-even
* mode with relu saturation.
*
* \details Performs \p half2 vector multiply on inputs \p a and \p b,
* then performs a \p half2 vector add of the result with \p c,
* rounding the result once in round-to-nearest-even mode.
* Then negative result is clamped to 0.
* NaN result is converted to canonical NaN.
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
* \param[in] c - half2. Is only being read.
*
* \returns half2
* \retval The result of elementwise fused multiply-add operation on vectors \p a, \p b, and \p c with relu saturation.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hfma2_relu(const __half2 a, const __half2 b, const __half2 c);
#endif /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800) */
/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Performs fast complex multiply-accumulate
*
* \details Interprets vector \p half2 input pairs \p a, \p b, and \p c as
* complex numbers in \p half precision and performs
* complex multiply-accumulate operation: a*b + c
* \param[in] a - half2. Is only being read.
* \param[in] b - half2. Is only being read.
* \param[in] c - half2. Is only being read.
*
* \returns half2
* \retval The result of complex multiply-accumulate operation on complex numbers \p a, \p b, and \p c
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half square root in round-to-nearest-even mode.
*
* \details Calculates \p half square root of input \p a in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The square root of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hsqrt(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half reciprocal square root in round-to-nearest-even
* mode.
*
* \details Calculates \p half reciprocal square root of input \p a in round-to-nearest
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The reciprocal square root of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hrsqrt(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half reciprocal in round-to-nearest-even mode.
*
* \details Calculates \p half reciprocal of input \p a in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The reciprocal of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hrcp(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half natural logarithm in round-to-nearest-even mode.
*
* \details Calculates \p half natural logarithm of input \p a in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The natural logarithm of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hlog(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half binary logarithm in round-to-nearest-even mode.
*
* \details Calculates \p half binary logarithm of input \p a in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The binary logarithm of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hlog2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half decimal logarithm in round-to-nearest-even mode.
*
* \details Calculates \p half decimal logarithm of input \p a in round-to-nearest-even
* mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The decimal logarithm of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hlog10(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half natural exponential function in round-to-nearest
* mode.
*
* \details Calculates \p half natural exponential function of input \p a in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The natural exponential function on \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hexp(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half binary exponential function in round-to-nearest
* mode.
*
* \details Calculates \p half binary exponential function of input \p a in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The binary exponential function on \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hexp2(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half decimal exponential function in round-to-nearest
* mode.
*
* \details Calculates \p half decimal exponential function of input \p a in
* round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The decimal exponential function on \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hexp10(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half cosine in round-to-nearest-even mode.
*
* \details Calculates \p half cosine of input \p a in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The cosine of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hcos(const __half a);
/**
* \ingroup CUDA_MATH__HALF_FUNCTIONS
* \brief Calculates \p half sine in round-to-nearest-even mode.
*
* \details Calculates \p half sine of input \p a in round-to-nearest-even mode.
* \param[in] a - half. Is only being read. 
*
* \returns half
* \retval The sine of \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half hsin(const __half a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector square root in round-to-nearest-even mode.
*
* \details Calculates \p half2 square root of input vector \p a in round-to-nearest
* mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise square root on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector reciprocal square root in round-to-nearest
* mode.
*
* \details Calculates \p half2 reciprocal square root of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise reciprocal square root on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector reciprocal in round-to-nearest-even mode.
*
* \details Calculates \p half2 reciprocal of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise reciprocal on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector natural logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 natural logarithm of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise natural logarithm on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2log(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector binary logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 binary logarithm of input vector \p a in round-to-nearest
* mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise binary logarithm on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector decimal logarithm in round-to-nearest-even
* mode.
*
* \details Calculates \p half2 decimal logarithm of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise decimal logarithm on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector exponential function in round-to-nearest
* mode.
*
* \details Calculates \p half2 exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise exponential function on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector binary exponential function in
* round-to-nearest-even mode.
*
* \details Calculates \p half2 binary exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
*
* \returns half2
* \retval The elementwise binary exponential function on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector decimal exponential function in
* round-to-nearest-even mode.
* 
* \details Calculates \p half2 decimal exponential function of input vector \p a in
* round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* \retval The elementwise decimal exponential function on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector cosine in round-to-nearest-even mode.
* 
* \details Calculates \p half2 cosine of input vector \p a in round-to-nearest-even
* mode.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* \retval The elementwise cosine on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a);
/**
* \ingroup CUDA_MATH__HALF2_FUNCTIONS
* \brief Calculates \p half2 vector sine in round-to-nearest-even mode.
* 
* \details Calculates \p half2 sine of input vector \p a in round-to-nearest-even mode.
* \param[in] a - half2. Is only being read. 
* 
* \returns half2
* \retval The elementwise sine on vector \p a.
* \internal
* \exception-guarantee no-throw guarantee
* \behavior reentrant, thread safe
* \endinternal
*/
__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a);

#endif /*if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)*/

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600)


/**
* \ingroup CUDA_MATH__HALF2_ARITHMETIC
* \brief Vector add \p val to the value stored at \p address in global or shared memory, and writes this
* value back to \p address. The atomicity of the add operation is guaranteed separately for each of the
* two __half elements; the entire __half2 is not guaranteed to be atomic as a single 32-bit access.
* 
* \details The location of \p address must be in global or shared memory. This operation has undefined
* behavior otherwise. This operation is only supported by devices of compute capability 6.x and higher.
* 
* \param[in] address - half2*. An address in global or shared memory.
* \param[in] val - half2. The value to be added.
* 
* \returns half2
* \retval The old value read from \p address.
*
* \note_ref_guide_atomic
*/
__CUDA_FP16_DECL__ __half2 atomicAdd(__half2 *const address, const __half2 val);

#endif /*if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 600)*/

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)

/**
* \ingroup CUDA_MATH__HALF_ARITHMETIC
* \brief Adds \p val to the value stored at \p address in global or shared memory, and writes this value
* back to \p address. This operation is performed in one atomic operation.
* 
* \details The location of \p address must be in global or shared memory. This operation has undefined
* behavior otherwise. This operation is only supported by devices of compute capability 7.x and higher.
* 
* \param[in] address - half*. An address in global or shared memory.
* \param[in] val - half. The value to be added.
* 
* \returns half
* \retval The old value read from \p address.
* 
* \note_ref_guide_atomic
*/
__CUDA_FP16_DECL__ __half atomicAdd(__half *const address, const __half val);

#endif /*if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)*/

#endif /* defined(__CUDACC__) */

#undef __CUDA_FP16_DECL__
#undef __CUDA_HOSTDEVICE_FP16_DECL__

#endif /* defined(__cplusplus) */

/* Note the .hpp file is included even for host-side compilation, to capture the "half" & "half2" definitions */
#include "cuda_fp16.hpp"
#undef ___CUDA_FP16_STRINGIFY_INNERMOST
#undef __CUDA_FP16_STRINGIFY

#endif /* end of include guard: __CUDA_FP16_H__ */
