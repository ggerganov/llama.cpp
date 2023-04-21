/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__LIBRARY_TYPES_H__)
#define __LIBRARY_TYPES_H__


typedef enum cudaDataType_t
{
	CUDA_R_16F  =  2, /* real as a half */
	CUDA_C_16F  =  6, /* complex as a pair of half numbers */
	CUDA_R_16BF = 14, /* real as a nv_bfloat16 */
	CUDA_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
	CUDA_R_32F  =  0, /* real as a float */
	CUDA_C_32F  =  4, /* complex as a pair of float numbers */
	CUDA_R_64F  =  1, /* real as a double */
	CUDA_C_64F  =  5, /* complex as a pair of double numbers */
	CUDA_R_4I   = 16, /* real as a signed 4-bit int */
	CUDA_C_4I   = 17, /* complex as a pair of signed 4-bit int numbers */
	CUDA_R_4U   = 18, /* real as a unsigned 4-bit int */
	CUDA_C_4U   = 19, /* complex as a pair of unsigned 4-bit int numbers */
	CUDA_R_8I   =  3, /* real as a signed 8-bit int */
	CUDA_C_8I   =  7, /* complex as a pair of signed 8-bit int numbers */
	CUDA_R_8U   =  8, /* real as a unsigned 8-bit int */
	CUDA_C_8U   =  9, /* complex as a pair of unsigned 8-bit int numbers */
	CUDA_R_16I  = 20, /* real as a signed 16-bit int */
	CUDA_C_16I  = 21, /* complex as a pair of signed 16-bit int numbers */
	CUDA_R_16U  = 22, /* real as a unsigned 16-bit int */
	CUDA_C_16U  = 23, /* complex as a pair of unsigned 16-bit int numbers */
	CUDA_R_32I  = 10, /* real as a signed 32-bit int */
	CUDA_C_32I  = 11, /* complex as a pair of signed 32-bit int numbers */
	CUDA_R_32U  = 12, /* real as a unsigned 32-bit int */
	CUDA_C_32U  = 13, /* complex as a pair of unsigned 32-bit int numbers */
	CUDA_R_64I  = 24, /* real as a signed 64-bit int */
	CUDA_C_64I  = 25, /* complex as a pair of signed 64-bit int numbers */
	CUDA_R_64U  = 26, /* real as a unsigned 64-bit int */
	CUDA_C_64U  = 27  /* complex as a pair of unsigned 64-bit int numbers */
} cudaDataType;


typedef enum libraryPropertyType_t
{
	MAJOR_VERSION,
	MINOR_VERSION,
	PATCH_LEVEL
} libraryPropertyType;


#ifndef __cplusplus
typedef enum cudaDataType_t cudaDataType_t;
typedef enum libraryPropertyType_t libraryPropertyType_t;
#endif

#endif /* !__LIBRARY_TYPES_H__ */
