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

#if !defined(__VECTOR_FUNCTIONS_H__)
#define __VECTOR_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

#if defined(__CUDACC_RTC__)
#define __VECTOR_FUNCTIONS_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#endif /* __CUDACC_RTC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__VECTOR_FUNCTIONS_DECL__ char1 make_char1(signed char x);

__VECTOR_FUNCTIONS_DECL__ uchar1 make_uchar1(unsigned char x);

__VECTOR_FUNCTIONS_DECL__ char2 make_char2(signed char x, signed char y);

__VECTOR_FUNCTIONS_DECL__ uchar2 make_uchar2(unsigned char x, unsigned char y);

__VECTOR_FUNCTIONS_DECL__ char3 make_char3(signed char x, signed char y, signed char z);

__VECTOR_FUNCTIONS_DECL__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z);

__VECTOR_FUNCTIONS_DECL__ char4 make_char4(signed char x, signed char y, signed char z, signed char w);

__VECTOR_FUNCTIONS_DECL__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w);

__VECTOR_FUNCTIONS_DECL__ short1 make_short1(short x);

__VECTOR_FUNCTIONS_DECL__ ushort1 make_ushort1(unsigned short x);

__VECTOR_FUNCTIONS_DECL__ short2 make_short2(short x, short y);

__VECTOR_FUNCTIONS_DECL__ ushort2 make_ushort2(unsigned short x, unsigned short y);

__VECTOR_FUNCTIONS_DECL__ short3 make_short3(short x,short y, short z);

__VECTOR_FUNCTIONS_DECL__ ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z);

__VECTOR_FUNCTIONS_DECL__ short4 make_short4(short x, short y, short z, short w);

__VECTOR_FUNCTIONS_DECL__ ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w);

__VECTOR_FUNCTIONS_DECL__ int1 make_int1(int x);

__VECTOR_FUNCTIONS_DECL__ uint1 make_uint1(unsigned int x);

__VECTOR_FUNCTIONS_DECL__ int2 make_int2(int x, int y);

__VECTOR_FUNCTIONS_DECL__ uint2 make_uint2(unsigned int x, unsigned int y);

__VECTOR_FUNCTIONS_DECL__ int3 make_int3(int x, int y, int z);

__VECTOR_FUNCTIONS_DECL__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z);

__VECTOR_FUNCTIONS_DECL__ int4 make_int4(int x, int y, int z, int w);

__VECTOR_FUNCTIONS_DECL__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);

__VECTOR_FUNCTIONS_DECL__ long1 make_long1(long int x);

__VECTOR_FUNCTIONS_DECL__ ulong1 make_ulong1(unsigned long int x);

__VECTOR_FUNCTIONS_DECL__ long2 make_long2(long int x, long int y);

__VECTOR_FUNCTIONS_DECL__ ulong2 make_ulong2(unsigned long int x, unsigned long int y);

__VECTOR_FUNCTIONS_DECL__ long3 make_long3(long int x, long int y, long int z);

__VECTOR_FUNCTIONS_DECL__ ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z);

__VECTOR_FUNCTIONS_DECL__ long4 make_long4(long int x, long int y, long int z, long int w);

__VECTOR_FUNCTIONS_DECL__ ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w);

__VECTOR_FUNCTIONS_DECL__ float1 make_float1(float x);

__VECTOR_FUNCTIONS_DECL__ float2 make_float2(float x, float y);

__VECTOR_FUNCTIONS_DECL__ float3 make_float3(float x, float y, float z);

__VECTOR_FUNCTIONS_DECL__ float4 make_float4(float x, float y, float z, float w);

__VECTOR_FUNCTIONS_DECL__ longlong1 make_longlong1(long long int x);

__VECTOR_FUNCTIONS_DECL__ ulonglong1 make_ulonglong1(unsigned long long int x);

__VECTOR_FUNCTIONS_DECL__ longlong2 make_longlong2(long long int x, long long int y);

__VECTOR_FUNCTIONS_DECL__ ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y);

__VECTOR_FUNCTIONS_DECL__ longlong3 make_longlong3(long long int x, long long int y, long long int z);

__VECTOR_FUNCTIONS_DECL__ ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z);

__VECTOR_FUNCTIONS_DECL__ longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w);

__VECTOR_FUNCTIONS_DECL__ ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w);

__VECTOR_FUNCTIONS_DECL__ double1 make_double1(double x);

__VECTOR_FUNCTIONS_DECL__ double2 make_double2(double x, double y);

__VECTOR_FUNCTIONS_DECL__ double3 make_double3(double x, double y, double z);

__VECTOR_FUNCTIONS_DECL__ double4 make_double4(double x, double y, double z, double w);

#undef __VECTOR_FUNCTIONS_DECL__

#if !defined(__CUDACC_RTC__)
#include "vector_functions.hpp"
#endif /* !__CUDACC_RTC__ */

#endif /* !__VECTOR_FUNCTIONS_H__ */
