 /* Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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
#ifndef CURAND_GLOBALS_H
#define CURAND_GLOBALS_H

#define MAX_XOR_N (5)
#define SKIPAHEAD_BLOCKSIZE (4)
#define SKIPAHEAD_MASK ((1<<SKIPAHEAD_BLOCKSIZE)-1)
#define CURAND_2POW32 (4294967296.f)
#define CURAND_2POW32_DOUBLE (4294967296.)
#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32_INV_DOUBLE (2.3283064365386963e-10) 
#define CURAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define CURAND_2PI (6.2831855f)
#define CURAND_2POW53_INV_2PI_DOUBLE (1.1102230246251565e-16 * 6.2831853071795860)
#define CURAND_PI_DOUBLE  (3.1415926535897932)
#define CURAND_2PI_DOUBLE (6.2831853071795860)
#define CURAND_SQRT2 (-1.4142135f)
#define CURAND_SQRT2_DOUBLE (-1.4142135623730951)

#define SOBOL64_ITR_BINARY_DIVIDE 2
#define SOBOL_M2_BINARY_DIVIDE 10
#define MTGP32_M2_BINARY_DIVIDE 32
#define MAX_LAMBDA 400000
#define MIN_GAUSS_LAMBDA 2000

struct normal_args_st {
    float mean;
    float stddev;
};

typedef struct normal_args_st normal_args_t;

struct normal_args_double_st {
    double mean;
    double stddev;
};

typedef struct normal_args_double_st normal_args_double_t;







#endif
