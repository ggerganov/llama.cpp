/*
 * Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
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

#ifndef CURAND_MTGP32_H
#define CURAND_MTGP32_H
/*
 * @file curand_mtgp32.h
 *
 * @brief Mersenne Twister for Graphic Processors (mtgp32), which
 * generates 32-bit unsigned integers and single precision floating
 * point numbers based on IEEE 754 format.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 */
/*
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.  All rights reserved.
 * Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University nor the names of
 *       its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213
#define MTGP32_STATE_SIZE 1024
#define MTGP32_STATE_MASK 1023
#define CURAND_NUM_MTGP32_PARAMS 200
#define MEXP 11213
#define THREAD_NUM MTGPDC_FLOOR_2P
#define LARGE_SIZE (THREAD_NUM * 3)
#define TBL_SIZE 16

/**
 * \addtogroup DEVICE Device API
 *
 * @{
 */

/*
 * \struct MTGP32_PARAMS_FAST_T
 * MTGP32 parameters.
 * Some element is redundant to keep structure simple.
 *
 * \b pos is a pick up position which is selected to have good
 * performance on graphic processors.  3 < \b pos < Q, where Q is a
 * maximum number such that the size of status array - Q is a power of
 * 2.  For example, when \b mexp is 44497, size of 32-bit status array
 * is 696, and Q is 184, then \b pos is between 4 and 183. This means
 * 512 parallel calculations is allowed when \b mexp is 44497.
 *
 * \b poly_sha1 is SHA1 digest of the characteristic polynomial of
 * state transition function. SHA1 is calculated based on printing
 * form of the polynomial. This is important when we use parameters
 * generated by the dynamic creator which
 *
 * \b mask This is a mask to make the dimension of state space have
 * just Mersenne Prime. This is redundant.
 */

struct mtgp32_params_fast;

struct mtgp32_params_fast {
    int mexp;			/*< Mersenne exponent. This is redundant. */
    int pos;			/*< pick up position. */
    int sh1;			/*< shift value 1. 0 < sh1 < 32. */
    int sh2;			/*< shift value 2. 0 < sh2 < 32. */
    unsigned int tbl[16];		/*< a small matrix. */
    unsigned int tmp_tbl[16];	/*< a small matrix for tempering. */
    unsigned int flt_tmp_tbl[16];	/*< a small matrix for tempering and
                 converting to float. */
    unsigned int mask;		/*< This is a mask for state space */
    unsigned char poly_sha1[21]; /*< SHA1 digest */
};

/** \cond UNHIDE_TYPEDEFS */
typedef struct mtgp32_params_fast mtgp32_params_fast_t;
/** \endcond */

/*
 * Generator Parameters.
 */
struct mtgp32_kernel_params;
struct mtgp32_kernel_params {
    unsigned int pos_tbl[CURAND_NUM_MTGP32_PARAMS];
    unsigned int param_tbl[CURAND_NUM_MTGP32_PARAMS][TBL_SIZE];
    unsigned int temper_tbl[CURAND_NUM_MTGP32_PARAMS][TBL_SIZE];
    unsigned int single_temper_tbl[CURAND_NUM_MTGP32_PARAMS][TBL_SIZE];
    unsigned int sh1_tbl[CURAND_NUM_MTGP32_PARAMS];
    unsigned int sh2_tbl[CURAND_NUM_MTGP32_PARAMS];
    unsigned int mask[1];
};

/** \cond UNHIDE_TYPEDEFS */
typedef struct mtgp32_kernel_params mtgp32_kernel_params_t;
/** \endcond */



/*
 * kernel I/O
 * This structure must be initialized before first use.
 */

/* MTGP (Mersenne Twister) RNG */
/* This generator uses the Mersenne Twister algorithm of
 * http://arxiv.org/abs/1005.4973v2
 * Has period 2^11213.
*/

/**
 * CURAND MTGP32 state 
 */
struct curandStateMtgp32;

struct curandStateMtgp32 {
    unsigned int s[MTGP32_STATE_SIZE];
    int offset;
    int pIdx;
    mtgp32_kernel_params_t * k;
};

/*
 * CURAND MTGP32 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateMtgp32 curandStateMtgp32_t;
/** \endcond */

/** @} */

#endif

