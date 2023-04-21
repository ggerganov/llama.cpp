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

/*
 * curand_mtgp32_host.h
 *
 *
 * MTGP32-11213
 *
 * Mersenne Twister RNG for the GPU
 *
 * The period of generated integers is 2<sup>11213</sup>-1.
 *
 * This code generates 32-bit unsigned integers, and
 * single precision floating point numbers uniformly distributed
 * in the range [1, 2). (float r; 1.0 <= r < 2.0)
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
#if !defined CURAND_MTGP32_HOST_H
#define CURAND_MTGP32_HOST_H

#if !defined(QUALIFIERS)
#define QUALIFIERS static inline __device__
#endif

#include <cuda.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include "curand.h"
#include "curand_mtgp32.h"
#include "curand_mtgp32dc_p_11213.h"


/**
 * \addtogroup DEVICE Device API
 *
 * @{
 */

static const unsigned int non_zero = 0x4d544750;

/*
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static __forceinline__ unsigned int ini_func1(unsigned int x) {
    return (x ^ (x >> 27)) * (1664525);
}

/*
 * This function represents a function used in the initialization
 * by mtgp32_init_by_array() and mtgp32_init_by_str().
 * @param[in] x 32-bit integer
 * @return 32-bit integer
 */
static __forceinline__ unsigned int ini_func2(unsigned int x) {
    return (x ^ (x >> 27)) * (1566083941);
}

/*
 * This function initializes the internal state array with a 32-bit
 * integer seed. The allocated memory should be freed by calling
 * mtgp32_free(). \b para should be one of the elements in the
 * parameter table (mtgp32-param-ref.c).
 *
 * This function is call by cuda program, because cuda program uses
 * another structure and another allocation method.
 *
 * @param[out] array MTGP internal status vector.
 * @param[in] para parameter structure
 * @param[in] seed a 32-bit integer used as the seed.
 */
static __forceinline__ __host__
void mtgp32_init_state(unsigned int state[],
                       const mtgp32_params_fast_t *para, unsigned int seed) {
    int i;
    int size = para->mexp / 32 + 1;
    unsigned int hidden_seed;
    unsigned int tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(state, tmp & 0xff, sizeof(unsigned int) * size);
    state[0] = seed;
    state[1] = hidden_seed;
    for (i = 1; i < size; i++) {
        state[i] ^= (1812433253) * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
    }
}

/*
 * This function initializes the internal state array
 * with a 32-bit integer array. \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a 32-bit integer array used as a seed.
 * @param[in] length length of the array.
 * @return CURAND_STATUS_SUCCESS
 */
static __forceinline__ __host__
int mtgp32_init_by_array(unsigned int state[],
                         const mtgp32_params_fast_t *para,
                         unsigned int *array, int length) {
    int i, j, count;
    unsigned int r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    unsigned int hidden_seed;
    unsigned int tmp;

    if (size >= 623) {
    lag = 11;
    } else if (size >= 68) {
    lag = 7;
    } else if (size >= 39) {
    lag = 5;
    } else {
    lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(state, tmp & 0xff, sizeof(unsigned int) * size);
    state[0] = hidden_seed;

    if (length + 1 > size) {
    count = length + 1;
    } else {
    count = size;
    }
    r = ini_func1(state[0] ^ state[mid] ^ state[size - 1]);
    state[mid] += r;
    r += length;
    state[(mid + lag) % size] += r;
    state[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += array[j] + i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (; j < count; j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
    r = ini_func2(state[i] + state[(i + mid) % size]
              + state[(i + size - 1) % size]);
    state[(i + mid) % size] ^= r;
    r -= i;
    state[(i + mid + lag) % size] ^= r;
    state[i] = r;
    i = (i + 1) % size;
    }
    if (state[size - 1] == 0) {
    state[size - 1] = non_zero;
    }
    return 0;
}

/*
 * This function initializes the internal state array
 * with a character array. \b para should be one of the elements in
 * the parameter table (mtgp32-param-ref.c).
 * This is the same algorithm with mtgp32_init_by_array(), but hope to
 * be more useful.
 *
 * @param[out] mtgp32 MTGP structure.
 * @param[in] para parameter structure
 * @param[in] array a character array used as a seed. (terminated by zero.)
 * @return memory allocation result. if 0 then O.K.
 */
static __forceinline__ __host__
int mtgp32_init_by_str(unsigned int state[],
                       const mtgp32_params_fast_t *para, unsigned char *array) {
    int i, j, count;
    unsigned int r;
    int lag;
    int mid;
    int size = para->mexp / 32 + 1;
    int length = (unsigned int)strlen((char *)array);
    unsigned int hidden_seed;
    unsigned int tmp;

    if (size >= 623) {
    lag = 11;
    } else if (size >= 68) {
    lag = 7;
    } else if (size >= 39) {
    lag = 5;
    } else {
    lag = 3;
    }
    mid = (size - lag) / 2;

    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(state, tmp & 0xff, sizeof(unsigned int) * size);
    state[0] = hidden_seed;

    if (length + 1 > size) {
    count = length + 1;
    } else {
    count = size;
    }
    r = ini_func1(state[0] ^ state[mid] ^ state[size - 1]);
    state[mid] += r;
    r += length;
    state[(mid + lag) % size] += r;
    state[0] = r;
    i = 1;
    count--;
    for (i = 1, j = 0; (j < count) && (j < length); j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += array[j] + i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (; j < count; j++) {
    r = ini_func1(state[i] ^ state[(i + mid) % size]
              ^ state[(i + size - 1) % size]);
    state[(i + mid) % size] += r;
    r += i;
    state[(i + mid + lag) % size] += r;
    state[i] = r;
    i = (i + 1) % size;
    }
    for (j = 0; j < size; j++) {
    r = ini_func2(state[i] + state[(i + mid) % size]
              + state[(i + size - 1) % size]);
    state[(i + mid) % size] ^= r;
    r -= i;
    state[(i + mid + lag) % size] ^= r;
    state[i] = r;
    i = (i + 1) % size;
    }
    if (state[size - 1] == 0) {
    state[size - 1] = non_zero;
    }
    return 0;
}

template<typename ParamsType>
static __forceinline__ __host__
curandStatus_t curandMakeMTGP32ConstantsImpl(const mtgp32_params_fast_t params[], ParamsType * p, const int block_num)
{
    const int size1 = sizeof(unsigned int) * block_num;
    const int size2 = sizeof(unsigned int) * block_num * TBL_SIZE;
    unsigned int *h_pos_tbl;
    unsigned int *h_sh1_tbl;
    unsigned int *h_sh2_tbl;
    unsigned int *h_param_tbl;
    unsigned int *h_temper_tbl;
    unsigned int *h_single_temper_tbl;
    unsigned int *h_mask;
    curandStatus_t status = CURAND_STATUS_SUCCESS;

    h_pos_tbl = (unsigned int *)malloc(size1);
    h_sh1_tbl = (unsigned int *)malloc(size1);
    h_sh2_tbl = (unsigned int *)malloc(size1);
    h_param_tbl = (unsigned int *)malloc(size2);
    h_temper_tbl = (unsigned int *)malloc(size2);
    h_single_temper_tbl = (unsigned int *)malloc(size2);
    h_mask = (unsigned int *)malloc(sizeof(unsigned int));
    if (h_pos_tbl == NULL
	    || h_sh1_tbl == NULL
	    || h_sh2_tbl == NULL
	    || h_param_tbl == NULL
	    || h_temper_tbl == NULL
	    || h_single_temper_tbl == NULL
	    || h_mask == NULL) {
        if (h_pos_tbl != NULL) free(h_pos_tbl);
        if (h_sh1_tbl != NULL) free(h_sh1_tbl);
        if (h_sh2_tbl != NULL) free(h_sh2_tbl);
        if (h_param_tbl != NULL) free(h_param_tbl);
        if (h_temper_tbl != NULL) free(h_temper_tbl);
        if (h_single_temper_tbl != NULL) free(h_single_temper_tbl);
        if (h_mask != NULL) free(h_mask);
        status = CURAND_STATUS_ALLOCATION_FAILED;
    } else {

        h_mask[0] = params[0].mask;
        for (int i = 0; i < block_num; i++) {
	        h_pos_tbl[i] = params[i].pos;
	        h_sh1_tbl[i] = params[i].sh1;
	        h_sh2_tbl[i] = params[i].sh2;
	        for (int j = 0; j < TBL_SIZE; j++) {
	            h_param_tbl[i * TBL_SIZE + j] = params[i].tbl[j];
	            h_temper_tbl[i * TBL_SIZE + j] = params[i].tmp_tbl[j];
	            h_single_temper_tbl[i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
	        }
        }
        if (cudaMemcpy( p->pos_tbl,
                        h_pos_tbl, size1, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->sh1_tbl,
                        h_sh1_tbl, size1, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->sh2_tbl,
                        h_sh2_tbl, size1, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->param_tbl,
                        h_param_tbl, size2, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->temper_tbl,
                        h_temper_tbl, size2, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->single_temper_tbl,
                        h_single_temper_tbl, size2, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        } else
        if (cudaMemcpy( p->mask,
                        h_mask, sizeof(unsigned int), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        }
    }
    if (h_pos_tbl != NULL) free(h_pos_tbl);
    if (h_sh1_tbl != NULL) free(h_sh1_tbl);
    if (h_sh2_tbl != NULL) free(h_sh2_tbl);
    if (h_param_tbl != NULL) free(h_param_tbl);
    if (h_temper_tbl != NULL) free(h_temper_tbl);
    if (h_single_temper_tbl != NULL)free(h_single_temper_tbl);
    if (h_mask != NULL) free(h_mask);
    return status;
}

/**
 * \brief Set up constant parameters for the mtgp32 generator
 *
 * This host-side helper function re-organizes CURAND_NUM_MTGP32_PARAMS sets of
 * generator parameters for use by kernel functions and copies the
 * result to the specified location in device memory.
 *
 * \param params - Pointer to an array of type mtgp32_params_fast_t in host memory
 * \param p - pointer to a structure of type mtgp32_kernel_params_t in device memory.
 *
 * \return
 * - CURAND_STATUS_ALLOCATION_FAILED if host memory could not be allocated
 * - CURAND_STATUS_INITIALIZATION_FAILED if the copy to device memory failed
 * - CURAND_STATUS_SUCCESS otherwise
 */
static __forceinline__ __host__
curandStatus_t curandMakeMTGP32Constants(const mtgp32_params_fast_t params[], mtgp32_kernel_params_t * p)
{
    return curandMakeMTGP32ConstantsImpl(params, p, CURAND_NUM_MTGP32_PARAMS);
}

/**
 * \brief Set up initial states for the mtgp32 generator
 *
 * This host-side helper function initializes a number of states (one parameter set per state) for
 * an mtgp32 generator. To accomplish this it allocates a state array in host memory,
 * initializes that array, and copies the result to device memory.
 *
 * \param s - pointer to an array of states in device memory
 * \param params - Pointer to an array of type mtgp32_params_fast_t in host memory
 * \param k - pointer to a structure of type mtgp32_kernel_params_t in device memory
 * \param n - number of parameter sets/states to initialize
 * \param seed - seed value
 *
 * \return
 * - CURAND_STATUS_ALLOCATION_FAILED if host memory state could not be allocated
 * - CURAND_STATUS_INITIALIZATION_FAILED if the copy to device memory failed
 * - CURAND_STATUS_SUCCESS otherwise
 */
static __forceinline__ __host__
curandStatus_t CURANDAPI curandMakeMTGP32KernelState(curandStateMtgp32_t *s,
                                                     mtgp32_params_fast_t params[],
                                                     mtgp32_kernel_params_t *k,
                                                     int n,
                                                     unsigned long long seed)
{
    int i;
    curandStatus_t status = CURAND_STATUS_SUCCESS;
    curandStateMtgp32_t *h_status =(curandStateMtgp32_t *) malloc(sizeof(curandStateMtgp32_t) * n);
    if (h_status == NULL) {
        status = CURAND_STATUS_ALLOCATION_FAILED;
    } else {
        seed = seed ^ (seed >> 32);
        for (i = 0; i < n; i++) {
            mtgp32_init_state(&(h_status[i].s[0]), &params[i],(unsigned int)seed + i + 1);
            h_status[i].offset = 0;
            h_status[i].pIdx = i;
            h_status[i].k = k;
        }
        if (cudaMemcpy(s, h_status,
                       sizeof(curandStateMtgp32_t) * n,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            status = CURAND_STATUS_INITIALIZATION_FAILED;
        }
     }
    free(h_status);
    return status;
}

/** @} */

#endif


