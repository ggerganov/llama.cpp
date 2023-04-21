
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


#if !defined(CURAND_KERNEL_H_)
#define CURAND_KERNEL_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#if !defined(QUALIFIERS)
#define QUALIFIERS static __forceinline__ __device__
#endif


#ifdef __CUDACC_RTC__
#define CURAND_DETAIL_USE_CUDA_STL
#endif

#if __cplusplus >= 201103L
#   ifdef CURAND_DETAIL_USE_CUDA_STL
#       define CURAND_STD cuda::std
#       include <cuda/std/type_traits>
#   else
#       define CURAND_STD std
#       include <type_traits>
#   endif // CURAND_DETAIL_USE_CUDA_STL
#else
// To support C++03 compilation
#   define CURAND_STD curand_detail
namespace curand_detail {
    template<bool B, class T = void>
    struct enable_if {};

    template<class T>
    struct enable_if<true, T> { typedef T type; };

    template<class T, class U>
    struct is_same { static const bool value = false; };

    template<class T>
    struct is_same<T, T> { static const bool value = true; };
} // namespace curand_detail
#endif // __cplusplus >= 201103L

#ifndef __CUDACC_RTC__
#include <math.h>
#endif // __CUDACC_RTC__

#include "curand.h"
#include "curand_discrete.h"
#include "curand_precalc.h"
#include "curand_mrg32k3a.h"
#include "curand_mtgp32_kernel.h"
#include "curand_philox4x32_x.h"
#include "curand_globals.h"

/* Test RNG */
/* This generator uses the formula:
   x_n = x_(n-1) + 1 mod 2^32
   x_0 = (unsigned int)seed * 3
   Subsequences are spaced 31337 steps apart.
*/
struct curandStateTest {
    unsigned int v;
};

/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateTest curandStateTest_t;
/** \endcond */

/* XORSHIFT FAMILY RNGs */
/* These generators are a family proposed by Marsaglia.  They keep state
   in 32 bit chunks, then use repeated shift and xor operations to scramble
   the bits.  The following generators are a combination of a simple Weyl
   generator with an N variable XORSHIFT generator.
*/

/* XORSHIFT RNG */
/* This generator uses the xorwow formula of
www.jstatsoft.org/v08/i14/paper page 5
Has period 2^192 - 2^32.
*/
/**
 * CURAND XORWOW state
 */
struct curandStateXORWOW;

/*
 * Implementation details not in reference documentation */
struct curandStateXORWOW {
    unsigned int d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

/*
 * CURAND XORWOW state
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateXORWOW curandStateXORWOW_t;

#define EXTRA_FLAG_NORMAL         0x00000001
#define EXTRA_FLAG_LOG_NORMAL     0x00000002
/** \endcond */

/* Combined Multiple Recursive Generators */
/* These generators are a family proposed by L'Ecuyer.  They keep state
   in sets of doubles, then use repeated modular arithmetic multiply operations
   to scramble the bits in each set, and combine the result.
*/

/* MRG32k3a RNG */
/* This generator uses the MRG32k3A formula of
http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c++/streams4.pdf
Has period 2^191.
*/

/* moduli for the recursions */
/** \cond UNHIDE_DEFINES */
#define MRG32K3A_MOD1 4294967087.
#define MRG32K3A_MOD2 4294944443.

/* Constants used in generation */

#define MRG32K3A_A12  1403580.
#define MRG32K3A_A13N 810728.
#define MRG32K3A_A21  527612.
#define MRG32K3A_A23N 1370589.
#define MRG32K3A_NORM (2.3283065498378288e-10)
//
// #define MRG32K3A_BITS_NORM ((double)((POW32_DOUBLE-1.0)/MOD1))
//  above constant, used verbatim, rounds differently on some host systems.
#define MRG32K3A_BITS_NORM 1.000000048662

/** \endcond */




/**
 * CURAND MRG32K3A state
 */
struct curandStateMRG32k3a;

/* Implementation details not in reference documentation */
struct curandStateMRG32k3a {
    unsigned int s1[3];
    unsigned int s2[3];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

/*
 * CURAND MRG32K3A state
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;
/** \endcond */

/* SOBOL QRNG */
/**
 * CURAND Sobol32 state
 */
struct curandStateSobol32;

/* Implementation details not in reference documentation */
struct curandStateSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};

/*
 * CURAND Sobol32 state
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateSobol32 curandStateSobol32_t;
/** \endcond */

/**
 * CURAND Scrambled Sobol32 state
 */
struct curandStateScrambledSobol32;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};

/*
 * CURAND Scrambled Sobol32 state
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;
/** \endcond */

/**
 * CURAND Sobol64 state
 */
struct curandStateSobol64;

/* Implementation details not in reference documentation */
struct curandStateSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};

/*
 * CURAND Sobol64 state
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateSobol64 curandStateSobol64_t;
/** \endcond */

/**
 * CURAND Scrambled Sobol64 state
 */
struct curandStateScrambledSobol64;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};

/*
 * CURAND Scrambled Sobol64 state
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateScrambledSobol64 curandStateScrambledSobol64_t;
/** \endcond */

/*
 * Default RNG
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateXORWOW curandState_t;
typedef struct curandStateXORWOW curandState;
/** \endcond */

/****************************************************************************/
/* Utility functions needed by RNGs */
/****************************************************************************/
/** \cond UNHIDE_UTILITIES */
/*
   multiply vector by matrix, store in result
   matrix is n x n, measured in 32 bit units
   matrix is stored in row major order
   vector and result cannot be same pointer
*/
template<int N>
QUALIFIERS void __curand_matvec_inplace(unsigned int *vector, unsigned int *matrix)
{
    unsigned int result[N] = { 0 };
    for(int i = 0; i < N; i++) {
        #ifdef __CUDA_ARCH__
        #pragma unroll 16
        #endif
        for(int j = 0; j < 32; j++) {
            if(vector[i] & (1 << j)) {
                for(int k = 0; k < N; k++) {
                    result[k] ^= matrix[N * (i * 32 + j) + k];
                }
            }
        }
    }
    for(int i = 0; i < N; i++) {
        vector[i] = result[i];
    }
}

QUALIFIERS void __curand_matvec(unsigned int *vector, unsigned int *matrix,
                                unsigned int *result, int n)
{
    for(int i = 0; i < n; i++) {
        result[i] = 0;
    }
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < 32; j++) {
            if(vector[i] & (1 << j)) {
                for(int k = 0; k < n; k++) {
                    result[k] ^= matrix[n * (i * 32 + j) + k];
                }
            }
        }
    }
}

/* generate identity matrix */
QUALIFIERS void __curand_matidentity(unsigned int *matrix, int n)
{
    int r;
    for(int i = 0; i < n * 32; i++) {
        for(int j = 0; j < n; j++) {
            r = i & 31;
            if(i / 32 == j) {
                matrix[i * n + j] = (1 << r);
            } else {
                matrix[i * n + j] = 0;
            }
        }
    }
}

/* multiply matrixA by matrixB, store back in matrixA
   matrixA and matrixB must not be same matrix */
QUALIFIERS void __curand_matmat(unsigned int *matrixA, unsigned int *matrixB, int n)
{
    unsigned int result[MAX_XOR_N];
    for(int i = 0; i < n * 32; i++) {
        __curand_matvec(matrixA + i * n, matrixB, result, n);
        for(int j = 0; j < n; j++) {
            matrixA[i * n + j] = result[j];
        }
    }
}

/* copy vectorA to vector */
QUALIFIERS void __curand_veccopy(unsigned int *vector, unsigned int *vectorA, int n)
{
    for(int i = 0; i < n; i++) {
        vector[i] = vectorA[i];
    }
}

/* copy matrixA to matrix */
QUALIFIERS void __curand_matcopy(unsigned int *matrix, unsigned int *matrixA, int n)
{
    for(int i = 0; i < n * n * 32; i++) {
        matrix[i] = matrixA[i];
    }
}

/* compute matrixA to power p, store result in matrix */
QUALIFIERS void __curand_matpow(unsigned int *matrix, unsigned int *matrixA,
                                unsigned long long p, int n)
{
    unsigned int matrixR[MAX_XOR_N * MAX_XOR_N * 32];
    unsigned int matrixS[MAX_XOR_N * MAX_XOR_N * 32];
    __curand_matidentity(matrix, n);
    __curand_matcopy(matrixR, matrixA, n);
    while(p) {
        if(p & 1) {
            __curand_matmat(matrix, matrixR, n);
        }
        __curand_matcopy(matrixS, matrixR, n);
        __curand_matmat(matrixR, matrixS, n);
        p >>= 1;
    }
}

/****************************************************************************/
/* Utility functions needed by MRG32k3a RNG                                 */
/* Matrix operations modulo some integer less than 2**32, done in           */
/* double precision floating point, with care not to overflow 53 bits       */
/****************************************************************************/

/* return i mod m.                                                          */
/* assumes i and m are integers represented accurately in doubles           */

QUALIFIERS double curand_MRGmod(double i, double m)
{
    double quo;
    double rem;
    quo = floor(i/m);
    rem = i - (quo*m);
    if (rem < 0.0) rem += m;
    return rem;
}

/* Multiplication modulo m. Inputs i and j less than 2**32                  */
/* Ensure intermediate results do not exceed 2**53                          */

QUALIFIERS double curand_MRGmodMul(double i, double j, double m)
{
    double tempHi;
    double tempLo;

    tempHi = floor(i/131072.0);
    tempLo = i - (tempHi*131072.0);
    tempLo = curand_MRGmod( curand_MRGmod( (tempHi * j), m) * 131072.0 + curand_MRGmod(tempLo * j, m),m);

    if (tempLo < 0.0) tempLo += m;
    return tempLo;
}

/* multiply 3 by 3 matrices of doubles, modulo m                            */

QUALIFIERS void curand_MRGmatMul3x3(unsigned int i1[][3],unsigned int i2[][3],unsigned int o[][3],double m)
{
    int i,j;
    double temp[3][3];
    for (i=0; i<3; i++){
        for (j=0; j<3; j++){
            temp[i][j] = ( curand_MRGmodMul(i1[i][0], i2[0][j], m) +
                           curand_MRGmodMul(i1[i][1], i2[1][j], m) +
                           curand_MRGmodMul(i1[i][2], i2[2][j], m));
            temp[i][j] = curand_MRGmod( temp[i][j], m );
        }
    }
    for (i=0; i<3; i++){
        for (j=0; j<3; j++){
            o[i][j] = (unsigned int)temp[i][j];
        }
    }
}

/* multiply 3 by 3 matrix times 3 by 1 vector of doubles, modulo m          */

QUALIFIERS void curand_MRGmatVecMul3x3( unsigned int i[][3], unsigned int v[], double m)
{
    int k;
    double t[3];
    for (k = 0; k < 3; k++) {
        t[k] = ( curand_MRGmodMul(i[k][0], v[0], m) +
                 curand_MRGmodMul(i[k][1], v[1], m) +
                 curand_MRGmodMul(i[k][2], v[2], m) );
        t[k] = curand_MRGmod( t[k], m );
    }
    for (k = 0; k < 3; k++) {
        v[k] = (unsigned int)t[k];
    }

}

/* raise a 3 by 3 matrix of doubles to a 64 bit integer power pow, modulo m */
/* input is index zero of an array of 3 by 3 matrices m,                    */
/* each m = m[0]**(2**index)                                                */

QUALIFIERS void curand_MRGmatPow3x3( unsigned int in[][3][3], unsigned int o[][3], double m, unsigned long long pow )
{
    int i,j;
    for ( i = 0; i < 3; i++ ) {
        for ( j = 0; j < 3; j++ ) {
            o[i][j] = 0;
            if ( i == j ) o[i][j] = 1;
        }
    }
    i = 0;
    curand_MRGmatVecMul3x3(o,o[0],m);
    while (pow) {
        if ( pow & 1ll ) {
             curand_MRGmatMul3x3(in[i], o, o, m);
        }
        i++;
        pow >>= 1;
    }
}

/* raise a 3 by 3 matrix of doubles to the power                            */
/* 2 to the power (pow modulo 191), modulo m                                */

QUALIFIERS void curnand_MRGmatPow2Pow3x3( double in[][3], double o[][3], double m, unsigned long pow )
{
    unsigned int temp[3][3];
    int i,j;
    pow = pow % 191;
    for ( i = 0; i < 3; i++ ) {
        for ( j = 0; j < 3; j++ ) {
            temp[i][j] = (unsigned int)in[i][j];
        }
    }
    while (pow) {
        curand_MRGmatMul3x3(temp, temp, temp, m);
        pow--;
    }
    for ( i = 0; i < 3; i++ ) {
        for ( j = 0; j < 3; j++ ) {
            o[i][j] = temp[i][j];
        }
    }
}

/** \endcond */

/****************************************************************************/
/* Kernel implementations of RNGs                                           */
/****************************************************************************/

/* Test RNG */

QUALIFIERS void curand_init(unsigned long long seed,
                                            unsigned long long subsequence,
                                            unsigned long long offset,
                                            curandStateTest_t *state)
{
    state->v = (unsigned int)(seed * 3) + (unsigned int)(subsequence * 31337) + \
                     (unsigned int)offset;
}


QUALIFIERS unsigned int curand(curandStateTest_t *state)
{
    unsigned int r = state->v++;
    return r;
}

QUALIFIERS void skipahead(unsigned long long n, curandStateTest_t *state)
{
    state->v += (unsigned int)n;
}

/* XORWOW RNG */

template <typename T, int n>
QUALIFIERS void __curand_generate_skipahead_matrix_xor(unsigned int matrix[])
{
    T state;
    // Generate matrix that advances one step
    // matrix has n * n * 32 32-bit elements
    // solve for matrix by stepping single bit states
    for(int i = 0; i < 32 * n; i++) {
        state.d = 0;
        for(int j = 0; j < n; j++) {
            state.v[j] = 0;
        }
        state.v[i / 32] = (1 << (i & 31));
        curand(&state);
        for(int j = 0; j < n; j++) {
            matrix[i * n + j] = state.v[j];
        }
    }
}

template <typename T, int n>
QUALIFIERS void _skipahead_scratch(unsigned long long x, T *state, unsigned int *scratch)
{
    // unsigned int matrix[n * n * 32];
    unsigned int *matrix = scratch;
    // unsigned int matrixA[n * n * 32];
    unsigned int *matrixA = scratch + (n * n * 32);
    // unsigned int vector[n];
    unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
    // unsigned int result[n];
    unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
    unsigned long long p = x;
    for(int i = 0; i < n; i++) {
        vector[i] = state->v[i];
    }
    int matrix_num = 0;
    while(p && (matrix_num < PRECALC_NUM_MATRICES - 1)) {
        for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
#ifdef __CUDA_ARCH__
            __curand_matvec(vector, precalc_xorwow_offset_matrix[matrix_num], result, n);
#else
            __curand_matvec(vector, precalc_xorwow_offset_matrix_host[matrix_num], result, n);
#endif
            __curand_veccopy(vector, result, n);
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    if(p) {
#ifdef __CUDA_ARCH__
        __curand_matcopy(matrix, precalc_xorwow_offset_matrix[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_offset_matrix[PRECALC_NUM_MATRICES - 1], n);
#else
        __curand_matcopy(matrix, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
#endif
    }
    while(p) {
        for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
            __curand_matvec(vector, matrixA, result, n);
            __curand_veccopy(vector, result, n);
        }
        p >>= SKIPAHEAD_BLOCKSIZE;
        if(p) {
            for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
                __curand_matmat(matrix, matrixA, n);
                __curand_matcopy(matrixA, matrix, n);
            }
        }
    }
    for(int i = 0; i < n; i++) {
        state->v[i] = vector[i];
    }
    state->d += 362437 * (unsigned int)x;
}

template <typename T, int n>
QUALIFIERS void _skipahead_sequence_scratch(unsigned long long x, T *state, unsigned int *scratch)
{
    // unsigned int matrix[n * n * 32];
    unsigned int *matrix = scratch;
    // unsigned int matrixA[n * n * 32];
    unsigned int *matrixA = scratch + (n * n * 32);
    // unsigned int vector[n];
    unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
    // unsigned int result[n];
    unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
    unsigned long long p = x;
    for(int i = 0; i < n; i++) {
        vector[i] = state->v[i];
    }
    int matrix_num = 0;
    while(p && matrix_num < PRECALC_NUM_MATRICES - 1) {
        for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
#ifdef __CUDA_ARCH__
            __curand_matvec(vector, precalc_xorwow_matrix[matrix_num], result, n);
#else
            __curand_matvec(vector, precalc_xorwow_matrix_host[matrix_num], result, n);
#endif
            __curand_veccopy(vector, result, n);
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    if(p) {
#ifdef __CUDA_ARCH__
        __curand_matcopy(matrix, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_matrix[PRECALC_NUM_MATRICES - 1], n);
#else
        __curand_matcopy(matrix, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
        __curand_matcopy(matrixA, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
#endif
    }
    while(p) {
        for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
            __curand_matvec(vector, matrixA, result, n);
            __curand_veccopy(vector, result, n);
        }
        p >>= SKIPAHEAD_BLOCKSIZE;
        if(p) {
            for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
                __curand_matmat(matrix, matrixA, n);
                __curand_matcopy(matrixA, matrix, n);
            }
        }
    }
    for(int i = 0; i < n; i++) {
        state->v[i] = vector[i];
    }
    /* No update of state->d needed, guaranteed to be a multiple of 2^32 */
}

template <typename T, int N>
QUALIFIERS void _skipahead_inplace(const unsigned long long x, T *state)
{
    unsigned long long p = x;
    int matrix_num = 0;
    while(p) {
        for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
#ifdef __CUDA_ARCH__
            __curand_matvec_inplace<N>(state->v, precalc_xorwow_offset_matrix[matrix_num]);
#else
            __curand_matvec_inplace<N>(state->v, precalc_xorwow_offset_matrix_host[matrix_num]);
#endif
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    state->d += 362437 * (unsigned int)x;
}

template <typename T, int N>
QUALIFIERS void _skipahead_sequence_inplace(unsigned long long x, T *state)
{
    int matrix_num = 0;
    while(x) {
        for(unsigned int t = 0; t < (x & PRECALC_BLOCK_MASK); t++) {
#ifdef __CUDA_ARCH__
            __curand_matvec_inplace<N>(state->v, precalc_xorwow_matrix[matrix_num]);
#else
            __curand_matvec_inplace<N>(state->v, precalc_xorwow_matrix_host[matrix_num]);
#endif
        }
        x >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    /* No update of state->d needed, guaranteed to be a multiple of 2^32 */
}

/**
 * \brief Update XORWOW state to skip \p n elements.
 *
 * Update the XORWOW state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead(unsigned long long n, curandStateXORWOW_t *state)
{
    _skipahead_inplace<curandStateXORWOW_t, 5>(n, state);
}

/**
 * \brief Update XORWOW state to skip ahead \p n subsequences.
 *
 * Update the XORWOW state in \p state to skip ahead \p n subsequences.  Each
 * subsequence is \xmlonly<ph outputclass="xmlonly">2<sup>67</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
 * \xmlonly<ph outputclass="xmlonly">2<sup>67</sup></ph>\endxmlonly  * n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of subsequences to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead_sequence(unsigned long long n, curandStateXORWOW_t *state)
{
    _skipahead_sequence_inplace<curandStateXORWOW_t, 5>(n, state);
}

QUALIFIERS void _curand_init_scratch(unsigned long long seed,
                                     unsigned long long subsequence,
                                     unsigned long long offset,
                                     curandStateXORWOW_t *state,
                                     unsigned int *scratch)
{
    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
    unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    unsigned int t0 = 1099087573UL * s0;
    unsigned int t1 = 2591861531UL * s1;
    state->d = 6615241 + t1 + t0;
    state->v[0] = 123456789UL + t0;
    state->v[1] = 362436069UL ^ t0;
    state->v[2] = 521288629UL + t1;
    state->v[3] = 88675123UL ^ t1;
    state->v[4] = 5783321UL + t0;
    _skipahead_sequence_scratch<curandStateXORWOW_t, 5>(subsequence, state, scratch);
    _skipahead_scratch<curandStateXORWOW_t, 5>(offset, state, scratch);
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.f;
    state->boxmuller_extra_double = 0.;
}

QUALIFIERS void _curand_init_inplace(unsigned long long seed,
                                     unsigned long long subsequence,
                                     unsigned long long offset,
                                     curandStateXORWOW_t *state)
{
    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
    unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    unsigned int t0 = 1099087573UL * s0;
    unsigned int t1 = 2591861531UL * s1;
    state->d = 6615241 + t1 + t0;
    state->v[0] = 123456789UL + t0;
    state->v[1] = 362436069UL ^ t0;
    state->v[2] = 521288629UL + t1;
    state->v[3] = 88675123UL ^ t1;
    state->v[4] = 5783321UL + t0;
    _skipahead_sequence_inplace<curandStateXORWOW_t, 5>(subsequence, state);
    _skipahead_inplace<curandStateXORWOW_t, 5>(offset, state);
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.f;
    state->boxmuller_extra_double = 0.;
}

/**
 * \brief Initialize XORWOW state.
 *
 * Initialize XORWOW state in \p state with the given \p seed, \p subsequence,
 * and \p offset.
 *
 * All input values of \p seed, \p subsequence, and \p offset are legal.  Large
 * values for \p subsequence and \p offset require more computation and so will
 * take more time to complete.
 *
 * A value of 0 for \p seed sets the state to the values of the original
 * published version of the \p xorwow algorithm.
 *
 * \param seed - Arbitrary bits to use as a seed
 * \param subsequence - Subsequence to start at
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            curandStateXORWOW_t *state)
{
    _curand_init_inplace(seed, subsequence, offset, state);
}

/**
 * \brief Return 32-bits of pseudorandomness from an XORWOW generator.
 *
 * Return 32-bits of pseudorandomness from the XORWOW generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */
QUALIFIERS unsigned int curand(curandStateXORWOW_t *state)
{
    unsigned int t;
    t = (state->v[0] ^ (state->v[0] >> 2));
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] <<4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}


/**
 * \brief Return 32-bits of pseudorandomness from an Philox4_32_10 generator.
 *
 * Return 32-bits of pseudorandomness from the Philox4_32_10 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */

QUALIFIERS unsigned int curand(curandStatePhilox4_32_10_t *state)
{
    // Maintain the invariant: output[STATE] is always "good" and
    //  is the next value to be returned by curand.
    unsigned int ret;
    switch(state->STATE++){
    default:
        ret = state->output.x;
        break;
    case 1:
        ret = state->output.y;
        break;
    case 2:
        ret = state->output.z;
        break;
    case 3:
        ret = state->output.w;
        break;
    }
    if(state->STATE == 4){
        Philox_State_Incr(state);
        state->output = curand_Philox4x32_10(state->ctr,state->key);
        state->STATE = 0;
    }
    return ret;
}

/**
 * \brief Return tuple of 4 32-bit pseudorandoms from a Philox4_32_10 generator.
 *
 * Return 128 bits of pseudorandomness from the Philox4_32_10 generator in \p state,
 * increment position of generator by four.
 *
 * \param state - Pointer to state to update
 *
 * \return 128-bits of pseudorandomness as a uint4, all bits valid to use.
 */

QUALIFIERS uint4 curand4(curandStatePhilox4_32_10_t *state)
{
    uint4 r;

    uint4 tmp = state->output;
    Philox_State_Incr(state);
    state->output= curand_Philox4x32_10(state->ctr,state->key);
    switch(state->STATE){
    case 0:
        return tmp;
    case 1:
        r.x = tmp.y;
        r.y = tmp.z;
        r.z = tmp.w;
        r.w = state->output.x;
        break;
    case 2:
        r.x = tmp.z;
        r.y = tmp.w;
        r.z = state->output.x;
        r.w = state->output.y;
        break;
    case 3:
        r.x = tmp.w;
        r.y = state->output.x;
        r.z = state->output.y;
        r.w = state->output.z;
        break;
    default:
        // NOT possible but needed to avoid compiler warnings
        return tmp;
    }
    return r;
}

/**
 * \brief Update Philox4_32_10 state to skip \p n elements.
 *
 * Update the Philox4_32_10 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead(unsigned long long n, curandStatePhilox4_32_10_t *state)
{
    state->STATE += (n & 3);
    n /= 4;
    if( state->STATE > 3 ){
        n += 1;
        state->STATE -= 4;
    }
    Philox_State_Incr(state, n);
    state->output = curand_Philox4x32_10(state->ctr,state->key);
}

/**
 * \brief Update Philox4_32_10 state to skip ahead \p n subsequences.
 *
 * Update the Philox4_32_10 state in \p state to skip ahead \p n subsequences.  Each
 * subsequence is \xmlonly<ph outputclass="xmlonly">2<sup>66</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
 * \xmlonly<ph outputclass="xmlonly">2<sup>66</sup></ph>\endxmlonly * n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of subsequences to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_t *state)
{
    Philox_State_Incr_hi(state, n);
    state->output = curand_Philox4x32_10(state->ctr,state->key);
}

/**
 * \brief Initialize Philox4_32_10 state.
 *
 * Initialize Philox4_32_10 state in \p state with the given \p seed, p\ subsequence,
 * and \p offset.
 *
 * All input values for \p seed, \p subseqence and \p offset are legal.  Each of the
 * \xmlonly<ph outputclass="xmlonly">2<sup>64</sup></ph>\endxmlonly possible
 * values of seed selects an independent sequence of length
 * \xmlonly<ph outputclass="xmlonly">2<sup>130</sup></ph>\endxmlonly.
 * The first
 * \xmlonly<ph outputclass="xmlonly">2<sup>66</sup> * subsequence + offset</ph>\endxmlonly.
 * values of the sequence are skipped.
 * I.e., subsequences are of length
 * \xmlonly<ph outputclass="xmlonly">2<sup>66</sup></ph>\endxmlonly.
 *
 * \param seed - Arbitrary bits to use as a seed
 * \param subsequence - Subsequence to start at
 * \param offset - Absolute offset into subsequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(unsigned long long seed,
                                 unsigned long long subsequence,
                                 unsigned long long offset,
                                 curandStatePhilox4_32_10_t *state)
{
    state->ctr = make_uint4(0, 0, 0, 0);
    state->key.x = (unsigned int)seed;
    state->key.y = (unsigned int)(seed>>32);
    state->STATE = 0;
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.f;
    state->boxmuller_extra_double = 0.;
    skipahead_sequence(subsequence, state);
    skipahead(offset, state);
}


/* MRG32k3a RNG */

/* Base generator for MRG32k3a                                              */
#if __CUDA_ARCH__ > 600
QUALIFIERS unsigned long long __curand_umad(unsigned int a, unsigned int b, unsigned long long c)
{
    unsigned long long r;
    asm("mad.wide.u32 %0, %1, %2, %3;"
        : "=l"(r) : "r"(a), "r"(b), "l"(c));
    return r;
}
QUALIFIERS unsigned long long __curand_umul(unsigned int a, unsigned int b)
{
    unsigned long long r;
    asm("mul.wide.u32 %0, %1, %2;"
        : "=l"(r) : "r"(a), "r"(b));
    return r;
}

QUALIFIERS double curand_MRG32k3a (curandStateMRG32k3a_t *state)
{
    const unsigned int m1  = 4294967087u;
    const unsigned int m2  = 4294944443u;
    const unsigned int m1c = 209u;
    const unsigned int m2c = 22853u;
    const unsigned int a12  = 1403580u;
    const unsigned int a13n = 810728u;
    const unsigned int a21  = 527612u;
    const unsigned int a23n = 1370589u;

    unsigned long long p1, p2;
    const unsigned long long p3 = __curand_umul(a13n, m1 - state->s1[0]);
    p1 = __curand_umad(a12, state->s1[1], p3);

    // Putting addition inside and changing umul to umad
    // slowed this function down on GV100
    p1 =  __curand_umul(p1 >> 32, m1c) + (p1 & 0xffffffff);
    if (p1 >= m1) p1 -= m1;

    state->s1[0] = state->s1[1]; state->s1[1] = state->s1[2]; state->s1[2] = p1;
    const unsigned long long p4 = __curand_umul(a23n, m2 - state->s2[0]);
    p2 = __curand_umad(a21, state->s2[2], p4);

    // Putting addition inside and changing umul to umad
    // slowed this function down on GV100
    p2 =  __curand_umul(p2 >> 32, m2c) + (p2 & 0xffffffff);
    p2 =  __curand_umul(p2 >> 32, m2c) + (p2 & 0xffffffff);
    if (p2 >= m2) p2 -= m2;

    state->s2[0] = state->s2[1]; state->s2[1] = state->s2[2]; state->s2[2] = p2;

    const unsigned int p5 = (unsigned int)p1 - (unsigned int)p2;
    if(p1 <= p2) return p5 + m1;
    return p5;
}
#elif __CUDA_ARCH__ > 0
/*  nj's implementation */
QUALIFIERS double curand_MRG32k3a (curandStateMRG32k3a_t *state)
{
    const double m1 = 4294967087.;
    const double m2 = 4294944443.;
    const double a12  = 1403580.;
    const double a13n = 810728.;
    const double a21  = 527612.;
    const double a23n = 1370589.;

    const double rh1 =  2.3283065498378290e-010;  /* (1.0 / m1)__hi */
    const double rl1 = -1.7354913086174288e-026;  /* (1.0 / m1)__lo */
    const double rh2 =  2.3283188252407387e-010;  /* (1.0 / m2)__hi */
    const double rl2 =  2.4081018096503646e-026;  /* (1.0 / m2)__lo */

    double q, p1, p2;
    p1 = a12 * state->s1[1] - a13n * state->s1[0];
    q = trunc (fma (p1, rh1, p1 * rl1));
    p1 -= q * m1;
    if (p1 < 0.0) p1 += m1;
    state->s1[0] = state->s1[1];   state->s1[1] = state->s1[2];   state->s1[2] = (unsigned int)p1;
    p2 = a21 * state->s2[2] - a23n * state->s2[0];
    q = trunc (fma (p2, rh2, p2 * rl2));
    p2 -= q * m2;
    if (p2 < 0.0) p2 += m2;
    state->s2[0] = state->s2[1];   state->s2[1] = state->s2[2];   state->s2[2] = (unsigned int)p2;
    if (p1 <= p2) return (p1 - p2 + m1);
    else return (p1 - p2);
}
/* end nj's implementation */
#else
QUALIFIERS double curand_MRG32k3a(curandStateMRG32k3a_t *state)
{
    double p1,p2,r;
    p1 = (MRG32K3A_A12 * state->s1[1]) - (MRG32K3A_A13N * state->s1[0]);
    p1 = curand_MRGmod(p1, MRG32K3A_MOD1);
    if (p1 < 0.0) p1 += MRG32K3A_MOD1;
    state->s1[0] = state->s1[1];
    state->s1[1] = state->s1[2];
    state->s1[2] = (unsigned int)p1;
    p2 = (MRG32K3A_A21 * state->s2[2]) - (MRG32K3A_A23N * state->s2[0]);
    p2 = curand_MRGmod(p2, MRG32K3A_MOD2);
    if (p2 < 0) p2 += MRG32K3A_MOD2;
    state->s2[0] = state->s2[1];
    state->s2[1] = state->s2[2];
    state->s2[2] = (unsigned int)p2;
    r = p1 - p2;
    if (r <= 0) r += MRG32K3A_MOD1;
    return r;
}
#endif


/**
 * \brief Return 32-bits of pseudorandomness from an MRG32k3a generator.
 *
 * Return 32-bits of pseudorandomness from the MRG32k3a generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of pseudorandomness as an unsigned int, all bits valid to use.
 */
QUALIFIERS unsigned int curand(curandStateMRG32k3a_t *state)
{
    double dRet;
    dRet = (double)curand_MRG32k3a(state)*(double)MRG32K3A_BITS_NORM;
    return (unsigned int)dRet;
}



/**
 * \brief Update MRG32k3a state to skip \p n elements.
 *
 * Update the MRG32k3a state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead(unsigned long long n, curandStateMRG32k3a_t *state)
{
    unsigned int t[3][3];
#ifdef __CUDA_ARCH__
    curand_MRGmatPow3x3( mrg32k3aM1, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(mrg32k3aM2, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#else
    curand_MRGmatPow3x3( mrg32k3aM1Host, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(mrg32k3aM2Host, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#endif
}

/**
 * \brief Update MRG32k3a state to skip ahead \p n subsequences.
 *
 * Update the MRG32k3a state in \p state to skip ahead \p n subsequences.  Each
 * subsequence is \xmlonly<ph outputclass="xmlonly">2<sup>127</sup></ph>\endxmlonly
 *
 * \xmlonly<ph outputclass="xmlonly">2<sup>76</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
 * \xmlonly<ph outputclass="xmlonly">2<sup>67</sup></ph>\endxmlonly * n elements.
 *
 * Valid values of \p n are 0 to \xmlonly<ph outputclass="xmlonly">2<sup>51</sup></ph>\endxmlonly.  Note \p n will be masked to 51 bits
 *
 * \param n - Number of subsequences to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead_subsequence(unsigned long long n, curandStateMRG32k3a_t *state)
{
    unsigned int t[3][3];
#ifdef __CUDA_ARCH__
    curand_MRGmatPow3x3( mrg32k3aM1SubSeq, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3( mrg32k3aM2SubSeq, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#else
    curand_MRGmatPow3x3( mrg32k3aM1SubSeqHost, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3( mrg32k3aM2SubSeqHost, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#endif
}

/**
 * \brief Update MRG32k3a state to skip ahead \p n sequences.
 *
 * Update the MRG32k3a state in \p state to skip ahead \p n sequences.  Each
 * sequence is \xmlonly<ph outputclass="xmlonly">2<sup>127</sup></ph>\endxmlonly elements long, so this means the function will skip ahead
 * \xmlonly<ph outputclass="xmlonly">2<sup>127</sup></ph>\endxmlonly * n elements.
 *
 * All values of \p n are valid.  Large values require more computation and so
 * will take more time to complete.
 *
 * \param n - Number of sequences to skip
 * \param state - Pointer to state to update
 */
QUALIFIERS void skipahead_sequence(unsigned long long n, curandStateMRG32k3a_t *state)
{
    unsigned int t[3][3];
#ifdef __CUDA_ARCH__
    curand_MRGmatPow3x3( mrg32k3aM1Seq, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(  mrg32k3aM2Seq, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#else
    curand_MRGmatPow3x3( mrg32k3aM1SeqHost, t, MRG32K3A_MOD1, n);
    curand_MRGmatVecMul3x3( t, state->s1, MRG32K3A_MOD1);
    curand_MRGmatPow3x3(  mrg32k3aM2SeqHost, t, MRG32K3A_MOD2, n);
    curand_MRGmatVecMul3x3( t, state->s2, MRG32K3A_MOD2);
#endif
}


/**
 * \brief Initialize MRG32k3a state.
 *
 * Initialize MRG32k3a state in \p state with the given \p seed, \p subsequence,
 * and \p offset.
 *
 * All input values of \p seed, \p subsequence, and \p offset are legal.
 * \p subsequence will be truncated to 51 bits to avoid running into the next sequence
 *
 * A value of 0 for \p seed sets the state to the values of the original
 * published version of the \p MRG32k3a algorithm.
 *
 * \param seed - Arbitrary bits to use as a seed
 * \param subsequence - Subsequence to start at
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            curandStateMRG32k3a_t *state)
{
    int i;
    for ( i=0; i<3; i++ ) {
        state->s1[i] = 12345u;
        state->s2[i] = 12345u;
    }
    if (seed != 0ull) {
        unsigned int x1 = ((unsigned int)seed) ^ 0x55555555UL;
        unsigned int x2 = (unsigned int)((seed >> 32) ^ 0xAAAAAAAAUL);
        state->s1[0] = (unsigned int)curand_MRGmodMul(x1, state->s1[0], MRG32K3A_MOD1);
        state->s1[1] = (unsigned int)curand_MRGmodMul(x2, state->s1[1], MRG32K3A_MOD1);
        state->s1[2] = (unsigned int)curand_MRGmodMul(x1, state->s1[2], MRG32K3A_MOD1);
        state->s2[0] = (unsigned int)curand_MRGmodMul(x2, state->s2[0], MRG32K3A_MOD2);
        state->s2[1] = (unsigned int)curand_MRGmodMul(x1, state->s2[1], MRG32K3A_MOD2);
        state->s2[2] = (unsigned int)curand_MRGmodMul(x2, state->s2[2], MRG32K3A_MOD2);
    }
    skipahead_subsequence( subsequence, state );
    skipahead( offset, state );
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.f;
    state->boxmuller_extra_double = 0.;
}

/**
 * \brief Update Sobol32 state to skip \p n elements.
 *
 * Update the Sobol32 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
template <typename T>
QUALIFIERS
typename CURAND_STD::enable_if<CURAND_STD::is_same<curandStateSobol32_t*, T>::value || CURAND_STD::is_same<curandStateScrambledSobol32_t*, T>::value>::type
skipahead(unsigned int n, T state)
{
    unsigned int i_gray;
    state->x = state->c;
    state->i += n;
    /* Convert state->i to gray code */
    i_gray = state->i ^ (state->i >> 1);
    for(unsigned int k = 0; k < 32; k++) {
        if(i_gray & (1 << k)) {
            state->x ^= state->direction_vectors[k];
        }
    }
    return;
}

/**
 * \brief Update Sobol64 state to skip \p n elements.
 *
 * Update the Sobol64 state in \p state to skip ahead \p n elements.
 *
 * All values of \p n are valid.
 *
 * \param n - Number of elements to skip
 * \param state - Pointer to state to update
 */
template <typename T>
QUALIFIERS
typename CURAND_STD::enable_if<CURAND_STD::is_same<curandStateSobol64_t*, T>::value || CURAND_STD::is_same<curandStateScrambledSobol64_t*, T>::value>::type
skipahead(unsigned long long n, T state)
{
    unsigned long long i_gray;
    state->x = state->c;
    state->i += n;
    /* Convert state->i to gray code */
    i_gray = state->i ^ (state->i >> 1);
    for(unsigned k = 0; k < 64; k++) {
        if(i_gray & (1ULL << k)) {
            state->x ^= state->direction_vectors[k];
        }
    }
    return;
}

/**
 * \brief Initialize Sobol32 state.
 *
 * Initialize Sobol32 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 32 unsigned ints.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
 * direction vectors for the desired dimension
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors32_t direction_vectors,
                                            unsigned int offset,
                                            curandStateSobol32_t *state)
{
    state->i = 0;
    state->c = 0;
    for(int i = 0; i < 32; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = 0;
    skipahead<curandStateSobol32_t *>(offset, state);
}
/**
 * \brief Initialize Scrambled Sobol32 state.
 *
 * Initialize Sobol32 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 32 unsigned ints.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 32 unsigned ints representing the
 direction vectors for the desired dimension
 * \param scramble_c Scramble constant
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors32_t direction_vectors,
                                            unsigned int scramble_c,
                                            unsigned int offset,
                                            curandStateScrambledSobol32_t *state)
{
    state->i = 0;
    state->c = scramble_c;
    for(int i = 0; i < 32; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = state->c;
    skipahead<curandStateScrambledSobol32_t *>(offset, state);
}

QUALIFIERS int __curand_find_trailing_zero(unsigned int x)
{
#if __CUDA_ARCH__ > 0
    int y = __ffs(~x);
    if(y)
        return y - 1;
    return 31;
#else
    int i = 1;
    while(x & 1) {
        i++;
        x >>= 1;
    }
    i = i - 1;
    return i == 32 ? 31 : i;
#endif
}

QUALIFIERS int __curand_find_trailing_zero(unsigned long long x)
{
#if __CUDA_ARCH__ > 0
    int y = __ffsll(~x);
    if(y)
        return y - 1;
    return 63;
#else
    int i = 1;
    while(x & 1) {
        i++;
        x >>= 1;
    }
    i = i - 1;
    return i == 64 ? 63 : i;
#endif
}

/**
 * \brief Initialize Sobol64 state.
 *
 * Initialize Sobol64 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 64 unsigned long longs.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
 direction vectors for the desired dimension
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors64_t direction_vectors,
                                            unsigned long long offset,
                                            curandStateSobol64_t *state)
{
    state->i = 0;
    state->c = 0;
    for(int i = 0; i < 64; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = 0;
    skipahead<curandStateSobol64_t *>(offset, state);
}

/**
 * \brief Initialize Scrambled Sobol64 state.
 *
 * Initialize Sobol64 state in \p state with the given \p direction \p vectors and
 * \p offset.
 *
 * The direction vector is a device pointer to an array of 64 unsigned long longs.
 * All input values of \p offset are legal.
 *
 * \param direction_vectors - Pointer to array of 64 unsigned long longs representing the
 direction vectors for the desired dimension
 * \param scramble_c Scramble constant
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
QUALIFIERS void curand_init(curandDirectionVectors64_t direction_vectors,
                                            unsigned long long scramble_c,
                                            unsigned long long offset,
                                            curandStateScrambledSobol64_t *state)
{
    state->i = 0;
    state->c = scramble_c;
    for(int i = 0; i < 64; i++) {
        state->direction_vectors[i] = direction_vectors[i];
    }
    state->x = state->c;
    skipahead<curandStateScrambledSobol64_t *>(offset, state);
}

/**
 * \brief Return 32-bits of quasirandomness from a Sobol32 generator.
 *
 * Return 32-bits of quasirandomness from the Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
 */

QUALIFIERS unsigned int curand(curandStateSobol32_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned int res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

/**
 * \brief Return 32-bits of quasirandomness from a scrambled Sobol32 generator.
 *
 * Return 32-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 32-bits of quasirandomness as an unsigned int, all bits valid to use.
 */

QUALIFIERS unsigned int curand(curandStateScrambledSobol32_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned int res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

/**
 * \brief Return 64-bits of quasirandomness from a Sobol64 generator.
 *
 * Return 64-bits of quasirandomness from the Sobol64 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
 */

QUALIFIERS unsigned long long curand(curandStateSobol64_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned long long res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

/**
 * \brief Return 64-bits of quasirandomness from a scrambled Sobol64 generator.
 *
 * Return 64-bits of quasirandomness from the scrambled Sobol32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return 64-bits of quasirandomness as an unsigned long long, all bits valid to use.
 */

QUALIFIERS unsigned long long curand(curandStateScrambledSobol64_t * state)
{
    /* Moving from i to i+1 element in gray code is flipping one bit,
       the trailing zero bit of i
    */
    unsigned long long res = state->x;
    state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
    state->i ++;
    return res;
}

#include "curand_uniform.h"
#include "curand_normal.h"
#include "curand_lognormal.h"
#include "curand_poisson.h"
#include "curand_discrete2.h"

__device__ static inline unsigned int *__get_precalculated_matrix(int n)
{
    if(n == 0) {
        return precalc_xorwow_matrix[n];
    }
    if(n == 2) {
        return precalc_xorwow_offset_matrix[n];
    }
    return precalc_xorwow_matrix[n];
}

#ifndef __CUDACC_RTC__
__host__ static inline unsigned int *__get_precalculated_matrix_host(int n)
{
    if(n == 1) {
        return precalc_xorwow_matrix_host[n];
    }
    if(n == 3) {
        return precalc_xorwow_offset_matrix_host[n];
    }
    return precalc_xorwow_matrix_host[n];
}
#endif // #ifndef __CUDACC_RTC__

__device__ static inline unsigned int *__get_mrg32k3a_matrix(int n)
{
    if(n == 0) {
        return mrg32k3aM1[n][0];
    }
    if(n == 2) {
        return mrg32k3aM2[n][0];
    }
    if(n == 4) {
        return mrg32k3aM1SubSeq[n][0];
    }
    if(n == 6) {
        return mrg32k3aM2SubSeq[n][0];
    }
    if(n == 8) {
        return mrg32k3aM1Seq[n][0];
    }
    if(n == 10) {
        return mrg32k3aM2Seq[n][0];
    }
    return mrg32k3aM1[n][0];
}

#ifndef __CUDACC_RTC__
__host__ static inline unsigned int *__get_mrg32k3a_matrix_host(int n)
{
    if(n == 1) {
        return mrg32k3aM1Host[n][0];
    }
    if(n == 3) {
        return mrg32k3aM2Host[n][0];
    }
    if(n == 5) {
        return mrg32k3aM1SubSeqHost[n][0];
    }
    if(n == 7) {
        return mrg32k3aM2SubSeqHost[n][0];
    }
    if(n == 9) {
        return mrg32k3aM1SeqHost[n][0];
    }
    if(n == 11) {
        return mrg32k3aM2SeqHost[n][0];
    }
    return mrg32k3aM1Host[n][0];
}

__host__ static inline double *__get__cr_lgamma_table_host(void) {
    return __cr_lgamma_table;
}
#endif // #ifndef __CUDACC_RTC__

/** @} */

#endif // !defined(CURAND_KERNEL_H_)
