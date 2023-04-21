
 /* Copyright 2010-2018 NVIDIA Corporation.  All rights reserved.
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


#if !defined(CURAND_UNIFORM_H_)
#define CURAND_UNIFORM_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#ifndef __CUDACC_RTC__
#include <math.h>
#endif // __CUDACC_RTC__

#include "curand_mrg32k3a.h"
#include "curand_mtgp32_kernel.h"
#include "curand_philox4x32_x.h"


QUALIFIERS float _curand_uniform(unsigned int x)
{
    return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
}

QUALIFIERS float4 _curand_uniform4(uint4 x)
{
    float4 y;
    y.x = x.x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.y = x.y * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.z = x.z * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    y.w = x.w * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
    return y;
}

QUALIFIERS float _curand_uniform(unsigned long long x)
{
    unsigned int t;
    t = (unsigned int)(x >> 32);
    return t * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
}

QUALIFIERS double _curand_uniform_double(unsigned int x)
{
    return x * CURAND_2POW32_INV_DOUBLE + CURAND_2POW32_INV_DOUBLE;
}

QUALIFIERS double _curand_uniform_double(unsigned long long x)
{
    return (x >> 11) * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
}

QUALIFIERS double _curand_uniform_double_hq(unsigned int x, unsigned int y)
{
    unsigned long long z = (unsigned long long)x ^
        ((unsigned long long)y << (53 - 32));
    return z * CURAND_2POW53_INV_DOUBLE + (CURAND_2POW53_INV_DOUBLE/2.0);
}

QUALIFIERS float curand_uniform(curandStateTest_t *state)
{
    return _curand_uniform(curand(state));
}

QUALIFIERS double curand_uniform_double(curandStateTest_t *state)
{
    return _curand_uniform_double(curand(state));
}

/**
 * \brief Return a uniformly distributed float from an XORWOW generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses one call.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateXORWOW_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from an XORWOW generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the XORWOW generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation may use any number of calls to \p curand() to
 * get enough random bits to create the return value.  The current
 * implementation uses exactly two calls.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateXORWOW_t *state)
{
    unsigned int x, y;
    x = curand(state);
    y = curand(state);
    return _curand_uniform_double_hq(x, y);
}
/**
 * \brief Return a uniformly distributed float from an MRG32k3a generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the MRG32k3a generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation returns up to 23 bits of mantissa, with the minimum
 * return value \f$ 2^{-32} \f$
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateMRG32k3a_t *state)
{
    return ((float)(curand_MRG32k3a(state)*MRG32K3A_NORM));
}

/**
 * \brief Return a uniformly distributed double from an MRG32k3a generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the MRG32k3a generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * Note the implementation returns at most 32 random bits of mantissa as
 * outlined in the seminal paper by L'Ecuyer.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateMRG32k3a_t *state)
{
    return curand_MRG32k3a(state)*MRG32K3A_NORM;
}



/**
 * \brief Return a uniformly distributed tuple of 2 doubles from an Philox4_32_10 generator.
 *
 * Return a uniformly distributed 2 doubles (double4) between \p 0.0 and \p 1.0
 * from the Philox4_32_10 generator in \p state, increment position of generator by 4.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return 2 uniformly distributed doubles between \p 0.0 and \p 1.0
 */

QUALIFIERS double2 curand_uniform2_double(curandStatePhilox4_32_10_t *state)
{
    uint4 _x;
    double2 result;
    _x = curand4(state);
    result.x = _curand_uniform_double_hq(_x.x,_x.y);
    result.y = _curand_uniform_double_hq(_x.z,_x.w);
    return result;
}


// not a part of API
QUALIFIERS double4 curand_uniform4_double(curandStatePhilox4_32_10_t *state)
{
    uint4 _x, _y;
    double4 result;
    _x = curand4(state);
    _y = curand4(state);
    result.x = _curand_uniform_double_hq(_x.x,_x.y);
    result.y = _curand_uniform_double_hq(_x.z,_x.w);
    result.z = _curand_uniform_double_hq(_y.x,_y.y);
    result.w = _curand_uniform_double_hq(_y.z,_y.w);
    return result;
}

/**
 * \brief Return a uniformly distributed float from a Philox4_32_10 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the Philox4_32_10 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0 and \p 1.0
 *
 */
QUALIFIERS float curand_uniform(curandStatePhilox4_32_10_t *state)
{
   return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed tuple of 4 floats from a Philox4_32_10 generator.
 *
 * Return a uniformly distributed 4 floats between \p 0.0f and \p 1.0f
 * from the Philox4_32_10 generator in \p state, increment position of generator by 4.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0 and \p 1.0
 *
 */
QUALIFIERS float4 curand_uniform4(curandStatePhilox4_32_10_t *state)
{
   return _curand_uniform4(curand4(state));
}

/**
 * \brief Return a uniformly distributed float from a MTGP32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the MTGP32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateMtgp32_t *state)
{
    return _curand_uniform(curand(state));
}
/**
 * \brief Return a uniformly distributed double from a MTGP32 generator.
 *
 * Return a uniformly distributed double between \p 0.0f and \p 1.0f
 * from the MTGP32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * Note that the implementation uses only 32 random bits to generate a single double
 * precision value.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0f and \p 1.0f
 */
QUALIFIERS double curand_uniform_double(curandStateMtgp32_t *state)
{
    return _curand_uniform_double(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Philox4_32_10 generator.
 *
 * Return a uniformly distributed double between \p 0.0f and \p 1.0f
 * from the Philox4_32_10 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * Note that the implementation uses only 32 random bits to generate a single double
 * precision value.
 *
 * \p curand_uniform2_double() is recommended for higher quality uniformly distributed
 * double precision values.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0f and \p 1.0f
 */

QUALIFIERS double curand_uniform_double(curandStatePhilox4_32_10_t *state)
{
    return _curand_uniform_double(curand(state));
}


/**
 * \brief Return a uniformly distributed float from a Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateSobol32_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * Note that the implementation uses only 32 random bits to generate a single double
 * precision value.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateSobol32_t *state)
{
    return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateScrambledSobol32_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol32 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the scrambled Sobol32 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * Note that the implementation uses only 32 random bits to generate a single double
 * precision value.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateScrambledSobol32_t *state)
{
    return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateSobol64_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateSobol64_t *state)
{
    return _curand_uniform_double(curand(state));
}
/**
 * \brief Return a uniformly distributed float from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0f but includes \p 1.0f.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand().
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
QUALIFIERS float curand_uniform(curandStateScrambledSobol64_t *state)
{
    return _curand_uniform(curand(state));
}

/**
 * \brief Return a uniformly distributed double from a scrambled Sobol64 generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from the scrambled Sobol64 generator in \p state, increment position of generator.
 * Output range excludes \p 0.0 but includes \p 1.0.  Denormalized floating
 * point outputs are never returned.
 *
 * The implementation is guaranteed to use a single call to \p curand()
 * to preserve the quasirandom properties of the sequence.
 *
 * \param state - Pointer to state to update
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
QUALIFIERS double curand_uniform_double(curandStateScrambledSobol64_t *state)
{
    return _curand_uniform_double(curand(state));
}

#endif // !defined(CURAND_UNIFORM_H_)
