/*******************************************************************************
 * Copyright (c) 2019-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/**
 * This is a header-only utility library that provides OpenCL host code with
 * routines for converting to/from cl_half values.
 *
 * Example usage:
 *
 *    #include <CL/cl_half.h>
 *    ...
 *    cl_half h = cl_half_from_float(0.5f, CL_HALF_RTE);
 *    cl_float f = cl_half_to_float(h);
 */

#ifndef OPENCL_CL_HALF_H
#define OPENCL_CL_HALF_H

#include <CL/cl_platform.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Rounding mode used when converting to cl_half.
 */
typedef enum
{
  CL_HALF_RTE, // round to nearest even
  CL_HALF_RTZ, // round towards zero
  CL_HALF_RTP, // round towards positive infinity
  CL_HALF_RTN, // round towards negative infinity
} cl_half_rounding_mode;


/* Private utility macros. */
#define CL_HALF_EXP_MASK 0x7C00
#define CL_HALF_MAX_FINITE_MAG 0x7BFF


/*
 * Utility to deal with values that overflow when converting to half precision.
 */
static inline cl_half cl_half_handle_overflow(cl_half_rounding_mode rounding_mode,
                                              uint16_t sign)
{
  if (rounding_mode == CL_HALF_RTZ)
  {
    // Round overflow towards zero -> largest finite number (preserving sign)
    return (sign << 15) | CL_HALF_MAX_FINITE_MAG;
  }
  else if (rounding_mode == CL_HALF_RTP && sign)
  {
    // Round negative overflow towards positive infinity -> most negative finite number
    return (1 << 15) | CL_HALF_MAX_FINITE_MAG;
  }
  else if (rounding_mode == CL_HALF_RTN && !sign)
  {
    // Round positive overflow towards negative infinity -> largest finite number
    return CL_HALF_MAX_FINITE_MAG;
  }

  // Overflow to infinity
  return (sign << 15) | CL_HALF_EXP_MASK;
}

/*
 * Utility to deal with values that underflow when converting to half precision.
 */
static inline cl_half cl_half_handle_underflow(cl_half_rounding_mode rounding_mode,
                                               uint16_t sign)
{
  if (rounding_mode == CL_HALF_RTP && !sign)
  {
    // Round underflow towards positive infinity -> smallest positive value
    return (sign << 15) | 1;
  }
  else if (rounding_mode == CL_HALF_RTN && sign)
  {
    // Round underflow towards negative infinity -> largest negative value
    return (sign << 15) | 1;
  }

  // Flush to zero
  return (sign << 15);
}


/**
 * Convert a cl_float to a cl_half.
 */
static inline cl_half cl_half_from_float(cl_float f, cl_half_rounding_mode rounding_mode)
{
  // Type-punning to get direct access to underlying bits
  union
  {
    cl_float f;
    uint32_t i;
  } f32;
  f32.f = f;

  // Extract sign bit
  uint16_t sign = f32.i >> 31;

  // Extract FP32 exponent and mantissa
  uint32_t f_exp = (f32.i >> (CL_FLT_MANT_DIG - 1)) & 0xFF;
  uint32_t f_mant = f32.i & ((1 << (CL_FLT_MANT_DIG - 1)) - 1);

  // Remove FP32 exponent bias
  int32_t exp = f_exp - CL_FLT_MAX_EXP + 1;

  // Add FP16 exponent bias
  uint16_t h_exp = (uint16_t)(exp + CL_HALF_MAX_EXP - 1);

  // Position of the bit that will become the FP16 mantissa LSB
  uint32_t lsb_pos = CL_FLT_MANT_DIG - CL_HALF_MANT_DIG;

  // Check for NaN / infinity
  if (f_exp == 0xFF)
  {
    if (f_mant)
    {
      // NaN -> propagate mantissa and silence it
      uint16_t h_mant = (uint16_t)(f_mant >> lsb_pos);
      h_mant |= 0x200;
      return (sign << 15) | CL_HALF_EXP_MASK | h_mant;
    }
    else
    {
      // Infinity -> zero mantissa
      return (sign << 15) | CL_HALF_EXP_MASK;
    }
  }

  // Check for zero
  if (!f_exp && !f_mant)
  {
    return (sign << 15);
  }

  // Check for overflow
  if (exp >= CL_HALF_MAX_EXP)
  {
    return cl_half_handle_overflow(rounding_mode, sign);
  }

  // Check for underflow
  if (exp < (CL_HALF_MIN_EXP - CL_HALF_MANT_DIG - 1))
  {
    return cl_half_handle_underflow(rounding_mode, sign);
  }

  // Check for value that will become denormal
  if (exp < -14)
  {
    // Denormal -> include the implicit 1 from the FP32 mantissa
    h_exp = 0;
    f_mant |= 1 << (CL_FLT_MANT_DIG - 1);

    // Mantissa shift amount depends on exponent
    lsb_pos = -exp + (CL_FLT_MANT_DIG - 25);
  }

  // Generate FP16 mantissa by shifting FP32 mantissa
  uint16_t h_mant = (uint16_t)(f_mant >> lsb_pos);

  // Check whether we need to round
  uint32_t halfway = 1 << (lsb_pos - 1);
  uint32_t mask = (halfway << 1) - 1;
  switch (rounding_mode)
  {
    case CL_HALF_RTE:
      if ((f_mant & mask) > halfway)
      {
        // More than halfway -> round up
        h_mant += 1;
      }
      else if ((f_mant & mask) == halfway)
      {
        // Exactly halfway -> round to nearest even
        if (h_mant & 0x1)
          h_mant += 1;
      }
      break;
    case CL_HALF_RTZ:
      // Mantissa has already been truncated -> do nothing
      break;
    case CL_HALF_RTP:
      if ((f_mant & mask) && !sign)
      {
        // Round positive numbers up
        h_mant += 1;
      }
      break;
    case CL_HALF_RTN:
      if ((f_mant & mask) && sign)
      {
        // Round negative numbers down
        h_mant += 1;
      }
      break;
  }

  // Check for mantissa overflow
  if (h_mant & 0x400)
  {
    h_exp += 1;
    h_mant = 0;
  }

  return (sign << 15) | (h_exp << 10) | h_mant;
}


/**
 * Convert a cl_double to a cl_half.
 */
static inline cl_half cl_half_from_double(cl_double d, cl_half_rounding_mode rounding_mode)
{
  // Type-punning to get direct access to underlying bits
  union
  {
    cl_double d;
    uint64_t i;
  } f64;
  f64.d = d;

  // Extract sign bit
  uint16_t sign = f64.i >> 63;

  // Extract FP64 exponent and mantissa
  uint64_t d_exp = (f64.i >> (CL_DBL_MANT_DIG - 1)) & 0x7FF;
  uint64_t d_mant = f64.i & (((uint64_t)1 << (CL_DBL_MANT_DIG - 1)) - 1);

  // Remove FP64 exponent bias
  int64_t exp = d_exp - CL_DBL_MAX_EXP + 1;

  // Add FP16 exponent bias
  uint16_t h_exp = (uint16_t)(exp + CL_HALF_MAX_EXP - 1);

  // Position of the bit that will become the FP16 mantissa LSB
  uint32_t lsb_pos = CL_DBL_MANT_DIG - CL_HALF_MANT_DIG;

  // Check for NaN / infinity
  if (d_exp == 0x7FF)
  {
    if (d_mant)
    {
      // NaN -> propagate mantissa and silence it
      uint16_t h_mant = (uint16_t)(d_mant >> lsb_pos);
      h_mant |= 0x200;
      return (sign << 15) | CL_HALF_EXP_MASK | h_mant;
    }
    else
    {
      // Infinity -> zero mantissa
      return (sign << 15) | CL_HALF_EXP_MASK;
    }
  }

  // Check for zero
  if (!d_exp && !d_mant)
  {
    return (sign << 15);
  }

  // Check for overflow
  if (exp >= CL_HALF_MAX_EXP)
  {
    return cl_half_handle_overflow(rounding_mode, sign);
  }

  // Check for underflow
  if (exp < (CL_HALF_MIN_EXP - CL_HALF_MANT_DIG - 1))
  {
    return cl_half_handle_underflow(rounding_mode, sign);
  }

  // Check for value that will become denormal
  if (exp < -14)
  {
    // Include the implicit 1 from the FP64 mantissa
    h_exp = 0;
    d_mant |= (uint64_t)1 << (CL_DBL_MANT_DIG - 1);

    // Mantissa shift amount depends on exponent
    lsb_pos = (uint32_t)(-exp + (CL_DBL_MANT_DIG - 25));
  }

  // Generate FP16 mantissa by shifting FP64 mantissa
  uint16_t h_mant = (uint16_t)(d_mant >> lsb_pos);

  // Check whether we need to round
  uint64_t halfway = (uint64_t)1 << (lsb_pos - 1);
  uint64_t mask = (halfway << 1) - 1;
  switch (rounding_mode)
  {
    case CL_HALF_RTE:
      if ((d_mant & mask) > halfway)
      {
        // More than halfway -> round up
        h_mant += 1;
      }
      else if ((d_mant & mask) == halfway)
      {
        // Exactly halfway -> round to nearest even
        if (h_mant & 0x1)
          h_mant += 1;
      }
      break;
    case CL_HALF_RTZ:
      // Mantissa has already been truncated -> do nothing
      break;
    case CL_HALF_RTP:
      if ((d_mant & mask) && !sign)
      {
        // Round positive numbers up
        h_mant += 1;
      }
      break;
    case CL_HALF_RTN:
      if ((d_mant & mask) && sign)
      {
        // Round negative numbers down
        h_mant += 1;
      }
      break;
  }

  // Check for mantissa overflow
  if (h_mant & 0x400)
  {
    h_exp += 1;
    h_mant = 0;
  }

  return (sign << 15) | (h_exp << 10) | h_mant;
}


/**
 * Convert a cl_half to a cl_float.
 */
static inline cl_float cl_half_to_float(cl_half h)
{
  // Type-punning to get direct access to underlying bits
  union
  {
    cl_float f;
    uint32_t i;
  } f32;

  // Extract sign bit
  uint16_t sign = h >> 15;

  // Extract FP16 exponent and mantissa
  uint16_t h_exp = (h >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
  uint16_t h_mant = h & 0x3FF;

  // Remove FP16 exponent bias
  int32_t exp = h_exp - CL_HALF_MAX_EXP + 1;

  // Add FP32 exponent bias
  uint32_t f_exp = exp + CL_FLT_MAX_EXP - 1;

  // Check for NaN / infinity
  if (h_exp == 0x1F)
  {
    if (h_mant)
    {
      // NaN -> propagate mantissa and silence it
      uint32_t f_mant = h_mant << (CL_FLT_MANT_DIG - CL_HALF_MANT_DIG);
      f_mant |= 0x400000;
      f32.i = (sign << 31) | 0x7F800000 | f_mant;
      return f32.f;
    }
    else
    {
      // Infinity -> zero mantissa
      f32.i = (sign << 31) | 0x7F800000;
      return f32.f;
    }
  }

  // Check for zero / denormal
  if (h_exp == 0)
  {
    if (h_mant == 0)
    {
      // Zero -> zero exponent
      f_exp = 0;
    }
    else
    {
      // Denormal -> normalize it
      // - Shift mantissa to make most-significant 1 implicit
      // - Adjust exponent accordingly
      uint32_t shift = 0;
      while ((h_mant & 0x400) == 0)
      {
        h_mant <<= 1;
        shift++;
      }
      h_mant &= 0x3FF;
      f_exp -= shift - 1;
    }
  }

  f32.i = (sign << 31) | (f_exp << 23) | (h_mant << 13);
  return f32.f;
}


#undef CL_HALF_EXP_MASK
#undef CL_HALF_MAX_FINITE_MAG


#ifdef __cplusplus
}
#endif


#endif  /* OPENCL_CL_HALF_H */
