//==---- math.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MATH_HPP__
#define __DPCT_MATH_HPP__

#include <limits>
#include <climits>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace dpct {
namespace detail {
template <typename VecT, class BinaryOperation, class = void>
class vectorized_binary {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    VecT v4;
    for (size_t i = 0; i < v4.size(); ++i) {
      v4[i] = binary_op(a[i], b[i]);
    }
    return v4;
  }
};
template <typename VecT, class BinaryOperation>
class vectorized_binary<
    VecT, BinaryOperation,
    std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>> {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    return binary_op(a, b).template as<VecT>();
  }
};

template <typename T> inline bool isnan(const T a) { return sycl::isnan(a); }
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
inline bool isnan(const sycl::ext::oneapi::bfloat16 a) {
  return sycl::ext::oneapi::experimental::isnan(a);
}
#endif
} // namespace detail

/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \returns The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::fast_length(sycl::float2(a[0], a[1]));
  case 3:
    return sycl::fast_length(sycl::float3(a[0], a[1], a[2]));
  case 4:
    return sycl::fast_length(sycl::float4(a[0], a[1], a[2], a[3]));
  case 0:
    return 0;
  default:
    float f = 0;
    for (int i = 0; i < len; ++i)
      f += a[i] * a[i];
    return sycl::sqrt(f);
  }
}

/// Calculate the square root of the input array.
/// \param [in] a The array pointer
/// \param [in] len Length of the array
/// \returns The square root
template <typename T> inline T length(const T *a, const int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::length(sycl::vec<T, 2>(a[0], a[1]));
  case 3:
    return sycl::length(sycl::vec<T, 3>(a[0], a[1], a[2]));
  case 4:
    return sycl::length(sycl::vec<T, 4>(a[0], a[1], a[2], a[3]));
  default:
    T ret = 0;
    for (int i = 0; i < len; ++i)
      ret += a[i] * a[i];
    return sycl::sqrt(ret);
  }
}

/// Returns min(max(val, min_val), max_val)
/// \param [in] val The input value
/// \param [in] min_val The minimum value
/// \param [in] max_val The maximum value
/// \returns the value between min_val and max_val
template <typename T> inline T clamp(T val, T min_val, T max_val) {
  return sycl::clamp(val, min_val, max_val);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16 clamp(sycl::ext::oneapi::bfloat16 val,
                                         sycl::ext::oneapi::bfloat16 min_val,
                                         sycl::ext::oneapi::bfloat16 max_val) {
  if (val < min_val)
    return min_val;
  if (val > max_val)
    return max_val;
  return val;
}
#endif
template <typename T>
inline sycl::marray<T, 2> clamp(sycl::marray<T, 2> val,
                                sycl::marray<T, 2> min_val,
                                sycl::marray<T, 2> max_val) {
  return {clamp(val[0], min_val[0], max_val[0]),
          clamp(val[1], min_val[1], max_val[1])};
}

/// Performs comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, T, T>, bool>, bool>
compare(const T a, const T b, const BinaryOperation binary_op) {
  return binary_op(a, b);
}
template <typename T>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<std::not_equal_to<>, T, T>, bool>, bool>
compare(const T a, const T b, const std::not_equal_to<> binary_op) {
  return !detail::isnan(a) && !detail::isnan(b) && binary_op(a, b);
}

/// Performs unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, T, T>, bool>, bool>
unordered_compare(const T a, const T b, const BinaryOperation binary_op) {
  return detail::isnan(a) || detail::isnan(b) || binary_op(a, b);
}

/// Performs 2 element comparison and return true if both results are true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, bool>
compare_both(const T a, const T b, const BinaryOperation binary_op) {
  return compare(a[0], b[0], binary_op) && compare(a[1], b[1], binary_op);
}

/// Performs 2 element unordered comparison and return true if both results are
/// true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, bool>
unordered_compare_both(const T a, const T b, const BinaryOperation binary_op) {
  return unordered_compare(a[0], b[0], binary_op) &&
         unordered_compare(a[1], b[1], binary_op);
}

/// Performs 2 element comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, T>
compare(const T a, const T b, const BinaryOperation binary_op) {
  return {compare(a[0], b[0], binary_op), compare(a[1], b[1], binary_op)};
}

/// Performs 2 elements comparison, compare result of each element is 0 (false)
/// or 0xffff (true), returns an unsigned int by composing compare result of two
/// elements.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline unsigned compare_mask(const sycl::vec<T, 2> a, const sycl::vec<T, 2> b,
                             const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-compare(a[0], b[0], binary_op),
                             -compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}
template <typename T, class BinaryOperation>
inline unsigned compare_mask(const sycl::marray<T, 2> a,
                             const sycl::marray<T, 2> b,
                             const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-compare(a[0], b[0], binary_op),
                             -compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}

/// Performs 2 element unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline std::enable_if_t<T::size() == 2, T>
unordered_compare(const T a, const T b, const BinaryOperation binary_op) {
  return {unordered_compare(a[0], b[0], binary_op),
          unordered_compare(a[1], b[1], binary_op)};
}

/// Performs 2 elements unordered comparison, compare result of each element is
/// 0 (false) or 0xffff (true), returns an unsigned int by composing compare
/// result of two elements.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename T, class BinaryOperation>
inline unsigned unordered_compare_mask(const sycl::vec<T, 2> a,
                                       const sycl::vec<T, 2> b,
                                       const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-unordered_compare(a[0], b[0], binary_op),
                             -unordered_compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}
template <typename T, class BinaryOperation>
inline unsigned unordered_compare_mask(const sycl::marray<T, 2> a,
                                       const sycl::marray<T, 2> b,
                                       const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-unordered_compare(a[0], b[0], binary_op),
                             -unordered_compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}

/// Bitfield-extract.
///
/// \tparam T The type of \param source value, must be an integer.
/// \param source The source value to extracting.
/// \param bit_start The position to start extracting.
/// \param num_bits The number of bits to extracting.
template <typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T>
bfe(const T source, const uint32_t bit_start, const uint32_t num_bits) {
  const T mask = (T{1} << num_bits) - 1;
  return (source >> bit_start) & mask;
}

/// Bitfield-extract with boundary checking.
///
/// Extract bit field from \param source and return the zero or sign-extended
/// result. Source \param bit_start gives the bit field starting bit position,
/// and source \param num_bits gives the bit field length in bits.
///
/// The result is padded with the sign bit of the extracted field. If the start
/// position is beyond the msb of the input, the result was filled with the
/// replicated sign bit of the extracted field.
///
/// \tparam T The type of \param source value, must be an integer.
/// \param source The source value to extracting.
/// \param bit_start The position to start extracting.
/// \param num_bits The number of bits to extracting.
template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T>
bfe_safe(const T source, const uint32_t bit_start, const uint32_t num_bits) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                std::is_same_v<T, int32_t>) {
    int32_t res{};
    asm volatile("bfe.s32 %0, %1, %2, %3;"
                 : "=r"(res)
                 : "r"((int32_t)source), "r"(bit_start), "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, uint8_t> ||
                       std::is_same_v<T, uint16_t> ||
                       std::is_same_v<T, uint32_t>) {
    uint32_t res{};
    asm volatile("bfe.u32 %0, %1, %2, %3;"
                 : "=r"(res)
                 : "r"((uint32_t)source), "r"(bit_start), "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    T res{};
    asm volatile("bfe.s64 %0, %1, %2, %3;"
                 : "=l"(res)
                 : "l"(source), "r"(bit_start), "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    T res{};
    asm volatile("bfe.u64 %0, %1, %2, %3;"
                 : "=l"(res)
                 : "l"(source), "r"(bit_start), "r"(num_bits));
    return res;
  }
#endif
  const uint32_t bit_width = CHAR_BIT * sizeof(T);
  const uint32_t pos = std::min(bit_start, bit_width);
  const uint32_t len = std::min(pos + num_bits, bit_width) - pos;
  if constexpr (std::is_signed_v<T>) {
    const T mask = (T{1} << len) - 1;

    // Find the sign-bit, the result is padded with the sign bit of the
    // extracted field.
    //
    // sign_bit = len == 0 ? 0 : source[min(pos + len - 1, bit_width - 1)]
    const uint32_t sign_bit_pos = std::min(pos + len - 1, bit_width - 1);
    const T sign_bit = len != 0 && ((source >> sign_bit_pos) & 1);
    const T sign_bit_padding = (-sign_bit & ~mask);
    return ((source >> pos) & mask) | sign_bit_padding;
  } else {
    return dpct::bfe(source, pos, len);
  }
}

/// Bitfield-insert.
///
/// \tparam T The type of \param x and \param y , must be an unsigned integer.
/// \param x The source of the bitfield.
/// \param y The source where bitfield is inserted.
/// \param bit_start The position to start insertion.
/// \param num_bits The number of bits to insertion.
template <typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T>
bfi(const T x, const T y, const uint32_t bit_start, const uint32_t num_bits) {
  constexpr unsigned bit_width = CHAR_BIT * sizeof(T);

  // if bit_start > bit_width || len == 0, should return y.
  const uint32_t ignore_bfi = bit_start > bit_width || num_bits == 0;
  T extract_bitfield_mask = (~(T{0}) >> (bit_width - num_bits)) << bit_start;
  T clean_bitfield_mask = ~extract_bitfield_mask;
  return (y & (-ignore_bfi | clean_bitfield_mask)) |
         (~-ignore_bfi & ((x << bit_start) & extract_bitfield_mask));
}

/// Bitfield-insert with boundary checking.
///
/// Align and insert a bit field from \param x into \param y . Source \param
/// bit_start gives the starting bit position for the insertion, and source
/// \param num_bits gives the bit field length in bits.
///
/// \tparam T The type of \param x and \param y , must be an unsigned integer.
/// \param x The source of the bitfield.
/// \param y The source where bitfield is inserted.
/// \param bit_start The position to start insertion.
/// \param num_bits The number of bits to insertion.
template <typename T>
inline std::enable_if_t<std::is_unsigned_v<T>, T>
bfi_safe(const T x, const T y, const uint32_t bit_start,
         const uint32_t num_bits) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
                std::is_same_v<T, uint32_t>) {
    uint32_t res{};
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;"
                 : "=r"(res)
                 : "r"((uint32_t)x), "r"((uint32_t)y), "r"(bit_start),
                   "r"(num_bits));
    return res;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    uint64_t res{};
    asm volatile("bfi.b64 %0, %1, %2, %3, %4;"
                 : "=l"(res)
                 : "l"(x), "l"(y), "r"(bit_start), "r"(num_bits));
    return res;
  }
#endif
  constexpr unsigned bit_width = CHAR_BIT * sizeof(T);
  const uint32_t pos = std::min(bit_start, bit_width);
  const uint32_t len = std::min(pos + num_bits, bit_width) - pos;
  return dpct::bfi(x, y, pos, len);
}

/// Determine whether 2 element value is NaN.
/// \param [in] a The input value
/// \returns the comparison result
template <typename T>
inline std::enable_if_t<T::size() == 2, T> isnan(const T a) {
  return {detail::isnan(a[0]), detail::isnan(a[1])};
}

/// Emulated function for __funnelshift_l
inline unsigned int funnelshift_l(unsigned int low, unsigned int high,
                                  unsigned int shift) {
  return (sycl::upsample(high, low) << (shift & 31U)) >> 32;
}

/// Emulated function for __funnelshift_lc
inline unsigned int funnelshift_lc(unsigned int low, unsigned int high,
                                   unsigned int shift) {
  return (sycl::upsample(high, low) << sycl::min(shift, 32U)) >> 32;
}

/// Emulated function for __funnelshift_r
inline unsigned int funnelshift_r(unsigned int low, unsigned int high,
                                  unsigned int shift) {
  return (sycl::upsample(high, low) >> (shift & 31U)) & 0xFFFFFFFF;
}

/// Emulated function for __funnelshift_rc
inline unsigned int funnelshift_rc(unsigned int low, unsigned int high,
                                   unsigned int shift) {
  return (sycl::upsample(high, low) >> sycl::min(shift, 32U)) & 0xFFFFFFFF;
}

/// cbrt function wrapper.
template <typename T> inline T cbrt(T val) { return sycl::cbrt((T)val); }

// min function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
inline double min(const double a, const float b) {
  return sycl::fmin(a, static_cast<double>(b));
}
inline double min(const float a, const double b) {
  return sycl::fmin(static_cast<double>(a), b);
}
inline float min(const float a, const float b) { return sycl::fmin(a, b); }
inline double min(const double a, const double b) { return sycl::fmin(a, b); }
inline std::uint32_t min(const std::uint32_t a, const std::int32_t b) {
  return sycl::min(a, static_cast<std::uint32_t>(b));
}
inline std::uint32_t min(const std::int32_t a, const std::uint32_t b) {
  return sycl::min(static_cast<std::uint32_t>(a), b);
}
inline std::int32_t min(const std::int32_t a, const std::int32_t b) {
  return sycl::min(a, b);
}
inline std::uint32_t min(const std::uint32_t a, const std::uint32_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::int64_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::int64_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
inline std::int64_t min(const std::int64_t a, const std::int64_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::uint64_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::int32_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::int32_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
inline std::uint64_t min(const std::uint64_t a, const std::uint32_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::uint32_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
// max function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
inline double max(const double a, const float b) {
  return sycl::fmax(a, static_cast<double>(b));
}
inline double max(const float a, const double b) {
  return sycl::fmax(static_cast<double>(a), b);
}
inline float max(const float a, const float b) { return sycl::fmax(a, b); }
inline double max(const double a, const double b) { return sycl::fmax(a, b); }
inline std::uint32_t max(const std::uint32_t a, const std::int32_t b) {
  return sycl::max(a, static_cast<std::uint32_t>(b));
}
inline std::uint32_t max(const std::int32_t a, const std::uint32_t b) {
  return sycl::max(static_cast<std::uint32_t>(a), b);
}
inline std::int32_t max(const std::int32_t a, const std::int32_t b) {
  return sycl::max(a, b);
}
inline std::uint32_t max(const std::uint32_t a, const std::uint32_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::int64_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::int64_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}
inline std::int64_t max(const std::int64_t a, const std::int64_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::uint64_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::int32_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::int32_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}
inline std::uint64_t max(const std::uint64_t a, const std::uint32_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::uint32_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}

// pow functions overload.
inline float pow(const float a, const int b) { return sycl::pown(a, b); }
inline double pow(const double a, const int b) { return sycl::pown(a, b); }
inline float pow(const float a, const float b) { return sycl::pow(a, b); }
inline double pow(const double a, const double b) { return sycl::pow(a, b); }
template <typename T, typename U>
inline typename std::enable_if_t<std::is_floating_point_v<T>, T>
pow(const T a, const U b) {
  return sycl::pow(a, static_cast<T>(b));
}
template <typename T, typename U>
inline typename std::enable_if_t<!std::is_floating_point_v<T>, double>
pow(const T a, const U b) {
  return sycl::pow(static_cast<double>(a), static_cast<double>(b));
}

namespace detail {
template <typename T>
constexpr bool is_floating_point =
    std::disjunction_v<std::is_floating_point<T>, std::is_same<T, sycl::half>
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
                       ,
                       std::is_same<T, sycl::ext::oneapi::bfloat16>
#endif
                       >;
} // namespace detail

/// Performs relu saturation.
/// \param [in] a The input value
/// \returns the relu saturation result
template <typename T> inline T relu(T a) {
  T zero{};
  if constexpr (detail::is_floating_point<T>)
    return !detail::isnan(a) && a < zero ? zero : a;
  else
    return a < zero ? zero : a;
}
template <class T> inline sycl::vec<T, 2> relu(const sycl::vec<T, 2> a) {
  return {relu(a[0]), relu(a[1])};
}
template <class T> inline sycl::marray<T, 2> relu(const sycl::marray<T, 2> a) {
  return {relu(a[0]), relu(a[1])};
}

/// Performs complex number multiply addition.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns the operation result
template <typename T>
inline sycl::vec<T, 2> complex_mul_add(const sycl::vec<T, 2> a,
                                       const sycl::vec<T, 2> b,
                                       const sycl::vec<T, 2> c) {
  return sycl::vec<T, 2>{a[0] * b[0] - a[1] * b[1] + c[0],
                         a[0] * b[1] + a[1] * b[0] + c[1]};
}
template <typename T>
inline sycl::marray<T, 2> complex_mul_add(const sycl::marray<T, 2> a,
                                          const sycl::marray<T, 2> b,
                                          const sycl::marray<T, 2> c) {
  return sycl::marray<T, 2>{a[0] * b[0] - a[1] * b[1] + c[0],
                            a[0] * b[1] + a[1] * b[0] + c[1]};
}

/// Performs 2 elements comparison and returns the bigger one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the bigger value
template <typename T> inline T fmax_nan(const T a, const T b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmax(a, b);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16
fmax_nan(const sycl::ext::oneapi::bfloat16 a,
         const sycl::ext::oneapi::bfloat16 b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmax(float(a), float(b));
}
#endif
template <typename T>
inline sycl::vec<T, 2> fmax_nan(const sycl::vec<T, 2> a,
                                const sycl::vec<T, 2> b) {
  return {fmax_nan(a[0], b[0]), fmax_nan(a[1], b[1])};
}
template <typename T>
inline sycl::marray<T, 2> fmax_nan(const sycl::marray<T, 2> a,
                                   const sycl::marray<T, 2> b) {
  return {fmax_nan(a[0], b[0]), fmax_nan(a[1], b[1])};
}

/// Performs 2 elements comparison and returns the smaller one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the smaller value
template <typename T> inline T fmin_nan(const T a, const T b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmin(a, b);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16
fmin_nan(const sycl::ext::oneapi::bfloat16 a,
         const sycl::ext::oneapi::bfloat16 b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmin(float(a), float(b));
}
#endif
template <typename T>
inline sycl::vec<T, 2> fmin_nan(const sycl::vec<T, 2> a,
                                const sycl::vec<T, 2> b) {
  return {fmin_nan(a[0], b[0]), fmin_nan(a[1], b[1])};
}
template <typename T>
inline sycl::marray<T, 2> fmin_nan(const sycl::marray<T, 2> a,
                                   const sycl::marray<T, 2> b) {
  return {fmin_nan(a[0], b[0]), fmin_nan(a[1], b[1])};
}

/// A sycl::abs wrapper functors.
struct abs {
  template <typename T> auto operator()(const T x) const {
    return sycl::abs(x);
  }
};

/// A sycl::abs_diff wrapper functors.
struct abs_diff {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::abs_diff(x, y);
  }
};

/// A sycl::add_sat wrapper functors.
struct add_sat {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::add_sat(x, y);
  }
};

/// A sycl::rhadd wrapper functors.
struct rhadd {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::rhadd(x, y);
  }
};

/// A sycl::hadd wrapper functors.
struct hadd {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::hadd(x, y);
  }
};

/// A sycl::max wrapper functors.
struct maximum {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::max(x, y);
  }
};

/// A sycl::min wrapper functors.
struct minimum {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::min(x, y);
  }
};

/// A sycl::sub_sat wrapper functors.
struct sub_sat {
  template <typename T> auto operator()(const T x, const T y) const {
    return sycl::sub_sat(x, y);
  }
};

/// Compute vectorized binary operation value for two values, with each value
/// treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] BinaryOperation The binary operation class
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized binary operation value of the two values
template <typename VecT, class BinaryOperation>
inline unsigned vectorized_binary(unsigned a, unsigned b,
                                  const BinaryOperation binary_op) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 =
      detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <typename S, typename T> inline T vectorized_isgreater(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = v2 > v3;
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized max for two values, with each value treated as a vector
/// type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
template <typename S, typename T> inline T vectorized_max(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::max(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized min for two values, with each value treated as a vector
/// type \p S.
/// \tparam [in] S The type of the vector
/// \tparam [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
template <typename S, typename T> inline T vectorized_min(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::min(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized unary operation for a value, with the value treated as a
/// vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] UnaryOperation The unary operation class
/// \param [in] a The input value
/// \returns The vectorized unary operation value of the input value
template <typename VecT, class UnaryOperation>
inline unsigned vectorized_unary(unsigned a, const UnaryOperation unary_op) {
  sycl::vec<unsigned, 1> v0{a};
  auto v1 = v0.as<VecT>();
  auto v2 = unary_op(v1);
  v0 = v2.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized absolute difference for two values without modulo
/// overflow, with each value treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized absolute difference of the two values
template <typename VecT>
inline unsigned vectorized_sum_abs_diff(unsigned a, unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 = sycl::abs_diff(v2, v3);
  unsigned sum = 0;
  for (size_t i = 0; i < v4.size(); ++i) {
    sum += v4[i];
  }
  return sum;
}

namespace detail {
/// Extend the 'val' to 'bit' size, zero extend for unsigned int and signed
/// extend for signed int.
template <typename T> inline auto zero_or_signed_extent(T val, unsigned bit) {
  if constexpr (std::is_signed_v<T>) {
    if constexpr (std::is_same_v<T, int32_t>) {
      assert(bit < 64 &&
             "When extend int32 value, bit must be smaller than 64.");
      return int64_t(val) << (64 - bit) >> (64 - bit);
    } else if constexpr (std::is_same_v<T, int16_t>) {
      assert(bit < 32 &&
             "When extend int16 value, bit must be smaller than 32.");
      return int32_t(val) << (32 - bit) >> (32 - bit);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      assert(bit < 16 &&
             "When extend int8 value, bit must be smaller than 16.");
      return int16_t(val) << (16 - bit) >> (16 - bit);
    } else {
      assert(bit < 64 && "Cannot extend int64 value.");
      return val;
    }
  } else
    return val;
}

template <typename RetT, bool NeedSat, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<RetT> && sizeof(AT) == 4 && sizeof(BT) == 4 &&
        sizeof(RetT) == 4,
    RetT>
extend_binary(AT a, BT b, BinaryOperation binary_op) {
  int64_t extend_a = zero_or_signed_extent(a, 33);
  int64_t extend_b = zero_or_signed_extent(b, 33);
  int64_t ret = binary_op(extend_a, extend_b);
  if constexpr (NeedSat)
    return dpct::clamp<int64_t>(ret, std::numeric_limits<RetT>::min(),
                                std::numeric_limits<RetT>::max());
  return ret;
}

template <typename RetT, bool NeedSat, typename AT, typename BT, typename CT,
          typename BinaryOperation1, typename BinaryOperation2>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<CT> && std::is_integral_v<RetT> && sizeof(AT) == 4 &&
        sizeof(BT) == 4 && sizeof(CT) == 4 && sizeof(RetT) == 4,
    RetT>
extend_binary(AT a, BT b, CT c, BinaryOperation1 binary_op,
              BinaryOperation2 second_op) {
  int64_t extend_a = zero_or_signed_extent(a, 33);
  int64_t extend_b = zero_or_signed_extent(b, 33);
  int64_t extend_temp =
      zero_or_signed_extent(binary_op(extend_a, extend_b), 34);
  if constexpr (NeedSat)
    extend_temp =
        dpct::clamp<int64_t>(extend_temp, std::numeric_limits<RetT>::min(),
                             std::numeric_limits<RetT>::max());
  int64_t extend_c = zero_or_signed_extent(c, 33);
  return second_op(extend_temp, extend_c);
}

template <typename T> sycl::vec<int32_t, 2> extractAndExtend2(T a) {
  sycl::vec<int32_t, 2> ret;
  sycl::vec<T, 1> va{a};
  if constexpr (std::is_signed_v<T>) {
    auto v = va.template as<sycl::vec<int16_t, 2>>();
    ret[0] = zero_or_signed_extent(v[0], 17);
    ret[1] = zero_or_signed_extent(v[1], 17);
  } else {
    auto v = va.template as<sycl::vec<uint16_t, 2>>();
    ret[0] = zero_or_signed_extent(v[0], 17);
    ret[1] = zero_or_signed_extent(v[1], 17);
  }
  return ret;
}

template <typename T> sycl::vec<int16_t, 4> extractAndExtend4(T a) {
  sycl::vec<int16_t, 4> ret;
  sycl::vec<T, 1> va{a};
  if constexpr (std::is_signed_v<T>) {
    auto v = va.template as<sycl::vec<int8_t, 4>>();
    ret[0] = zero_or_signed_extent(v[0], 9);
    ret[1] = zero_or_signed_extent(v[1], 9);
    ret[2] = zero_or_signed_extent(v[2], 9);
    ret[3] = zero_or_signed_extent(v[3], 9);
  } else {
    auto v = va.template as<sycl::vec<uint8_t, 4>>();
    ret[0] = zero_or_signed_extent(v[0], 9);
    ret[1] = zero_or_signed_extent(v[1], 9);
    ret[2] = zero_or_signed_extent(v[2], 9);
    ret[3] = zero_or_signed_extent(v[3], 9);
  }
  return ret;
}

template <typename RetT, bool NeedSat, bool NeedAdd, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<RetT> && sizeof(AT) == 4 && sizeof(BT) == 4 &&
        sizeof(RetT) == 4,
    RetT>
extend_vbinary2(AT a, BT b, RetT c, BinaryOperation binary_op) {
  sycl::vec<int32_t, 2> extend_a = extractAndExtend2(a);
  sycl::vec<int32_t, 2> extend_b = extractAndExtend2(b);
  sycl::vec<int32_t, 2> temp{binary_op(extend_a[0], extend_b[0]),
                             binary_op(extend_a[1], extend_b[1])};
  if constexpr (NeedSat) {
    int32_t min_val = 0, max_val = 0;
    if constexpr (std::is_signed_v<RetT>) {
      min_val = std::numeric_limits<int16_t>::min();
      max_val = std::numeric_limits<int16_t>::max();
    } else {
      min_val = std::numeric_limits<uint16_t>::min();
      max_val = std::numeric_limits<uint16_t>::max();
    }
    temp = dpct::clamp(temp, {min_val, min_val}, {max_val, max_val});
  }
  if constexpr (NeedAdd) {
    return temp[0] + temp[1] + c;
  }
  if constexpr (std::is_signed_v<RetT>) {
    return sycl::vec<int16_t, 2>{temp[0], temp[1]}.as<sycl::vec<RetT, 1>>();
  } else {
    return sycl::vec<uint16_t, 2>{temp[0], temp[1]}.as<sycl::vec<RetT, 1>>();
  }
}

template <typename RetT, bool NeedSat, bool NeedAdd, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr std::enable_if_t<
    std::is_integral_v<AT> && std::is_integral_v<BT> &&
        std::is_integral_v<RetT> && sizeof(AT) == 4 && sizeof(BT) == 4 &&
        sizeof(RetT) == 4,
    RetT>
extend_vbinary4(AT a, BT b, RetT c, BinaryOperation binary_op) {
  sycl::vec<int16_t, 4> extend_a = extractAndExtend4(a);
  sycl::vec<int16_t, 4> extend_b = extractAndExtend4(b);
  sycl::vec<int16_t, 4> temp{
      binary_op(extend_a[0], extend_b[0]), binary_op(extend_a[1], extend_b[1]),
      binary_op(extend_a[2], extend_b[2]), binary_op(extend_a[3], extend_b[3])};
  if constexpr (NeedSat) {
    int16_t min_val = 0, max_val = 0;
    if constexpr (std::is_signed_v<RetT>) {
      min_val = std::numeric_limits<int8_t>::min();
      max_val = std::numeric_limits<int8_t>::max();
    } else {
      min_val = std::numeric_limits<uint8_t>::min();
      max_val = std::numeric_limits<uint8_t>::max();
    }
    temp = dpct::clamp(temp, {min_val, min_val, min_val, min_val},
                       {max_val, max_val, max_val, max_val});
  }
  if constexpr (NeedAdd) {
    return temp[0] + temp[1] + temp[2] + temp[3] + c;
  }
  if constexpr (std::is_signed_v<RetT>) {
    return sycl::vec<int8_t, 4>{temp[0], temp[1], temp[2], temp[3]}
        .as<sycl::vec<RetT, 1>>();
  } else {
    return sycl::vec<uint8_t, 4>{temp[0], temp[1], temp[2], temp[3]}
        .as<sycl::vec<RetT, 1>>();
  }
}

template <typename T1, typename T2>
using dot_product_acc_t =
    std::conditional_t<std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
                       uint32_t, int32_t>;

template <typename T> sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
  return sycl::vec<T, 1>(val)
      .template as<sycl::vec<
          std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>, 4>>()
      .template convert<T>();
}

template <typename T> sycl::vec<T, 2> extract_and_sign_or_zero_extend2(T val) {
  return sycl::vec<T, 1>(val)
      .template as<sycl::vec<
          std::conditional_t<std::is_signed_v<T>, int16_t, uint16_t>, 2>>()
      .template convert<T>();
}
} // namespace detail

/// Two-way dot product-accumulate. Calculate and return interger_vector2(
/// \param a) dot product interger_vector2(low16_bit( \param b))  + \param c
///
/// \tparam [in] T1 The type of first value.
/// \tparam [in] T2 The type of second value.
/// \param [in] a The first value.
/// \param [in] b The second value.
/// \param [in] c The third value. It has type uint32_t if both T1 and T1 are
/// uint32_t else has type int32_t.
/// \return Two-way 16-bit to 8-bit dot product which is accumulated in 32-bit
/// result.
template <typename T1, typename T2, typename T3>
inline auto dp2a_lo(T1 a, T2 b, T3 c) {
  detail::dot_product_acc_t<T1, T2> res = c;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  res = __dp2a_lo(a, b, c);
#else
  auto va = ::dpct::detail::extract_and_sign_or_zero_extend2(a);
  auto vb = ::dpct::detail::extract_and_sign_or_zero_extend4(b);
  res += va[0] * vb[0];
  res += va[1] * vb[1];
#endif
  return res;
}

/// Two-way dot product-accumulate. Calculate and return interger_vector2(
/// \param a) dot product interger_vector2(high_16bit( \param b)) + \param c
///
/// \tparam [in] T1 The type of first value.
/// \tparam [in] T2 The type of second value.
/// \param [in] a The first value.
/// \param [in] b The second value.
/// \param [in] c The third value. It has type uint32_t if both T1 and T1 are
/// uint32_t else has type int32_t.
/// \return Two-way 16-bit to 8-bit dot product which is accumulated in 32-bit
/// result.
template <typename T1, typename T2, typename T3>
inline auto dp2a_hi(T1 a, T2 b, T3 c) {
  detail::dot_product_acc_t<T1, T2> res = c;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  res = __dp2a_hi(a, b, c);
#else
  auto va = ::dpct::detail::extract_and_sign_or_zero_extend2(a);
  auto vb = ::dpct::detail::extract_and_sign_or_zero_extend4(b);
  res += va[0] * vb[2];
  res += va[1] * vb[3];
#endif
  return res;
}

/// Four-way byte dot product-accumulate. Calculate and return interger_vector4(
/// \param a) dot product interger_vector4( \param b)  + \param c
///
/// \tparam [in] T1 The type of first value.
/// \tparam [in] T2 The type of second value.
/// \param [in] a The first value.
/// \param [in] b The second value.
/// \param [in] c The third value. It has type uint32_t if both T1 and T1 are
/// uint32_t else has type int32_t.
/// \return Four-way byte dot product which is accumulated in 32-bit result.
template <typename T1, typename T2, typename T3>
inline auto dp4a(T1 a, T2 b, T3 c) {
  detail::dot_product_acc_t<T1, T2> res = c;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  res = __dp4a(a, b, c);
#else
  auto va = ::dpct::detail::extract_and_sign_or_zero_extend4(a);
  auto vb = ::dpct::detail::extract_and_sign_or_zero_extend4(b);
  res += va[0] * vb[0];
  res += va[1] * vb[1];
  res += va[2] * vb[2];
  res += va[3] * vb[3];
#endif
  return res;
}

/// Extend \p a and \p b to 33 bit and add them.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, std::plus());
}

/// Extend Inputs to 33 bit, add \p a, \p b, then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend addition of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, std::plus(), second_op);
}

/// Extend \p a and \p b to 33 bit and add them with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, std::plus());
}

/// Extend Inputs to 33 bit, add \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend addition of \p a, \p b with saturation and \p second_op
/// with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, std::plus(), second_op);
}

/// Extend \p a and \p b to 33 bit and minus them.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, std::minus());
}

/// Extend Inputs to 33 bit, minus \p a, \p b, then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend subtraction of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, std::minus(), second_op);
}

/// Extend \p a and \p b to 33 bit and minus them with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, std::minus());
}

/// Extend Inputs to 33 bit, minus \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend subtraction of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, std::minus(), second_op);
}

/// Extend \p a and \p b to 33 bit and do abs_diff.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, abs_diff());
}

/// Extend Inputs to 33 bit, abs_diff \p a, \p b, then do \p second_op with \p
/// c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend abs_diff of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, abs_diff(), second_op);
}

/// Extend \p a and \p b to 33 bit and do abs_diff with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, abs_diff());
}

/// Extend Inputs to 33 bit, abs_diff \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend abs_diff of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff_sat(AT a, BT b, CT c,
                                         BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, abs_diff(), second_op);
}

/// Extend \p a and \p b to 33 bit and return smaller one.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The smaller one of the two extended values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, minimum());
}

/// Extend Inputs to 33 bit, find the smaller one in \p a, \p b, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The smaller one of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, minimum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return smaller one with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The smaller one of the two extended values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, minimum());
}

/// Extend Inputs to 33 bit, find the smaller one in \p a, \p b with saturation,
/// then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The smaller one of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, minimum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return bigger one.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The bigger one of the two extended values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, maximum());
}

/// Extend Inputs to 33 bit, find the bigger one in \p a, \p b, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The bigger one of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, maximum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return bigger one with saturation.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The bigger one of the two extended values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, maximum());
}

/// Extend Inputs to 33 bit, find the bigger one in \p a, \p b with saturation,
/// then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \tparam [in] CT The type of the third value, can only be 32 bit integer
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The bigger one of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, maximum(), second_op);
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a
/// 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized addition of the two
/// values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, std::plus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 2 elements vector type and extend each element to 17 bit. Then add each
/// half of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized subtraction of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, std::minus());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized abs_diff of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b with saturation, with each
/// value treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, abs_diff());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized minimum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, minimum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, false>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 2
/// elements vector type and extend each element to 17 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, false, true>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b with saturation, with each value
/// treated as a 2 elements vector type and extend each element to 17 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax2_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary2<RetT, true, false>(a, b, c, maximum());
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a
/// 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized addition of the two
/// values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, std::plus());
}

/// Compute vectorized addition of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vadd4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, std::plus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b, with each value treated as
/// a 4 elements vector type and extend each element to 9 bit. Then add each
/// half of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized subtraction of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, std::minus());
}

/// Compute vectorized subtraction of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vsub4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, std::minus());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized abs_diff of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, abs_diff());
}

/// Compute vectorized abs_diff of \p a and \p b with saturation, with each
/// value treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vabsdiff4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, abs_diff());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized minimum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, minimum());
}

/// Compute vectorized minimum of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized minimum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmin4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, minimum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, false>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b, with each value treated as a 4
/// elements vector type and extend each element to 9 bit. Then add each half
/// of the result and add with \p c.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The addition of each half of extend vectorized maximum of the
/// two values and the third value
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4_add(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, false, true>(a, b, c, maximum());
}

/// Compute vectorized maximum of \p a and \p b with saturation, with each value
/// treated as a 4 elements vector type and extend each element to 9 bit.
/// \tparam [in] RetT The type of the return value, can only be 32 bit integer
/// \tparam [in] AT The type of the first value, can only be 32 bit integer
/// \tparam [in] BT The type of the second value, can only be 32 bit integer
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns The extend vectorized maximum of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_vmax4_sat(AT a, BT b, RetT c) {
  return detail::extend_vbinary4<RetT, true, false>(a, b, c, maximum());
}
} // namespace dpct

#endif // __DPCT_MATH_HPP__
