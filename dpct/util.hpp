//==---- util.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_UTIL_HPP__
#define __DPCT_UTIL_HPP__

#include <sycl/sycl.hpp>
#include <complex>
#include <type_traits>
#include <cassert>
#include <cstdint>

// TODO: Remove these function definitions once they exist in the DPC++ compiler
#if defined(__SYCL_DEVICE_ONLY__) && defined(__INTEL_LLVM_COMPILER)
template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffle(__spv::Scope::Flag, T, unsigned) noexcept;

template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffleDown(__spv::Scope::Flag, T, unsigned) noexcept;

template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffleUp(__spv::Scope::Flag, T, unsigned) noexcept;
#endif

namespace dpct {

namespace detail {

template <typename tag, typename T> class generic_error_type {
public:
  generic_error_type() = default;
  generic_error_type(T value) : value{value} {}
  operator T() const { return value; }

private:
  T value;
};

} // namespace detail

using err0 = detail::generic_error_type<struct err0_tag, int>;
using err1 = detail::generic_error_type<struct err1_tag, int>;

template <int... Ints> struct integer_sequence {};
template <int Size, int... Ints>
struct make_index_sequence
    : public make_index_sequence<Size - 1, Size - 1, Ints...> {};
template <int... Ints>
struct make_index_sequence<0, Ints...> : public integer_sequence<Ints...> {};

template <typename T> struct DataType { using T2 = T; };
template <typename T> struct DataType<sycl::vec<T, 2>> {
  using T2 = std::complex<T>;
};

inline void matrix_mem_copy(void *to_ptr, const void *from_ptr, int to_ld,
                            int from_ld, int rows, int cols, int elem_size,
                            memcpy_direction direction = automatic,
                            sycl::queue &queue = dpct::get_default_queue(),
                            bool async = false) {
  if (to_ptr == from_ptr && to_ld == from_ld) {
    return;
  }

  if (to_ld == from_ld) {
    size_t copy_size = elem_size * ((cols - 1) * (size_t)to_ld + rows);
    if (async)
      detail::dpct_memcpy(queue, (void *)to_ptr, (void *)from_ptr,
                          copy_size, direction);
    else
      detail::dpct_memcpy(queue, (void *)to_ptr, (void *)from_ptr,
                          copy_size, direction).wait();
  } else {
    if (async)
      detail::dpct_memcpy(queue, to_ptr, from_ptr, elem_size * to_ld,
                          elem_size * from_ld, elem_size * rows, cols,
                          direction);
    else
      sycl::event::wait(detail::dpct_memcpy(
          queue, to_ptr, from_ptr, elem_size * to_ld, elem_size * from_ld,
          elem_size * rows, cols, direction));
  }
}

/// Copy matrix data. The default leading dimension is column.
/// \param [out] to_ptr A pointer points to the destination location.
/// \param [in] from_ptr A pointer points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] direction The direction of the data copy.
/// \param [in] queue The queue where the routine should be executed.
/// \param [in] async If this argument is true, the return of the function
/// does NOT guarantee the copy is completed.
template <typename T>
inline void matrix_mem_copy(T *to_ptr, const T *from_ptr, int to_ld,
                            int from_ld, int rows, int cols,
                            memcpy_direction direction = automatic,
                            sycl::queue &queue = dpct::get_default_queue(),
                            bool async = false) {
  using Ty = typename DataType<T>::T2;
  matrix_mem_copy((void *)to_ptr, (void *)from_ptr, to_ld, from_ld, rows, cols,
                  sizeof(Ty), direction, queue, async);
}

/// Cast the high or low 32 bits of a double to an integer.
/// \param [in] d The double value.
/// \param [in] use_high32 Cast the high 32 bits of the double if true;
/// otherwise cast the low 32 bits.
inline int cast_double_to_int(double d, bool use_high32 = true) {
  sycl::vec<double, 1> v0{d};
  auto v1 = v0.as<sycl::int2>();
  if (use_high32)
    return v1[1];
  return v1[0];
}

/// Combine two integers, the first as the high 32 bits and the second
/// as the low 32 bits, into a double.
/// \param [in] high32 The integer as the high 32 bits
/// \param [in] low32 The integer as the low 32 bits
inline double cast_ints_to_double(int high32, int low32) {
  sycl::int2 v0{low32, high32};
  auto v1 = v0.as<sycl::vec<double, 1>>();
  return v1;
}

/// Reverse the bit order of an unsigned integer
/// \param [in] a Input unsigned integer value
/// \returns Value of a with the bit order reversed
template <typename T> inline T reverse_bits(T a) {
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                "unsigned integer required");
  if (!a)
    return 0;
  T mask = 0;
  size_t count = 4 * sizeof(T);
  mask = ~mask >> count;
  while (count) {
    a = ((a & mask) << count) | ((a & ~mask) >> count);
    count = count >> 1;
    mask = mask ^ (mask << count);
  }
  return a;
}

/// \param [in] a The first value contains 4 bytes
/// \param [in] b The second value contains 4 bytes
/// \param [in] s The selector value, only lower 16bit used
/// \returns the permutation result of 4 bytes selected in the way
/// specified by \p s from \p a and \p b
inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s) {
  unsigned int ret;
  ret =
      ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
  return ret;
}

/// Find position of first least significant set bit in an integer.
/// ffs(0) returns 0.
///
/// \param [in] a Input integer value
/// \returns The position
template <typename T> inline int ffs(T a) {
  static_assert(std::is_integral<T>::value, "integer required");
  return (sycl::ctz(a) + 1) % (sizeof(T) * 8 + 1);
}

/// select_from_sub_group allows work-items to obtain a copy of a value held by
/// any other work-item in the sub_group. The input sub_group will be divided
/// into several logical sub_groups with id range [0, \p logical_sub_group_size
/// - 1]. Each work-item in logical sub_group gets value from another work-item
/// whose id is \p remote_local_id. If \p remote_local_id is outside the
/// logical sub_group id range, \p remote_local_id will modulo with \p
/// logical_sub_group_size. The \p logical_sub_group_size must be a power of 2
/// and not exceed input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] remote_local_id Input source work item id
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T select_from_sub_group(sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  return sycl::select_from_group(
      g, x, start_index + remote_local_id % logical_sub_group_size);
}

/// shift_sub_group_left move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the left. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is caller's id adds \p delta. If calculated id is outside the logical
/// sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = sycl::shift_group_left(g, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
}

/// shift_sub_group_right move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the right. The input sub_group will be divided into
/// several logical_sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical_sub_group gets value from another work-item whose
/// id is caller's id subtracts \p delta. If calculated id is outside the
/// logical sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_right(sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = sycl::shift_group_right(g, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
}

/// permute_sub_group_by_xor permutes values by exchanging values held by pairs
/// of work-items identified by computing the bitwise exclusive OR of the
/// work-item id and some fixed mask. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is bitwise exclusive OR of the caller's id and \p mask. If calculated id
/// is outside the logical sub_group id range, the work-item will get value from
/// itself. The \p logical_sub_group_size must be a power of 2 and not exceed
/// input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] mask Input mask
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
  return sycl::select_from_group(g, x,
                                 target_offset < logical_sub_group_size
                                     ? start_index + target_offset
                                     : id);
}

/// The function match_any_over_sub_group conducts a comparison of values
/// across work-items within a sub-group. match_any_over_sub_group return a mask
/// in which some bits are set to 1, indicating that the \p value provided by
/// the work-item represented by these bits are equal. The n-th bit of mask
/// representing the work-item with id n. The parameter \p member_mask
/// indicating the work-items participating the call.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] member_mask Input mask
/// \param [in] value Input value
/// \returns The result
template <typename T>
unsigned int match_any_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value) {
  static_assert(std::is_arithmetic_v<T>, "Value type must be arithmetic type.");                    
  if (!member_mask) {
    return 0;
  }
  unsigned int id = g.get_local_linear_id();
  unsigned int flag = 0, result = 0, reduce_result = 0;
  unsigned int bit_index = 0x1 << id;
  bool is_participate = member_mask & bit_index;
  T broadcast_value = 0;
  bool matched = false;
  while (flag != member_mask) {
    broadcast_value =
        sycl::select_from_group(g, value, sycl::ctz((~flag & member_mask)));
    reduce_result = sycl::reduce_over_group(
        g, is_participate ? (broadcast_value == value ? bit_index : 0) : 0,
        sycl::plus<>());
    flag |= reduce_result;
    matched = reduce_result & bit_index;
    result = matched * reduce_result + (1 - matched) * result;
  }
  return result;
}

/// The function match_all_over_sub_group conducts a comparison of values
/// across work-items within a sub-group. match_all_over_sub_group return \p
/// member_mask and predicate \p pred will be set to 1 if all \p value that
/// provided by each work-item in \p member_mask are equal, otherwise return 0
/// and the predicate \p pred will be set to 0. The n-th bit of \p member_mask
/// representing the work-item with id n. The parameter \p member_mask
/// indicating the work-items participating the call.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] member_mask Input mask
/// \param [in] value Input value
/// \param [out] pred Output predicate
/// \returns The result
template <typename T>
unsigned int match_all_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value, int *pred) {
  static_assert(std::is_arithmetic_v<T>, "Value type must be arithmetic type."); 
  if (!member_mask) {
    return 0;
  }
  unsigned int id = g.get_local_linear_id();
  unsigned int bit_index = 0x1 << id;
  bool is_participate = member_mask & bit_index;
  T broadcast_value = sycl::select_from_group(g, value, sycl::ctz(member_mask));
  unsigned int reduce_result = sycl::reduce_over_group(
      g,
      (member_mask & bit_index) ? (broadcast_value == value ? bit_index : 0)
                                : 0,
      sycl::plus<>());
  bool all_equal = (reduce_result == member_mask);
  *pred = is_participate & all_equal;
  return all_equal * member_mask;
}

namespace experimental {
#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
#define SHFL_SYNC(RES, MASK, VAL, SHFL_PARAM, C, SHUFFLE_INSTR)                \
  if constexpr (std::is_same_v<T, double>) {                                   \
    int x_a, x_b;                                                              \
    asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "d"(VAL));              \
    auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);   \
    auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);   \
    asm("mov.b64 %0,{%1,%2};" : "=d"(RES) : "r"(tmp_a), "r"(tmp_b));           \
  } else if constexpr (std::is_same_v<T, long> ||                              \
                       std::is_same_v<T, unsigned long>) {                     \
    int x_a, x_b;                                                              \
    asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "l"(VAL));              \
    auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);   \
    auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);   \
    asm("mov.b64 %0,{%1,%2};" : "=l"(RES) : "r"(tmp_a), "r"(tmp_b));           \
  } else if constexpr (std::is_same_v<T, sycl::half>) {                        \
    short tmp_b16;                                                             \
    asm("mov.b16 %0,%1;" : "=h"(tmp_b16) : "h"(VAL));                          \
    auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                           \
        MASK, static_cast<int>(tmp_b16), SHFL_PARAM, C);                       \
    asm("mov.b16 %0,%1;" : "=h"(RES) : "h"(static_cast<short>(tmp_b32)));      \
  } else if constexpr (std::is_same_v<T, float>) {                             \
    auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                           \
        MASK, __nvvm_bitcast_f2i(VAL), SHFL_PARAM, C);                         \
    RES = __nvvm_bitcast_i2f(tmp_b32);                                         \
  } else {                                                                     \
    RES = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, VAL, SHFL_PARAM, C);          \
  }
#endif
/// Masked version of select_from_sub_group, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] remote_local_id Input source work item id
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T select_from_sub_group(unsigned int member_mask,
                        sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  unsigned logical_remote_id =
      start_index + remote_local_id % logical_sub_group_size;
  return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, remote_local_id, cVal, idx_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)remote_local_id;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of select_from_sub_group not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/// Masked version of shift_sub_group_left, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(unsigned int member_mask,
                       sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = __spirv_GroupNonUniformShuffleDown(__spv::Scope::Subgroup, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, delta, cVal, down_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)delta;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of select_from_sub_group not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/// Masked version of shift_sub_group_right, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_right(unsigned int member_mask,
                        sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = __spirv_GroupNonUniformShuffleUp(__spv::Scope::Subgroup, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8);
  SHFL_SYNC(result, member_mask, x, delta, cVal, up_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)delta;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of select_from_sub_group not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/// Masked version of permute_sub_group_by_xor, which execute masked sub-group
/// operation. The parameter member_mask indicating the work-items participating
/// the call. Whether the n-th bit is set to 1 representing whether the
/// work-item with id n is participating the call. All work-items named in
/// member_mask must be executed with the same member_mask, or the result is
/// undefined.
/// \tparam T Input value type
/// \param [in] member_mask Input mask
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] mask Input mask
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T permute_sub_group_by_xor(unsigned int member_mask,
                           sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
  unsigned logical_remote_id = (target_offset < logical_sub_group_size) ? start_index + target_offset : id;
  return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, mask, cVal, bfly_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)mask;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of select_from_sub_group not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}
#if defined(__NVPTX__)
#undef SHFL_SYNC
#endif
} // namespace experimental

/// Computes the multiplication of two complex numbers.
/// \tparam T Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> cmul(sycl::vec<T, 2> x, sycl::vec<T, 2> y) {
  std::complex<T> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 * t2;
  return sycl::vec<T, 2>(t1.real(), t1.imag());
}

/// Computes the division of two complex numbers.
/// \tparam T Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> cdiv(sycl::vec<T, 2> x, sycl::vec<T, 2> y) {
  std::complex<T> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 / t2;
  return sycl::vec<T, 2>(t1.real(), t1.imag());
}

/// Computes the magnitude of a complex number.
/// \tparam T Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename T>
T cabs(sycl::vec<T, 2> x) {
  std::complex<T> t(x[0], x[1]);
  return std::abs(t);
}

/// Computes the complex conjugate of a complex number.
/// \tparam T Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> conj(sycl::vec<T, 2> x) {
  std::complex<T> t(x[0], x[1]);
  t = std::conj(t);
  return sycl::vec<T, 2>(t.real(), t.imag());
}

inline int get_sycl_language_version() {
#ifdef SYCL_LANGUAGE_VERSION
  return SYCL_LANGUAGE_VERSION;
#else
  return 202000;
#endif
}

namespace experimental {
/// Synchronize work items from all work groups within a SYCL kernel.
/// \param [in] item:  Represents a work group.
/// \param [in] counter: An atomic object defined on a device memory which can
/// be accessed by work items in all work groups. The initial value of the
/// counter should be zero.
/// Note: Please make sure that all the work items of all work groups within
/// a SYCL kernel can be scheduled actively at the same time on a device.
template <int dimensions = 3>
inline void
nd_range_barrier(const sycl::nd_item<dimensions> &item,
                 sycl::atomic_ref<
                     unsigned int, sycl::memory_order::seq_cst,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter) {

  static_assert(dimensions == 3, "dimensions must be 3.");

  unsigned int num_groups = item.get_group_range(2) * item.get_group_range(1) *
                            item.get_group_range(0);

  item.barrier();

  if (item.get_local_linear_id() == 0) {
    unsigned int inc = 1;
    unsigned int old_arrive = 0;
    bool is_group0 =
        (item.get_group(2) + item.get_group(1) + item.get_group(0) == 0);
    if (is_group0) {
      inc = 0x80000000 - (num_groups - 1);
    }

    old_arrive = counter.fetch_add(inc);
    // Synchronize all the work groups
    while (((old_arrive ^ counter.load()) & 0x80000000) == 0)
      ;
  }

  item.barrier();
}

/// Synchronize work items from all work groups within a SYCL kernel.
/// \param [in] item:  Represents a work group.
/// \param [in] counter: An atomic object defined on a device memory which can
/// be accessed by work items in all work groups. The initial value of the
/// counter should be zero.
/// Note: Please make sure that all the work items of all work groups within
/// a SYCL kernel can be scheduled actively at the same time on a device.
template <>
inline void
nd_range_barrier(const sycl::nd_item<1> &item,
                 sycl::atomic_ref<
                     unsigned int, sycl::memory_order::seq_cst,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter) {
  unsigned int num_groups = item.get_group_range(0);

  item.barrier();

  if (item.get_local_linear_id() == 0) {
    unsigned int inc = 1;
    unsigned int old_arrive = 0;
    bool is_group0 = (item.get_group(0) == 0);
    if (is_group0) {
      inc = 0x80000000 - (num_groups - 1);
    }

    old_arrive = counter.fetch_add(inc);
    // Synchronize all the work groups
    while (((old_arrive ^ counter.load()) & 0x80000000) == 0)
      ;
  }

  item.barrier();
}

/// The logical-group is a logical collection of some work-items within a
/// work-group.
/// Note: Please make sure that the logical-group size is a power of 2 in the
/// range [1, current_sub_group_size].
template <int dimensions = 3> class logical_group {
  sycl::nd_item<dimensions> _item;
  sycl::group<dimensions> _g;
  uint32_t _logical_group_size;
  uint32_t _group_linear_range_in_parent;

public:
  /// Dividing \p parent_group into several logical-groups.
  /// \param [in] item Current work-item.
  /// \param [in] parent_group The group to be divided.
  /// \param [in] size The logical-group size.
  logical_group(sycl::nd_item<dimensions> item,
                sycl::group<dimensions> parent_group, uint32_t size)
      : _item(item), _g(parent_group), _logical_group_size(size) {
    _group_linear_range_in_parent =
        (_g.get_local_linear_range() - 1) / _logical_group_size + 1;
  }
  logical_group(sycl::nd_item<dimensions> item)
      : _item(item), _g(item.get_group()) {}
  /// Returns the index of the work-item within the logical-group.
  uint32_t get_local_linear_id() const {
    return _item.get_local_linear_id() % _logical_group_size;
  }
  /// Returns the index of the logical-group in the parent group.
  uint32_t get_group_linear_id() const {
    return _item.get_local_linear_id() / _logical_group_size;
  }
  /// Returns the number of work-items in the logical-group.
  uint32_t get_local_linear_range() const {
    if (_g.get_local_linear_range() % _logical_group_size == 0) {
      return _logical_group_size;
    }
    uint32_t last_item_group_id =
        _g.get_local_linear_range() / _logical_group_size;
    uint32_t first_of_last_group = last_item_group_id * _logical_group_size;
    if (_item.get_local_linear_id() >= first_of_last_group) {
      return _g.get_local_linear_range() - first_of_last_group;
    } else {
      return _logical_group_size;
    }
  }
  /// Returns the number of logical-group in the parent group.
  uint32_t get_group_linear_range() const {
    return _group_linear_range_in_parent;
  }
};

// The original source of the functions calculate_max_active_wg_per_xecore and
// calculate_max_potential_wg were under the license below:
//
// Copyright (C) Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
/// This function is used for occupancy calculation, it computes the max active
/// work-group number per Xe-Core. Ref to
/// https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/GPU-Occupancy-Calculator
/// \param [out] num_wg Active work-group number.
/// \param [in] wg_size Work-group size.
/// \param [in] slm_size Share local memory size.
/// \param [in] sg_size Sub-group size.
/// \param [in] used_barrier Whether barrier is used.
/// \param [in] used_large_grf Whether large General Register File is used.
/// \return If no error, returns 0.
/// If \p wg_size exceeds the max work-group size, the max work-group size will
/// be used instead of \p wg_size and returns -1.
inline int calculate_max_active_wg_per_xecore(int *num_wg, int wg_size,
                                              int slm_size = 0,
                                              int sg_size = 32,
                                              bool used_barrier = false,
                                              bool used_large_grf = false) {
  int ret = 0;
  const int slm_size_per_xe_core = 64 * 1024;
  const int max_barrier_registers = 32;
  dpct::device_ext &dev = dpct::get_current_device();

  size_t max_wg_size = dev.get_info<sycl::info::device::max_work_group_size>();
  if (wg_size > max_wg_size) {
    wg_size = max_wg_size;
    ret = -1;
  }

  int num_threads_ss = 56;
  int max_num_wg = 56;
  if (dev.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice) &&
      dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
    auto eu_count =
        dev.get_info<sycl::info::device::ext_intel_gpu_eu_count_per_subslice>();
    auto threads_count =
        dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    num_threads_ss = eu_count * threads_count;
    max_num_wg = eu_count * threads_count;
  }

  if (used_barrier) {
    max_num_wg = max_barrier_registers;
  }

  // Calculate num_wg_slm
  int num_wg_slm = 0;
  if (slm_size == 0) {
    num_wg_slm = max_num_wg;
  } else {
    num_wg_slm = std::floor((float)slm_size_per_xe_core / slm_size);
  }

  // Calculate num_wg_threads
  if (used_large_grf)
    num_threads_ss = num_threads_ss / 2;
  int num_threads = std::ceil((float)wg_size / sg_size);
  int num_wg_threads = std::floor((float)num_threads_ss / num_threads);

  // Calculate num_wg
  *num_wg = std::min(num_wg_slm, num_wg_threads);
  *num_wg = std::min(*num_wg, max_num_wg);
  return ret;
}

/// This function is used for occupancy calculation, it computes the work-group
/// number and the work-group size which achieves the maximum occupancy of the
/// device potentially. Ref to
/// https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/GPU-Occupancy-Calculator
/// \param [out] num_wg Work-group number.
/// \param [out] wg_size Work-group size.
/// \param [in] max_ws_size_for_device_code The maximum working work-group size
/// for current device code logic. Zero means no limitation.
/// \param [in] slm_size Share local memory size.
/// \param [in] sg_size Sub-group size.
/// \param [in] used_barrier Whether barrier is used.
/// \param [in] used_large_grf Whether large General Register File is used.
/// \return Returns 0.
inline int calculate_max_potential_wg(int *num_wg, int *wg_size,
                                      int max_ws_size_for_device_code,
                                      int slm_size = 0, int sg_size = 32,
                                      bool used_barrier = false,
                                      bool used_large_grf = false) {
  sycl::device &dev = dpct::get_current_device();
  size_t max_wg_size = dev.get_info<sycl::info::device::max_work_group_size>();
  if (max_ws_size_for_device_code == 0 ||
      max_ws_size_for_device_code >= max_wg_size)
    *wg_size = (int)max_wg_size;
  else
    *wg_size = max_ws_size_for_device_code;
  calculate_max_active_wg_per_xecore(num_wg, *wg_size, slm_size, sg_size,
                                     used_barrier, used_large_grf);
  std::uint32_t num_ss = 1;
  if (dev.has(sycl::aspect::ext_intel_gpu_slices) &&
      dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
    num_ss =
        dev.get_info<sycl::ext::intel::info::device::gpu_slices>() *
        dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  }
  num_wg[0] = num_ss * num_wg[0];
  return 0;
}

/// Supported group type during migration.
enum class group_type { work_group, sub_group, logical_group, root_group };

/// The group_base will dispatch the function call to the specific interface
/// based on the group type.
template <int dimensions = 3> class group_base {
public:
  group_base(sycl::nd_item<dimensions> item)
      : nd_item(item), logical_group(item) {}
  ~group_base() {}
  /// Returns the number of work-items in the group.
  size_t get_local_linear_range() {
    switch (type) {
    case group_type::work_group:
      return nd_item.get_group().get_local_linear_range();
    case group_type::sub_group:
      return nd_item.get_sub_group().get_local_linear_range();
    case group_type::logical_group:
      return logical_group.get_local_linear_range();
    default:
      return -1; // Unkonwn group type
    }
  }
  /// Returns the index of the work-item within the group.
  size_t get_local_linear_id() {
    switch (type) {
    case group_type::work_group:
      return nd_item.get_group().get_local_linear_id();
    case group_type::sub_group:
      return nd_item.get_sub_group().get_local_linear_id();
    case group_type::logical_group:
      return logical_group.get_local_linear_id();
    default:
      return -1; // Unkonwn group type
    }
  }
  /// Wait for all the elements within the group to complete their execution
  /// before proceeding.
  void barrier() {
    switch (type) {
    case group_type::work_group:
      sycl::group_barrier(nd_item.get_group());
      break;
    case group_type::sub_group:
    case group_type::logical_group:
      sycl::group_barrier(nd_item.get_sub_group());
      break;
    default:
      break;
    }
  }

protected:
  logical_group<dimensions> logical_group;
  sycl::nd_item<dimensions> nd_item;
  group_type type;
};

/// The group class is a container type that can storage supported group_type.
template <typename T, int dimensions = 3>
class group : public group_base<dimensions> {
  using group_base<dimensions>::type;
  using group_base<dimensions>::logical_group;

public:
  group(T g, sycl::nd_item<dimensions> item) : group_base<dimensions>(item) {
    if constexpr (std::is_same_v<T, sycl::sub_group>) {
      type = group_type::sub_group;
    } else if constexpr (std::is_same_v<T, sycl::group<dimensions>>) {
      type = group_type::work_group;
    } else if constexpr (std::is_same_v<T, dpct::experimental::logical_group<
                                               dimensions>>) {
      logical_group = g;
      type = group_type::logical_group;
    }
  }
};
} // namespace experimental

/// If x <= 2, then return a pointer to the deafult queue;
/// otherwise, return x reinterpreted as a dpct::queue_ptr.
inline queue_ptr int_as_queue_ptr(uintptr_t x) {
  return x <= 2 ?
  &get_default_queue()
  : reinterpret_cast<queue_ptr>(x);
}

template <int n_nondefault_params, int n_default_params, typename T>
class args_selector;

/// args_selector is a helper class for extracting arguments from an
/// array of pointers to arguments or buffer of arguments to pass to a
/// kernel function.
///
/// \param R(Ts...) The type of the kernel
/// \param n_nondefault_params The number of nondefault parameters of the kernel
/// (excluding parameters that like sycl::nd_item, etc.)
/// \param n_default_params The number of default parameters of the kernel
///
/// Example usage:
/// With the following kernel:
///   void foo(sycl::float2 *x, int n, sycl::nd_item<3> item_ct1, float f=.1) {}
/// and with the declaration:
///   args_selector<2, 1, decltype(foo)> selector(kernelParams, extra);
/// we have:
///   selector.get<0>() returns a reference to sycl::float*,
///   selector.get<1>() returns a reference to int,
///   selector.get<2>() returns a reference to float
template <int n_nondefault_params, int n_default_params,
   typename R, typename... Ts>
class args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
private:
  void **kernel_params;
  char *args_buffer;

  template <int i>
  static constexpr int account_for_default_params() {
    constexpr int n_total_params = sizeof...(Ts);
    if constexpr (i >= n_nondefault_params) {
      return n_total_params - n_default_params + (i - n_nondefault_params);
    } else {
      return i;
    }
  }    

public:
  /// Get the type of the ith argument of R(Ts...)
  /// \param [in] i Index of parameter to get
  /// \returns Type of ith parameter
  template <int i>
  using arg_type = std::tuple_element_t<account_for_default_params<i>(),
					  std::tuple<Ts...>>;
private:
  template <int i>
  static constexpr int get_offset() {
    if constexpr (i == 0) {
      // we can assume args_buffer is properly aligned to the
      // first argument
      return 0;
    } else {
      constexpr int prev_off = get_offset<i-1>();
      constexpr int prev_past_end = prev_off + sizeof(arg_type<i-1>);
      using T = arg_type<i>;
      // is the past-the-end of the i-1st element properly aligned
      // with the ith element's alignment?
      if constexpr (prev_past_end % alignof(T) == 0) {
	return prev_past_end;
      }
      // otherwise bump prev_past_end to match alignment
      else {
	return prev_past_end + (alignof(T) - (prev_past_end % alignof(T)));
      }
    }
  }

  static char *get_args_buffer(void **extra) {
    if (!extra)
      return nullptr;
    for (; (std::size_t) *extra != 0; ++extra) {
      if ((std::size_t) *extra == 1) {
	return static_cast<char*>(*(extra+1));
      }
    }
    return nullptr;
  }
    
public:
  /// If kernel_params is nonnull, then args_selector will
  /// extract arguments from kernel_params. Otherwise, it
  /// will extract them from extra.
  /// \param [in] kernel_params Array of pointers to arguments
  /// a or null pointer.
  /// \param [in] extra Array containing pointer to argument buffer.
  args_selector(void **kernel_params, void **extra)
    : kernel_params(kernel_params),
      args_buffer(get_args_buffer(extra))
  {}

  /// Get a reference to the ith argument extracted from kernel_params
  /// or extra.
  /// \param [in] i Index of argument to get
  /// \returns Reference to the ith argument
  template <int i>    
  arg_type<i> &get() {
    if (kernel_params) {
      return *static_cast<arg_type<i>*>(kernel_params[i]);
    } else {
      return *reinterpret_cast<arg_type<i>*>(args_buffer + get_offset<i>());
    }
  }
};

#ifdef _WIN32
#define DPCT_EXPORT __declspec(dllexport)
#else
#define DPCT_EXPORT
#endif

} // namespace dpct

#endif // __DPCT_UTIL_HPP__
