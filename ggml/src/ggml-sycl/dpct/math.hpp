//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 **************************************************************************/

#pragma once

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

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

} // namespace detail

template <typename T> sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
    return sycl::vec<T, 1>(val)
        .template as<sycl::vec<
            std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>, 4>>()
        .template convert<T>();
}

template <typename T1, typename T2>
using dot_product_acc_t =
    std::conditional_t<std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
                       uint32_t, int32_t>;

template <typename T1, typename T2, typename T3>
inline auto dp4a(T1 a, T2 b, T3 c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                     \
    defined(__SYCL_CUDA_ARCH__) && __SYCL_CUDA_ARCH__ >= 610
    dot_product_acc_t<T1, T2> res;
    if constexpr (std::is_same_v<dot_product_acc_t<T1, T2>, uint32_t>) {
        asm volatile("dp4a.u32.u32 %0, %1, %2, %3;"
                     : "=r"(res)
                     : "r"(a), "r"(b), "r"(c));
    } else {
        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                     : "=r"(res)
                     : "r"(a), "r"(b), "r"(c));
    }
    return res;
#else
    dot_product_acc_t<T1, T2> res = c;
    auto va = extract_and_sign_or_zero_extend4(a);
    auto vb = extract_and_sign_or_zero_extend4(b);
    res += va[0] * vb[0];
    res += va[1] * vb[1];
    res += va[2] * vb[2];
    res += va[3] * vb[3];
    return res;
#endif
}

struct sub_sat {
    template <typename T> auto operator()(const T x, const T y) const {
        return sycl::sub_sat(x, y);
    }
};

template <typename S, typename T> inline T vectorized_min(T a, T b) {
    sycl::vec<T, 1> v0{a}, v1{b};
    auto v2 = v0.template as<S>();
    auto v3 = v1.template as<S>();
    auto v4 = sycl::min(v2, v3);
    v0 = v4.template as<sycl::vec<T, 1>>();
    return v0;
}

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

} // namespace dpct
