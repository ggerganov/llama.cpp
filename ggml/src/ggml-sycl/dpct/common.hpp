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

#include <complex>

#include <sycl/sycl.hpp>

#include "memory.hpp"

namespace dpct {
namespace detail {

template <typename T> struct DataType { using T2 = T; };

template <typename T> struct DataType<sycl::vec<T, 2>> {
    using T2 = std::complex<T>;
};

template <typename T>
inline typename DataType<T>::T2 get_value(const T *s, sycl::queue &q) {
    using Ty = typename DataType<T>::T2;
    Ty s_h;
    if (get_pointer_attribute(q, s) == pointer_access_attribute::device_only) {
        detail::dpct_memcpy(q, (void *)&s_h, (const void *)s, sizeof(T),
                            device_to_host)
            .wait();
    } else {
        s_h = *reinterpret_cast<const Ty *>(s);
    }
    return s_h;
}

} // namespace detail

template <typename T> inline auto get_value(const T *s, sycl::queue &q) {
    return detail::get_value(s, q);
}

} // namespace dpct
