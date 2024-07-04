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

#include <sycl/sycl.hpp>

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

template <typename T>
T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                           unsigned int logical_sub_group_size = 32) {
    unsigned int id = g.get_local_linear_id();
    unsigned int start_index =
        id / logical_sub_group_size * logical_sub_group_size;
    unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
    return sycl::select_from_group(g, x,
                                   target_offset < logical_sub_group_size
                                       ? start_index + target_offset
                                       : id);
}

using err0 = detail::generic_error_type<struct err0_tag, int>;
using err1 = detail::generic_error_type<struct err1_tag, int>;

} // namespace dpct
