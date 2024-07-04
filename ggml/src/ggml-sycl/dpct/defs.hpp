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

#define DPCT_COMPATIBILITY_TEMP (900)

#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#else
#define __dpct_noinline__ __attribute__((noinline))
#endif

namespace dpct {
enum error_code { success = 0, default_error = 999 };
}

#define CHECK_TRY_ERROR(expr)                                                  \
    [&]() {                                                                    \
        try {                                                                  \
            expr;                                                              \
            return dpct::success;                                              \
        } catch (std::exception const &e) {                                    \
            std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                      << ", line:" << __LINE__ << ", func:" << __func__        \
                      << std::endl;                                            \
            return dpct::default_error;                                        \
        }                                                                      \
    }()
