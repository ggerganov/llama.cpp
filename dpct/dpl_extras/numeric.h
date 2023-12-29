//==---- numeric.h --------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_NUMERIC_H__
#define __DPCT_NUMERIC_H__

namespace dpct {

template <typename Policy, typename InputIt1, typename InputIt2, typename T>
T inner_product(Policy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init) {
  return std::transform_reduce(std::forward<Policy>(policy), first1, last1,
                               first2, init);
}

template <typename Policy, typename InputIt1, typename InputIt2, typename T,
          typename BinaryOperation1, typename BinaryOperation2>
T inner_product(Policy &&policy, InputIt1 first1, InputIt1 last1,
                InputIt2 first2, T init, BinaryOperation1 op1,
                BinaryOperation2 op2) {
  return std::transform_reduce(std::forward<Policy>(policy), first1, last1,
                               first2, init, op1, op2);
}

} // end namespace dpct

#endif
