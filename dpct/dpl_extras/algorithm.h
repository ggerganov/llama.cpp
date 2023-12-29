//==---- algorithm.h ------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_ALGORITHM_H__
#define __DPCT_ALGORITHM_H__

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include "functional.h"
#include "iterators.h"
#include "vector.h"

namespace dpct {

template <typename Policy, typename Iter1, typename Iter2, typename Pred,
          typename T>
void replace_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask, Pred p,
                const T &new_value) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  std::transform(
      std::forward<Policy>(policy), first, last, mask, first,
      internal::replace_if_fun<typename std::iterator_traits<Iter1>::value_type,
                               Pred>(p, new_value));
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Pred, typename T>
Iter3 replace_copy_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask,
                      Iter3 result, Pred p, const T &new_value) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  return std::transform(
      std::forward<Policy>(policy), first, last, mask, result,
      internal::replace_if_fun<typename std::iterator_traits<Iter3>::value_type,
                               Pred>(p, new_value));
}

template <typename Policy, typename Iter1, typename Iter2, typename Pred>
internal::enable_if_hetero_execution_policy<Policy, Iter1>
remove_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using oneapi::dpl::make_zip_iterator;
  using policy_type = typename std::decay<Policy>::type;
  using internal::__buffer;
  using ValueType = typename std::iterator_traits<Iter1>::value_type;

  __buffer<ValueType> _tmp(std::distance(first, last));

  auto end = std::copy_if(
      policy, make_zip_iterator(first, mask),
      make_zip_iterator(last, mask + std::distance(first, last)),
      make_zip_iterator(_tmp.get(), oneapi::dpl::discard_iterator()),
      internal::negate_predicate_key_fun<Pred>(p));
  return std::copy(std::forward<Policy>(policy), _tmp.get(),
                   std::get<0>(end.base()), first);
}

template <typename Policy, typename Iter1, typename Iter2, typename Pred>
typename std::enable_if<!internal::is_hetero_execution_policy<
                            typename std::decay<Policy>::type>::value,
                        Iter1>::type
remove_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using oneapi::dpl::make_zip_iterator;
  using policy_type = typename std::decay<Policy>::type;
  using ValueType = typename std::iterator_traits<Iter1>::value_type;

  std::vector<ValueType> _tmp(std::distance(first, last));

  auto end = std::copy_if(
      policy, make_zip_iterator(first, mask),
      make_zip_iterator(last, mask + std::distance(first, last)),
      make_zip_iterator(_tmp.begin(), oneapi::dpl::discard_iterator()),
      internal::negate_predicate_key_fun<Pred>(p));
  return std::copy(policy, _tmp.begin(), std::get<0>(end.base()), first);
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Pred>
Iter3 remove_copy_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask,
                     Iter3 result, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using oneapi::dpl::make_zip_iterator;
  auto ret_val = std::remove_copy_if(
      std::forward<Policy>(policy), make_zip_iterator(first, mask),
      make_zip_iterator(last, mask + std::distance(first, last)),
      make_zip_iterator(result, oneapi::dpl::discard_iterator()),
      internal::predicate_key_fun<Pred>(p));
  return std::get<0>(ret_val.base());
}

template <class Policy, class Iter1, class Iter2, class BinaryPred>
std::pair<Iter1, Iter2> unique(Policy &&policy, Iter1 keys_first,
                               Iter1 keys_last, Iter2 values_first,
                               BinaryPred binary_pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::unique(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first, values_first),
      oneapi::dpl::make_zip_iterator(
          keys_last, values_first + std::distance(keys_first, keys_last)),
      internal::compare_key_fun<BinaryPred>(binary_pred));
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_first, values_first), ret_val);
  return std::make_pair(keys_first + n1, values_first + n1);
}

template <class Policy, class Iter1, class Iter2>
std::pair<Iter1, Iter2> unique(Policy &&policy, Iter1 keys_first,
                               Iter1 keys_last, Iter2 values_first) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using T = typename std::iterator_traits<Iter1>::value_type;
  return unique(std::forward<Policy>(policy), keys_first, keys_last,
                values_first, std::equal_to<T>());
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class BinaryPred>
std::pair<Iter3, Iter4> unique_copy(Policy &&policy, Iter1 keys_first,
                                    Iter1 keys_last, Iter2 values_first,
                                    Iter3 keys_result, Iter4 values_result,
                                    BinaryPred binary_pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::unique_copy(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first, values_first),
      oneapi::dpl::make_zip_iterator(
          keys_last, values_first + std::distance(keys_first, keys_last)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::unique_fun<BinaryPred>(binary_pred));
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4>
std::pair<Iter3, Iter4> unique_copy(Policy &&policy, Iter1 keys_first,
                                    Iter1 keys_last, Iter2 values_first,
                                    Iter3 keys_result, Iter4 values_result) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using T = typename std::iterator_traits<Iter1>::value_type;
  auto comp = std::equal_to<T>();
  return unique_copy(std::forward<Policy>(policy), keys_first, keys_last,
                     values_first, keys_result, values_result, comp);
}

template <typename Policy, typename Iter, typename Pred>
Iter partition_point(Policy &&policy, Iter first, Iter last, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  if (std::is_partitioned(policy, first, last, p))
    return std::find_if_not(std::forward<Policy>(policy), first, last, p);
  else
    return first;
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Pred>
Iter3 copy_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask,
              Iter3 result, Pred pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::copy_if(
      std::forward<Policy>(policy), oneapi::dpl::make_zip_iterator(first, mask),
      oneapi::dpl::make_zip_iterator(last, mask + std::distance(first, last)),
      oneapi::dpl::make_zip_iterator(result, oneapi::dpl::discard_iterator()),
      internal::predicate_key_fun<Pred>(pred));
  return std::get<0>(ret_val.base());
}

template <class Policy, class Iter1, class Iter2, class UnaryOperation,
          class Pred>
Iter2 transform_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 result,
                   UnaryOperation unary_op, Pred pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using T = typename std::iterator_traits<Iter1>::value_type;
  const auto n = std::distance(first, last);
  std::for_each(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(first, result),
      oneapi::dpl::make_zip_iterator(first, result) + n,
      internal::transform_if_fun<T, Pred, UnaryOperation>(pred, unary_op));
  return result + n;
}

template <class Policy, class Iter1, class Iter2, class Iter3,
          class UnaryOperation, class Pred>
Iter3 transform_if(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask,
                   Iter3 result, UnaryOperation unary_op, Pred pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using T = typename std::iterator_traits<Iter1>::value_type;
  using Ref1 = typename std::iterator_traits<Iter1>::reference;
  using Ref2 = typename std::iterator_traits<Iter2>::reference;
  const auto n = std::distance(first, last);
  std::for_each(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(first, mask, result),
      oneapi::dpl::make_zip_iterator(first, mask, result) + n,
      internal::transform_if_unary_zip_mask_fun<T, Pred, UnaryOperation>(
          pred, unary_op));
  return result + n;
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class BinaryOperation, class Pred>
Iter4 transform_if(Policy &&policy, Iter1 first1, Iter1 last1, Iter2 first2,
                   Iter3 mask, Iter4 result, BinaryOperation binary_op,
                   Pred pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  const auto n = std::distance(first1, last1);
  using ZipIterator =
      typename oneapi::dpl::zip_iterator<Iter1, Iter2, Iter3, Iter4>;
  using T = typename std::iterator_traits<ZipIterator>::value_type;
  std::for_each(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(first1, first2, mask, result),
      oneapi::dpl::make_zip_iterator(last1, first2 + n, mask + n, result + n),
      internal::transform_if_zip_mask_fun<T, Pred, BinaryOperation>(pred,
                                                                    binary_op));
  return result + n;
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename OutputIter>
void scatter(Policy &&policy, InputIter1 first, InputIter1 last, InputIter2 map,
             OutputIter result) {
  static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter2>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<OutputIter>::iterator_category,
              std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  oneapi::dpl::copy(policy, first, last,
                    oneapi::dpl::make_permutation_iterator(result, map));
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename OutputIter>
OutputIter gather(Policy &&policy, InputIter1 map_first, InputIter1 map_last,
                  InputIter2 input_first, OutputIter result) {
  static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter2>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<OutputIter>::iterator_category,
              std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto perm_begin =
      oneapi::dpl::make_permutation_iterator(input_first, map_first);
  const int n = ::std::distance(map_first, map_last);

  return oneapi::dpl::copy(policy, perm_begin, perm_begin + n, result);
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename InputIter3, typename OutputIter, typename Predicate>
void scatter_if(Policy &&policy, InputIter1 first, InputIter1 last,
                InputIter2 map, InputIter3 mask, OutputIter result,
                Predicate pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter2>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter3>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<OutputIter>::iterator_category,
              std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  transform_if(
      ::std::forward<Policy>(policy), first, last, mask,
      oneapi::dpl::make_permutation_iterator(result, map),
      [=](auto &&v) { return v; }, [=](auto &&m) { return pred(m); });
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename InputIter3, typename OutputIter>
void scatter_if(Policy &&policy, InputIter1 first, InputIter1 last,
                InputIter2 map, InputIter3 mask, OutputIter result) {
  scatter_if(::std::forward<Policy>(policy), first, last, map, mask, result,
             internal::no_op_fun());
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename InputIter3, typename OutputIter, typename Predicate>
OutputIter gather_if(Policy &&policy, InputIter1 map_first, InputIter1 map_last,
                     InputIter2 mask, InputIter3 input_first, OutputIter result,
                     Predicate pred) {
  static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter2>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter3>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<OutputIter>::iterator_category,
              std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto perm_begin =
      oneapi::dpl::make_permutation_iterator(input_first, map_first);
  const int n = std::distance(map_first, map_last);

  return transform_if(
      ::std::forward<Policy>(policy), perm_begin, perm_begin + n, mask, result,
      [=](auto &&v) { return v; }, [=](auto &&m) { return pred(m); });
}

template <typename Policy, typename InputIter1, typename InputIter2,
          typename InputIter3, typename OutputIter>
OutputIter gather_if(Policy &&policy, InputIter1 map_first, InputIter1 map_last,
                     InputIter2 mask, InputIter3 input_first,
                     OutputIter result) {
  return gather_if(::std::forward<Policy>(policy), map_first, map_last, mask,
                   input_first, result, internal::no_op_fun());
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Iter5, typename Iter6>
std::pair<Iter5, Iter6>
merge(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1, Iter2 keys_first2,
      Iter2 keys_last2, Iter3 values_first1, Iter4 values_first2,
      Iter5 keys_result, Iter6 values_result) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto n1 = std::distance(keys_first1, keys_last1);
  auto n2 = std::distance(keys_first2, keys_last2);
  std::merge(std::forward<Policy>(policy),
             oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
             oneapi::dpl::make_zip_iterator(keys_last1, values_first1 + n1),
             oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
             oneapi::dpl::make_zip_iterator(keys_last2, values_first2 + n2),
             oneapi::dpl::make_zip_iterator(keys_result, values_result),
             internal::compare_key_fun<>());
  return std::make_pair(keys_result + n1 + n2, values_result + n1 + n2);
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Iter5, typename Iter6, typename Comp>
std::pair<Iter5, Iter6>
merge(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1, Iter2 keys_first2,
      Iter2 keys_last2, Iter3 values_first1, Iter4 values_first2,
      Iter5 keys_result, Iter6 values_result, Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto n1 = std::distance(keys_first1, keys_last1);
  auto n2 = std::distance(keys_first2, keys_last2);
  std::merge(std::forward<Policy>(policy),
             oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
             oneapi::dpl::make_zip_iterator(keys_last1, values_first1 + n1),
             oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
             oneapi::dpl::make_zip_iterator(keys_last2, values_first2 + n2),
             oneapi::dpl::make_zip_iterator(keys_result, values_result),
             internal::compare_key_fun<Comp>(comp));
  return std::make_pair(keys_result + n1 + n2, values_result + n1 + n2);
}

template <class Policy, class Iter, class T>
void iota(Policy &&policy, Iter first, Iter last, T init, T step) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using DiffSize = typename std::iterator_traits<Iter>::difference_type;
  std::transform(
      std::forward<Policy>(policy), oneapi::dpl::counting_iterator<DiffSize>(0),
      oneapi::dpl::counting_iterator<DiffSize>(std::distance(first, last)),
      first, internal::sequence_fun<T>(init, step));
}

template <class Policy, class Iter, class T>
void iota(Policy &&policy, Iter first, Iter last, T init) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  iota(std::forward<Policy>(policy), first, last, init, T(1));
}

template <class Policy, class Iter>
void iota(Policy &&policy, Iter first, Iter last) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using DiffSize = typename std::iterator_traits<Iter>::difference_type;
  iota(std::forward<Policy>(policy), first, last, DiffSize(0), DiffSize(1));
}

template <class Policy, class Iter1, class Iter2, class Comp>
void sort(Policy &&policy, Iter1 keys_first, Iter1 keys_last,
          Iter2 values_first, Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto first = oneapi::dpl::make_zip_iterator(keys_first, values_first);
  auto last = first + std::distance(keys_first, keys_last);
  std::sort(std::forward<Policy>(policy), first, last,
            internal::compare_key_fun<Comp>(comp));
}

template <class Policy, class Iter1, class Iter2>
void sort(Policy &&policy, Iter1 keys_first, Iter1 keys_last,
          Iter2 values_first) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  sort(std::forward<Policy>(policy), keys_first, keys_last, values_first,
       internal::__less());
}

template <class Policy, class Iter1, class Iter2, class Comp>
void stable_sort(Policy &&policy, Iter1 keys_first, Iter1 keys_last,
                 Iter2 values_first, Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  std::stable_sort(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first, values_first),
      oneapi::dpl::make_zip_iterator(
          keys_last, values_first + std::distance(keys_first, keys_last)),
      internal::compare_key_fun<Comp>(comp));
}

template <class Policy, class Iter1, class Iter2>
void stable_sort(Policy &&policy, Iter1 keys_first, Iter1 keys_last,
                 Iter2 values_first) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  stable_sort(std::forward<Policy>(policy), keys_first, keys_last, values_first,
              internal::__less());
}

template <class Policy, class Iter, class Operator>
void for_each_index(Policy &&policy, Iter first, Iter last, Operator unary_op) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter>::iterator_category,
                   std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  using DiffSize = typename std::iterator_traits<Iter>::difference_type;
  std::transform(
      std::forward<Policy>(policy), oneapi::dpl::counting_iterator<DiffSize>(0),
      oneapi::dpl::counting_iterator<DiffSize>(std::distance(first, last)),
      first, unary_op);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5>
std::pair<Iter4, Iter5>
set_intersection(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
                 Iter2 keys_first2, Iter2 keys_last2, Iter3 values_first1,
                 Iter4 keys_result, Iter5 values_result) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_intersection(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2,
                                     oneapi::dpl::discard_iterator()),
      oneapi::dpl::make_zip_iterator(keys_last2,
                                     oneapi::dpl::discard_iterator()),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Comp>
std::pair<Iter4, Iter5>
set_intersection(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
                 Iter2 keys_first2, Iter2 keys_last2, Iter3 values_first1,
                 Iter4 keys_result, Iter5 values_result, Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_intersection(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2,
                                     oneapi::dpl::discard_iterator()),
      oneapi::dpl::make_zip_iterator(keys_last2,
                                     oneapi::dpl::discard_iterator()),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Comp>(comp));
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Iter6>
std::pair<Iter5, Iter6>
set_symmetric_difference(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
                         Iter2 keys_first2, Iter2 keys_last2,
                         Iter3 values_first1, Iter4 values_first2,
                         Iter5 keys_result, Iter6 values_result) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_symmetric_difference(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
      oneapi::dpl::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Iter6, class Comp>
std::pair<Iter5, Iter6>
set_symmetric_difference(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
                         Iter2 keys_first2, Iter2 keys_last2,
                         Iter3 values_first1, Iter4 values_first2,
                         Iter5 keys_result, Iter6 values_result, Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_symmetric_difference(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
      oneapi::dpl::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Comp>(comp));
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Iter6>
std::pair<Iter5, Iter6>
set_difference(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
               Iter2 keys_first2, Iter2 keys_last2, Iter3 values_first1,
               Iter4 values_first2, Iter5 keys_result, Iter6 values_result) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_difference(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
      oneapi::dpl::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Iter6, class Comp>
std::pair<Iter5, Iter6> set_difference(Policy &&policy, Iter1 keys_first1,
                                       Iter1 keys_last1, Iter2 keys_first2,
                                       Iter2 keys_last2, Iter3 values_first1,
                                       Iter4 values_first2, Iter5 keys_result,
                                       Iter6 values_result, Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_difference(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
      oneapi::dpl::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Comp>(comp));
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Iter6>
internal::enable_if_execution_policy<Policy, std::pair<Iter5, Iter6>>
set_union(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
          Iter2 keys_first2, Iter2 keys_last2, Iter3 values_first1,
          Iter4 values_first2, Iter5 keys_result, Iter6 values_result) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_union(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
      oneapi::dpl::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<>());
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <class Policy, class Iter1, class Iter2, class Iter3, class Iter4,
          class Iter5, class Iter6, class Comp>
internal::enable_if_execution_policy<Policy, std::pair<Iter5, Iter6>>
set_union(Policy &&policy, Iter1 keys_first1, Iter1 keys_last1,
          Iter2 keys_first2, Iter2 keys_last2, Iter3 values_first1,
          Iter4 values_first2, Iter5 keys_result, Iter6 values_result,
          Comp comp) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter5>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter6>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::set_union(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(keys_first1, values_first1),
      oneapi::dpl::make_zip_iterator(
          keys_last1, values_first1 + std::distance(keys_first1, keys_last1)),
      oneapi::dpl::make_zip_iterator(keys_first2, values_first2),
      oneapi::dpl::make_zip_iterator(
          keys_last2, values_first2 + std::distance(keys_first2, keys_last2)),
      oneapi::dpl::make_zip_iterator(keys_result, values_result),
      internal::compare_key_fun<Comp>(comp));
  auto n1 = std::distance(
      oneapi::dpl::make_zip_iterator(keys_result, values_result), ret_val);
  return std::make_pair(keys_result + n1, values_result + n1);
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Pred>
internal::enable_if_execution_policy<Policy, std::pair<Iter3, Iter4>>
stable_partition_copy(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask,
                      Iter3 out_true, Iter4 out_false, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  auto ret_val = std::partition_copy(
      std::forward<Policy>(policy), oneapi::dpl::make_zip_iterator(first, mask),
      oneapi::dpl::make_zip_iterator(last, mask + std::distance(first, last)),
      oneapi::dpl::make_zip_iterator(out_true, oneapi::dpl::discard_iterator()),
      oneapi::dpl::make_zip_iterator(out_false,
                                     oneapi::dpl::discard_iterator()),
      internal::predicate_key_fun<Pred>(p));
  return std::make_pair(std::get<0>(ret_val.first.base()),
                        std::get<0>(ret_val.second.base()));
}

template <typename Policy, typename Iter1, typename Iter3, typename Iter4,
          typename Pred>
internal::enable_if_execution_policy<Policy, std::pair<Iter3, Iter4>>
stable_partition_copy(Policy &&policy, Iter1 first, Iter1 last, Iter3 out_true,
                      Iter4 out_false, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  return std::partition_copy(std::forward<Policy>(policy), first, last,
                             out_true, out_false, p);
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Pred>
internal::enable_if_execution_policy<Policy, std::pair<Iter3, Iter4>>
partition_copy(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask,
               Iter3 out_true, Iter4 out_false, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter3>::iterator_category,
                       std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter4>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  return stable_partition_copy(std::forward<Policy>(policy), first, last, mask,
                               out_true, out_false, p);
}

template <typename Policy, typename Iter1, typename Iter2, typename Pred>
internal::enable_if_hetero_execution_policy<Policy, Iter1>
stable_partition(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  typedef typename std::decay<Policy>::type policy_type;
  internal::__buffer<typename std::iterator_traits<Iter1>::value_type> _tmp(
      std::distance(first, last));

  std::copy(policy, mask, mask + std::distance(first, last), _tmp.get());

  auto ret_val =
      std::stable_partition(std::forward<Policy>(policy),
                            oneapi::dpl::make_zip_iterator(first, _tmp.get()),
                            oneapi::dpl::make_zip_iterator(
                                last, _tmp.get() + std::distance(first, last)),
                            internal::predicate_key_fun<Pred>(p));
  return std::get<0>(ret_val.base());
}

template <typename Policy, typename Iter1, typename Iter2, typename Pred>
typename std::enable_if<!internal::is_hetero_execution_policy<
                            typename std::decay<Policy>::type>::value,
                        Iter1>::type
stable_partition(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  typedef typename std::decay<Policy>::type policy_type;
  std::vector<typename std::iterator_traits<Iter1>::value_type> _tmp(
      std::distance(first, last));

  std::copy(policy, mask, mask + std::distance(first, last), _tmp.begin());

  auto ret_val = std::stable_partition(
      std::forward<Policy>(policy),
      oneapi::dpl::make_zip_iterator(first, _tmp.begin()),
      oneapi::dpl::make_zip_iterator(last,
                                     _tmp.begin() + std::distance(first, last)),
      internal::predicate_key_fun<Pred>(p));
  return std::get<0>(ret_val.base());
}

template <typename Policy, typename Iter1, typename Iter2, typename Pred>
internal::enable_if_execution_policy<Policy, Iter1>
partition(Policy &&policy, Iter1 first, Iter1 last, Iter2 mask, Pred p) {
  static_assert(
      std::is_same<typename std::iterator_traits<Iter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<typename std::iterator_traits<Iter2>::iterator_category,
                       std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  return stable_partition(std::forward<Policy>(policy), first, last, mask, p);
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4>
inline ::std::enable_if_t<dpct::internal::is_iterator<Iter1>::value &&
                          dpct::internal::is_iterator<Iter2>::value &&
                          dpct::internal::is_iterator<Iter3>::value &&
                          dpct::internal::is_iterator<Iter4>::value>
sort_pairs(Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter3 values_in,
           Iter4 values_out, ::std::int64_t n, bool descending = false,
           int begin_bit = 0,
           int end_bit =
               sizeof(typename ::std::iterator_traits<Iter1>::value_type) * 8);

template <typename Policy, typename Iter1, typename Iter2>
inline ::std::enable_if_t<dpct::internal::is_iterator<Iter1>::value &&
                          dpct::internal::is_iterator<Iter2>::value>
sort_keys(Policy &&policy, Iter1 keys_in, Iter2 keys_out, ::std::int64_t n,
          bool descending = false, int begin_bit = 0,
          int end_bit =
              sizeof(typename ::std::iterator_traits<Iter1>::value_type) * 8);

namespace internal {

// Transforms key to a specific bit range and sorts the transformed key
template <typename Policy, typename Iter1, typename Iter2,
          typename TransformedKeyT>
inline void transform_and_sort(Policy &&policy, Iter1 keys_in, Iter2 keys_out,
                               ::std::int64_t n, bool descending, int begin_bit,
                               int end_bit) {
  using key_t_value_t = typename std::iterator_traits<Iter1>::value_type;
  auto trans_key =
      translate_key<key_t_value_t, TransformedKeyT>(begin_bit, end_bit);

  // Use of the comparison operator that is not simply std::greater() or
  // std::less() will result in
  //  not using radix sort which will cost some performance.  However, this is
  //  necessary to provide the transformation of the key to the bitrange
  //  desired.
  auto partial_sort_with_comp = [&](const auto &comp) {
    return oneapi::dpl::partial_sort_copy(
        std::forward<Policy>(policy), keys_in, keys_in + n, keys_out,
        keys_out + n, [=](const auto a, const auto b) {
          return comp(trans_key(a), trans_key(b));
        });
  };
  if (descending)
    partial_sort_with_comp(::std::greater<TransformedKeyT>());
  else
    partial_sort_with_comp(::std::less<TransformedKeyT>());
}

template <typename Policy, typename Iter1, typename Iter2>
inline void sort_only(Policy &&policy, Iter1 keys_in, Iter2 keys_out,
                      ::std::int64_t n, bool descending) {
  using key_t_value_t = typename ::std::iterator_traits<Iter1>::value_type;

  if constexpr (::std::is_floating_point<key_t_value_t>::value) {
    if (descending) {
      // Comparison operator that is not std::greater() ensures stability of
      // -0.0 and 0.0
      // at the cost of some performance because radix sort will not be used.
      auto comp_descending = [=](const auto a, const auto b) { return a > b; };

      oneapi::dpl::partial_sort_copy(::std::forward<Policy>(policy), keys_in,
                                     keys_in + n, keys_out, keys_out + n,
                                     comp_descending);
    } else {
      // Comparison operator that is not std::less() ensures stability of -0.0
      // and 0.0
      // at the cost of some performance because radix sort will not be used.
      auto comp_ascending = [=](const auto a, const auto b) { return a < b; };

      oneapi::dpl::partial_sort_copy(::std::forward<Policy>(policy), keys_in,
                                     keys_in + n, keys_out, keys_out + n,
                                     comp_ascending);
    }
  } else {
    if (descending) {
      oneapi::dpl::partial_sort_copy(::std::forward<Policy>(policy), keys_in,
                                     keys_in + n, keys_out, keys_out + n,
                                     ::std::greater<key_t_value_t>());
    } else {

      oneapi::dpl::partial_sort_copy(::std::forward<Policy>(policy), keys_in,
                                     keys_in + n, keys_out, keys_out + n);
    }
  }
}

// Transforms key from a pair to a specific bit range and sorts the pairs by the
// transformed key
template <typename Policy, typename Iter1, typename Iter2,
          typename TransformedKeyT, typename Iter3, typename Iter4>
inline void
transform_and_sort_pairs(Policy &&policy, Iter1 keys_in, Iter2 keys_out,
                         Iter3 values_in, Iter4 values_out, ::std::int64_t n,
                         bool descending, int begin_bit, int end_bit) {
  using key_t_value_t = typename std::iterator_traits<Iter1>::value_type;
  auto zip_input = oneapi::dpl::zip_iterator(keys_in, values_in);
  auto zip_output = oneapi::dpl::zip_iterator(keys_out, values_out);
  auto trans_key =
      translate_key<key_t_value_t, TransformedKeyT>(begin_bit, end_bit);

  // Use of the comparison operator that is not simply std::greater() or
  // std::less() will result in
  //  not using radix sort which will cost some performance.  However, this is
  //  necessary to provide the transformation of the key to the bitrange desired
  //  and also to select the key from the zipped pair.
  auto load_val = [=](const auto a) { return trans_key(std::get<0>(a)); };

  auto partial_sort_with_comp = [&](const auto &comp) {
    return oneapi::dpl::partial_sort_copy(
        std::forward<Policy>(policy), zip_input, zip_input + n, zip_output,
        zip_output + n, [=](const auto a, const auto b) {
          return comp(load_val(a), load_val(b));
        });
  };
  if (descending)
    partial_sort_with_comp(::std::greater<key_t_value_t>());
  else
    partial_sort_with_comp(::std::less<key_t_value_t>());
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4>
inline void sort_only_pairs(Policy &&policy, Iter1 keys_in, Iter2 keys_out,
                            Iter3 values_in, Iter4 values_out, ::std::int64_t n,
                            bool descending) {
  using key_t_value_t = typename ::std::iterator_traits<Iter1>::value_type;
  auto zip_input = oneapi::dpl::zip_iterator(keys_in, values_in);
  auto zip_output = oneapi::dpl::zip_iterator(keys_out, values_out);

  // Use of the comparison operator that is not simply std::greater() or
  // std::less() will result in
  //  not using radix sort which will cost some performance.  However, this is
  //  necessary to select the key from the zipped pair.
  auto load_val = [=](const auto a) { return std::get<0>(a); };

  auto partial_sort_with_comp = [&](const auto &comp) {
    return oneapi::dpl::partial_sort_copy(
        std::forward<Policy>(policy), zip_input, zip_input + n, zip_output,
        zip_output + n, [=](const auto a, const auto b) {
          return comp(load_val(a), load_val(b));
        });
  };
  if (descending)
    partial_sort_with_comp(::std::greater<key_t_value_t>());
  else
    partial_sort_with_comp(::std::less<key_t_value_t>());
}

// overload for Iter2 != std::nullptr_t
template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4>
typename ::std::enable_if<!::std::is_null_pointer<Iter2>::value>::type
sort_pairs_impl(Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter3 values_in,
                Iter4 values_out, ::std::int64_t n, bool descending,
                int begin_bit, int end_bit) {
  using key_t_value_t = typename ::std::iterator_traits<Iter1>::value_type;

  int clipped_begin_bit = ::std::max(begin_bit, 0);
  int clipped_end_bit =
      ::std::min((::std::uint64_t)end_bit, sizeof(key_t_value_t) * 8);
  int num_bytes = (clipped_end_bit - clipped_begin_bit - 1) / 8 + 1;

  auto transform_and_sort_pairs_f = [&](auto x) {
    using T = typename ::std::decay_t<decltype(x)>;
    internal::transform_and_sort_pairs<decltype(policy), Iter1, Iter2, T, Iter3,
                                       Iter4>(
        ::std::forward<Policy>(policy), keys_in, keys_out, values_in,
        values_out, n, descending, clipped_begin_bit, clipped_end_bit);
  };

  if (clipped_end_bit - clipped_begin_bit == sizeof(key_t_value_t) * 8) {
    internal::sort_only_pairs(::std::forward<Policy>(policy), keys_in, keys_out,
                              values_in, values_out, n, descending);
  } else if (num_bytes == 1) {
    transform_and_sort_pairs_f.template operator()<uint8_t>(0);
  } else if (num_bytes == 2) {
    transform_and_sort_pairs_f.template operator()<uint16_t>(0);
  } else if (num_bytes <= 4) {
    transform_and_sort_pairs_f.template operator()<uint32_t>(0);
  } else // if (num_bytes <= 8)
  {
    transform_and_sort_pairs_f.template operator()<::std::uint64_t>(0);
  }
}

// overload for Iter2 == std::nullptr_t
template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4>
typename ::std::enable_if<::std::is_null_pointer<Iter2>::value>::type
sort_pairs_impl(Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter3 values_in,
                Iter4 values_out, ::std::int64_t n, bool descending,
                int begin_bit, int end_bit) {
  // create temporary keys_out to discard, memory footprint could be improved by
  // a specialized iterator with a single
  // unchanging dummy Iter1 element
  using key_t_value_t = typename std::iterator_traits<Iter1>::value_type;
  sycl::buffer<key_t_value_t, 1> temp_keys_out{sycl::range<1>(n)};
  internal::sort_pairs_impl(std::forward<Policy>(policy), keys_in,
                            oneapi::dpl::begin(temp_keys_out), values_in,
                            values_out, n, descending, begin_bit, end_bit);
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Iter5>
inline void segmented_sort_pairs_by_parallel_sorts(
    Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter4 values_in,
    Iter3 values_out, ::std::int64_t n, ::std::int64_t nsegments,
    Iter5 begin_offsets, Iter5 end_offsets, bool descending = false,
    int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  using offset_type = typename ::std::iterator_traits<Iter5>::value_type;
  ::std::vector<offset_type> host_accessible_offset_starts(nsegments);
  ::std::vector<offset_type> host_accessible_offset_ends(nsegments);
  // make offsets accessible on host
  ::std::copy(policy, begin_offsets, begin_offsets + nsegments,
              host_accessible_offset_starts.begin());
  ::std::copy(policy, end_offsets, end_offsets + nsegments,
              host_accessible_offset_ends.begin());

  for (::std::uint64_t i = 0; i < nsegments; i++) {
    ::std::uint64_t segment_begin = host_accessible_offset_starts[i];
    ::std::uint64_t segment_end =
        ::std::min(n, (::std::int64_t)host_accessible_offset_ends[i]);
    if (segment_begin < segment_end) {
      ::dpct::sort_pairs(
          policy, keys_in + segment_begin, keys_out + segment_begin,
          values_in + segment_begin, values_out + segment_begin,
          segment_end - segment_begin, descending, begin_bit, end_bit);
    }
  }
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline void segmented_sort_keys_by_parallel_sorts(
    Policy &&policy, Iter1 keys_in, Iter2 keys_out, ::std::int64_t n,
    ::std::int64_t nsegments, Iter3 begin_offsets, Iter3 end_offsets,
    bool descending = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  using offset_type = typename ::std::iterator_traits<Iter3>::value_type;
  ::std::vector<offset_type> host_accessible_offset_starts(nsegments);
  ::std::vector<offset_type> host_accessible_offset_ends(nsegments);
  // make offsets accessible on host
  ::std::copy(policy, begin_offsets, begin_offsets + nsegments,
              host_accessible_offset_starts.begin());
  ::std::copy(policy, end_offsets, end_offsets + nsegments,
              host_accessible_offset_ends.begin());

  for (::std::uint64_t i = 0; i < nsegments; i++) {
    ::std::uint64_t segment_begin = host_accessible_offset_starts[i];
    ::std::uint64_t segment_end =
        ::std::min(n, (::std::int64_t)host_accessible_offset_ends[i]);
    if (segment_begin < segment_end) {
      ::dpct::sort_keys(policy, keys_in + segment_begin,
                        keys_out + segment_begin, segment_end - segment_begin,
                        descending, begin_bit, end_bit);
    }
  }
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Iter5>
inline void segmented_sort_pairs_by_parallel_for_of_sorts(
    Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter3 values_in,
    Iter4 values_out, ::std::int64_t n, ::std::int64_t nsegments,
    Iter5 begin_offsets, Iter5 end_offsets, bool descending = false,
    int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  policy.queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nsegments, [=](sycl::id<1> i) {
      ::std::uint64_t segment_begin = begin_offsets[i];
      ::std::uint64_t segment_end =
          ::std::min(n, (::std::int64_t)end_offsets[i]);
      if (segment_begin == segment_end) {
        return;
      }
      ::dpct::sort_pairs(::std::execution::seq, keys_in + segment_begin,
                         keys_out + segment_begin, values_in + segment_begin,
                         values_out + segment_begin,
                         segment_end - segment_begin, descending, begin_bit,
                         end_bit);
    });
  });
  policy.queue().wait();
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline void segmented_sort_keys_by_parallel_for_of_sorts(
    Policy &&policy, Iter1 keys_in, Iter2 keys_out, ::std::int64_t n,
    ::std::int64_t nsegments, Iter3 begin_offsets, Iter3 end_offsets,
    bool descending = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  policy.queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nsegments, [=](sycl::id<1> i) {
      ::std::uint64_t segment_begin = begin_offsets[i];
      ::std::uint64_t segment_end =
          ::std::min(n, (::std::int64_t)end_offsets[i]);
      if (segment_begin == segment_end) {
        return;
      }
      ::dpct::sort_keys(::std::execution::seq, keys_in + segment_begin,
                        keys_out + segment_begin, segment_end - segment_begin,
                        descending, begin_bit, end_bit);
    });
  });
  policy.queue().wait();
}

template <typename Policy, typename OffsetIteratorT>
inline void mark_segments(Policy &&policy, OffsetIteratorT begin_offsets,
                          OffsetIteratorT end_offsets, ::std::int64_t n,
                          ::std::int64_t nsegments,
                          sycl::buffer<::std::size_t, 1> segments) {

  ::std::size_t work_group_size =
      policy.queue()
          .get_device()
          .template get_info<sycl::info::device::max_work_group_size>();

  auto sg_sizes = policy.queue()
                      .get_device()
                      .template get_info<sycl::info::device::sub_group_sizes>();
  ::std::size_t sub_group_size = sg_sizes.empty() ? 0 : sg_sizes.back();

  float avg_seg_size = (float)n / (float)nsegments;
  if (avg_seg_size > work_group_size) {
    // If average segment size is larger than workgroup, use workgroup to
    // coordinate to mark segments
    policy.queue()
        .submit([&](sycl::handler &h) {
          auto segments_acc = segments.get_access<sycl::access_mode::write>(h);
          h.parallel_for(work_group_size, ([=](sycl::id<1> id) {
                           for (::std::size_t seg = 0; seg < nsegments; seg++) {
                             ::std::size_t i = begin_offsets[seg];
                             ::std::size_t end = end_offsets[seg];
                             while (i + id < end) {
                               segments_acc[i + id] = seg;
                               i += work_group_size;
                             }
                           }
                         }));
        })
        .wait();
  } else if (sub_group_size > 0 && avg_seg_size > sub_group_size / 2) {
    // If average segment size is larger than half a subgroup, use subgroup to
    // coordinate to mark segments
    policy.queue()
        .submit([&](sycl::handler &h) {
          auto segments_acc = segments.get_access<sycl::access_mode::write>(h);
          h.parallel_for(
              sycl::nd_range<1>{work_group_size, work_group_size},
              ([=](sycl::nd_item<1> item) {
                auto sub_group = item.get_sub_group();
                ::std::size_t num_subgroups =
                    sub_group.get_group_range().size();
                ::std::size_t local_size = sub_group.get_local_range().size();

                ::std::size_t sub_group_id = sub_group.get_group_id();
                while (sub_group_id < nsegments) {
                  ::std::size_t subgroup_local_id = sub_group.get_local_id();
                  ::std::size_t i = begin_offsets[sub_group_id];
                  ::std::size_t end = end_offsets[sub_group_id];
                  while (i + subgroup_local_id < end) {
                    segments_acc[i + subgroup_local_id] = sub_group_id;
                    i += local_size;
                  }
                  sub_group_id += num_subgroups;
                }
              }));
        })
        .wait();
  } else {
    // If average segment size is small as compared to subgroup, use single
    // work item to mark each segment
    policy.queue()
        .submit([&](sycl::handler &h) {
          auto segments_acc = segments.get_access<sycl::access_mode::write>(h);
          h.parallel_for(nsegments, ([=](sycl::id<1> seg) {
                           for (::std::size_t i = begin_offsets[seg];
                                i < end_offsets[seg]; i++) {
                             segments_acc[i] = seg;
                           }
                         }));
        })
        .wait();
  }
}

// The dpl_histogram namespace contains a temporary preview of an upcoming
// oneDPL histogram API.  This namespace will be removed and replaced with
// corresponding calls to oneapi::dpl::histogram()
namespace dpl_histogram {

template <typename T1, typename T2>
constexpr inline auto __ceiling_div(const T1 &number, const T2 &divisor) {
  return (number - 1) / divisor + 1;
}

template <typename T1, bool _IsFloatingPoint>
struct __evenly_divided_binhash_impl {};

template <typename T>
struct __evenly_divided_binhash_impl<T, /* is_floating_point = */ true> {
  T __minimum;
  ::std::uint32_t __num_bins;
  T __scale;
  T __maximum;
  __evenly_divided_binhash_impl(const T &min, const T &max,
                                const ::std::uint32_t &num_bins)
      : __minimum(min), __maximum(max), __num_bins(num_bins),
        __scale(T(num_bins) / (max - min)) {}
  template <typename T2> std::uint32_t operator()(T2 &&value) const {
    return ::std::uint32_t((::std::forward<T2>(value) - __minimum) * __scale);
  }

  template <typename T2> bool is_valid(const T2 &value) const {
    return value >= __minimum && value < __maximum;
  }
};

// non floating point type
template <typename T>
struct __evenly_divided_binhash_impl<T, /* is_floating_point= */ false> {
  T __minimum;
  ::std::uint32_t __num_bins;
  T __range_size;
  __evenly_divided_binhash_impl(const T &min, const T &max,
                                const ::std::uint32_t &num_bins)
      : __minimum(min), __num_bins(num_bins), __range_size(max - min) {}
  template <typename T2>::std::uint32_t operator()(T2 &&value) const {
    return ::std::uint32_t(
        ((::std::uint64_t(::std::forward<T2>(value)) - __minimum) *
         ::std::uint64_t(__num_bins)) /
        __range_size);
  }

  template <typename T2> bool is_valid(const T2 &value) const {
    return value >= __minimum && value < __minimum + __range_size;
  }
};

template <typename T1>
using __evenly_divided_binhash =
    __evenly_divided_binhash_impl<T1, ::std::is_floating_point_v<T1>>;

template <typename Range> struct __custom_range_binhash {
  Range __boundaries;
  __custom_range_binhash(Range boundaries) : __boundaries(boundaries) {}

  template <typename T>::std::uint32_t operator()(T &&value) const {
    return (::std::upper_bound(__boundaries.begin(), __boundaries.end(),
                               ::std::forward<T>(value)) -
            __boundaries.begin()) -
           1;
  }

  template <typename T2> bool is_valid(const T2 &value) const {
    return value >= __boundaries[0] &&
           value < __boundaries[__boundaries.size() - 1];
  }
};

template <typename HistAccessor, typename OffsetT, typename Size>
inline void __clear_wglocal_histograms(const HistAccessor &local_histogram,
                                       const OffsetT &offset,
                                       const Size &num_bins,
                                       const sycl::nd_item<1> &self_item) {
  ::std::uint32_t gSize = self_item.get_local_range()[0];
  ::std::uint32_t self_lidx = self_item.get_local_id(0);
  ::std::uint8_t factor = __ceiling_div(num_bins, gSize);
  ::std::uint8_t k;
  _DPCT_PRAGMA_UNROLL
  for (k = 0; k < factor - 1; k++) {
    local_histogram[offset + gSize * k + self_lidx] = 0;
  }
  if (gSize * k + self_lidx < num_bins) {
    local_histogram[offset + gSize * k + self_lidx] = 0;
  }
  self_item.barrier(sycl::access::fence_space::local_space);
}

template <typename BinIdxType, typename Iter1, typename HistReg,
          typename BinFunc>
inline void __accum_local_register_iter(const Iter1 &in_acc,
                                        const ::std::size_t &index,
                                        HistReg *histogram, BinFunc func) {
  const auto &x = in_acc[index];
  if (func.is_valid(x)) {
    BinIdxType c = func(x);
    histogram[c]++;
  }
}

template <typename BinIdxType, sycl::access::address_space AddressSpace,
          typename Iter1, typename HistAccessor, typename OffsetT,
          typename BinFunc>
inline void __accum_local_atomics_iter(const Iter1 &in_acc,
                                       const ::std::size_t &index,
                                       const HistAccessor &wg_local_histogram,
                                       const OffsetT &offset, BinFunc func) {
  using __histo_value_type = typename HistAccessor::value_type;
  const auto &x = in_acc[index];
  if (func.is_valid(x)) {
    BinIdxType c = func(x);
    sycl::atomic_ref<__histo_value_type, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group, AddressSpace>
        local_bin(wg_local_histogram[offset + c]);
    local_bin++;
  }
}

template <typename BinType, typename HistAccessorIn, typename OffsetT,
          typename HistAccessorOut, typename Size>
inline void __reduce_out_histograms(const HistAccessorIn &in_histogram,
                                    const OffsetT &offset,
                                    const HistAccessorOut &out_histogram,
                                    const Size &num_bins,
                                    const sycl::nd_item<1> &self_item) {
  ::std::uint32_t gSize = self_item.get_local_range()[0];
  ::std::uint32_t self_lidx = self_item.get_local_id(0);
  ::std::uint8_t factor = __ceiling_div(num_bins, gSize);
  ::std::uint8_t k;

  _DPCT_PRAGMA_UNROLL
  for (k = 0; k < factor - 1; k++) {
    sycl::atomic_ref<BinType, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        global_bin(out_histogram[gSize * k + self_lidx]);
    global_bin += in_histogram[offset + gSize * k + self_lidx];
  }
  if (gSize * k + self_lidx < num_bins) {
    sycl::atomic_ref<BinType, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        global_bin(out_histogram[gSize * k + self_lidx]);
    global_bin += in_histogram[offset + gSize * k + self_lidx];
  }
}

template <::std::uint16_t ItersPerWorkItem, ::std::uint8_t BinsPerWorkItem,
          typename BinType, typename Policy, typename Range1, typename Range2,
          typename Size, typename IdxHashFunc, typename... Range3>
inline void __histogram_general_registers_local_reduction(
    Policy &&policy, ::std::uint16_t work_group_size, Range1 &&input,
    Range2 &&bins, const Size &num_bins, IdxHashFunc func,
    Range3 &&...opt_range) {
  const ::std::size_t N = input.size();
  using __local_histogram_type = ::std::uint32_t;
  using __private_histogram_type = ::std::uint16_t;

  ::std::size_t segments = __ceiling_div(N, work_group_size * ItersPerWorkItem);
  auto e = policy.queue().submit([&](auto &h) {
    // Temporary use of stable non-public API from oneDPL,  this function will
    // be replaced with oneDPL call in an upcoming release.
    oneapi::dpl::__ranges::__require_access(h, input, bins, opt_range...);
    sycl::local_accessor<__local_histogram_type, 1> local_histogram(
        sycl::range(num_bins), h);
    h.parallel_for(
        sycl::nd_range<1>(segments * work_group_size, work_group_size),
        [=](sycl::nd_item<1> __self_item) {
          using __bin_idx_type = ::std::uint8_t;
          const ::std::size_t __self_lidx = __self_item.get_local_id(0);
          const ::std::size_t __wgroup_idx = __self_item.get_group(0);
          const ::std::size_t __seg_start =
              work_group_size * ItersPerWorkItem * __wgroup_idx;

          __clear_wglocal_histograms(local_histogram, 0, num_bins, __self_item);
          __private_histogram_type histogram[BinsPerWorkItem];
          _DPCT_PRAGMA_UNROLL
          for (::std::uint8_t k = 0; k < BinsPerWorkItem; k++) {
            histogram[k] = 0;
          }

          if (__seg_start + work_group_size * ItersPerWorkItem < N) {
            _DPCT_PRAGMA_UNROLL
            for (::std::uint8_t idx = 0; idx < ItersPerWorkItem; idx++) {
              __accum_local_register_iter<__bin_idx_type>(
                  input, __seg_start + idx * work_group_size + __self_lidx,
                  histogram, func);
            }
          } else {
            _DPCT_PRAGMA_UNROLL
            for (::std::uint8_t idx = 0; idx < ItersPerWorkItem; idx++) {
              ::std::size_t __val_idx =
                  __seg_start + idx * work_group_size + __self_lidx;
              if (__val_idx < N) {
                __accum_local_register_iter<__bin_idx_type>(input, __val_idx,
                                                            histogram, func);
              }
            }
          }

          _DPCT_PRAGMA_UNROLL
          for (::std::uint8_t k = 0; k < num_bins; k++) {
            sycl::atomic_ref<__local_histogram_type,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                local_bin(local_histogram[k]);
            local_bin += histogram[k];
          }

          __self_item.barrier(sycl::access::fence_space::local_space);

          __reduce_out_histograms<BinType>(local_histogram, 0, bins, num_bins,
                                           __self_item);
        });
  });
  e.wait();
}

template <::std::uint16_t ItersPerWorkItem, typename BinType, typename Policy,
          typename Range1, typename Range2, typename Size, typename IdxHashFunc,
          typename... Range3>
inline void __histogram_general_local_atomics(Policy &&policy,
                                              ::std::uint16_t work_group_size,
                                              Range1 &&input, Range2 &&bins,
                                              const Size &num_bins,
                                              IdxHashFunc func,
                                              Range3 &&...opt_range) {
  const ::std::size_t N = input.size();
  ::std::size_t segments = __ceiling_div(N, work_group_size * ItersPerWorkItem);
  auto e = policy.queue().submit([&](auto &h) {
    // Temporary use of stable non-public API from oneDPL,  this function will
    // be replaced with oneDPL call in an upcoming release.
    oneapi::dpl::__ranges::__require_access(h, input, bins, opt_range...);
    sycl::local_accessor<::std::uint32_t, 1> local_histogram(
        sycl::range(num_bins), h);
    h.parallel_for(
        sycl::nd_range<1>(segments * work_group_size, work_group_size),
        [=](sycl::nd_item<1> __self_item) {
          using __bin_idx_type = ::std::uint16_t;
          constexpr auto __atomic_address_space =
              sycl::access::address_space::local_space;
          const ::std::size_t __self_lidx = __self_item.get_local_id(0);
          const ::std::uint32_t __wgroup_idx = __self_item.get_group(0);
          const ::std::size_t __seg_start =
              work_group_size * __wgroup_idx * ItersPerWorkItem;

          __clear_wglocal_histograms(local_histogram, 0, num_bins, __self_item);

          if (__seg_start + work_group_size * ItersPerWorkItem < N) {
            _DPCT_PRAGMA_UNROLL
            for (::std::uint8_t idx = 0; idx < ItersPerWorkItem; idx++) {
              __accum_local_atomics_iter<__bin_idx_type,
                                         __atomic_address_space>(
                  input, __seg_start + idx * work_group_size + __self_lidx,
                  local_histogram, 0, func);
            }
          } else {
            _DPCT_PRAGMA_UNROLL
            for (::std::uint8_t idx = 0; idx < ItersPerWorkItem; idx++) {
              ::std::size_t __val_idx =
                  __seg_start + idx * work_group_size + __self_lidx;
              if (__val_idx < N) {
                __accum_local_atomics_iter<__bin_idx_type,
                                           __atomic_address_space>(
                    input, __val_idx, local_histogram, 0, func);
              }
            }
          }
          __self_item.barrier(sycl::access::fence_space::local_space);

          __reduce_out_histograms<BinType>(local_histogram, 0, bins, num_bins,
                                           __self_item);
        });
  });

  e.wait();
}

template <::std::uint16_t __min_iters_per_work_item, typename BinType,
          typename Policy, typename Range1, typename Range2, typename Size,
          typename IdxHashFunc, typename... Range3>
inline void __histogram_general_private_global_atomics(
    Policy &&policy, ::std::uint16_t work_group_size, Range1 &&input,
    Range2 &&bins, const Size &num_bins, IdxHashFunc func,
    Range3 &&...opt_range) {

  const ::std::size_t N = input.size();
  auto __global_mem_size =
      policy.queue()
          .get_device()
          .template get_info<sycl::info::device::global_mem_size>();
  const ::std::size_t max_segments =
      ::std::min(__global_mem_size / (num_bins * sizeof(BinType)),
                 __ceiling_div(N, work_group_size * __min_iters_per_work_item));
  const ::std::size_t iters_per_work_item =
      __ceiling_div(N, max_segments * work_group_size);
  ::std::size_t segments =
      __ceiling_div(N, work_group_size * iters_per_work_item);

  sycl::buffer<BinType, 1> private_histograms(
      sycl::range<1>(segments * num_bins));

  auto e = policy.queue().submit([&](auto &h) {
    // Temporary use of stable non-public API from oneDPL,  this function will
    // be replaced with oneDPL call in an upcoming release.
    oneapi::dpl::__ranges::__require_access(h, input, bins, opt_range...);
    sycl::accessor hacc_private(private_histograms, h, sycl::read_write,
                                sycl::no_init);
    h.parallel_for(
        sycl::nd_range<1>(segments * work_group_size, work_group_size),
        [=](sycl::nd_item<1> __self_item) {
          using __bin_idx_type = ::std::uint32_t;
          constexpr auto __atomic_address_space =
              sycl::access::address_space::global_space;
          const ::std::size_t __self_lidx = __self_item.get_local_id(0);
          const ::std::size_t __wgroup_idx = __self_item.get_group(0);
          const ::std::size_t __seg_start =
              work_group_size * iters_per_work_item * __wgroup_idx;

          __clear_wglocal_histograms(hacc_private, __wgroup_idx * num_bins,
                                     num_bins, __self_item);
          if (__seg_start + work_group_size * iters_per_work_item < N) {
            for (::std::size_t idx = 0; idx < iters_per_work_item; idx++) {
              __accum_local_atomics_iter<__bin_idx_type,
                                         __atomic_address_space>(
                  input, __seg_start + idx * work_group_size + __self_lidx,
                  hacc_private, __wgroup_idx * num_bins, func);
            }
          } else {
            for (::std::size_t idx = 0; idx < iters_per_work_item; idx++) {
              ::std::size_t __val_idx =
                  __seg_start + idx * work_group_size + __self_lidx;
              if (__val_idx < N) {
                __accum_local_atomics_iter<__bin_idx_type,
                                           __atomic_address_space>(
                    input, __val_idx, hacc_private, __wgroup_idx * num_bins,
                    func);
              }
            }
          }
          __self_item.barrier(sycl::access::fence_space::local_space);

          __reduce_out_histograms<BinType>(hacc_private,
                                           __wgroup_idx * num_bins, bins,
                                           num_bins, __self_item);
        });
  });
  e.wait();
}

template <typename Policy, typename Iter1, typename Iter2, typename Size,
          typename IdxHashFunc, typename... Range>
inline Iter2
__histogram_general_select_best(Policy &&policy, Iter1 first, Iter1 last,
                                Iter2 histogram_first, const Size &num_bins,
                                IdxHashFunc func, Range &&...opt_range) {
  using __histo_value_type = typename ::std::iterator_traits<Iter2>::value_type;
  auto __local_mem_size =
      policy.queue()
          .get_device()
          .template get_info<sycl::info::device::local_mem_size>();
  constexpr ::std::uint8_t __max_registers = 16;

  // Temporary use of stable non-public API from oneDPL,  this function will be
  // replaced with oneDPL call in an upcoming release.
  auto keep_bins = oneapi::dpl::__ranges::__get_sycl_range<
      oneapi::dpl::__par_backend_hetero::access_mode::write, Iter2>();
  auto bins_buf = keep_bins(histogram_first, histogram_first + num_bins);

  oneapi::dpl::fill(policy, bins_buf.all_view().begin(),
                    bins_buf.all_view().end(), __histo_value_type(0));
  auto N = last - first;
  if (N > 0) {
    // Temporary use of stable non-public API from oneDPL,  this function will
    // be replaced with oneDPL call in an upcoming release.
    auto keep_input = oneapi::dpl::__ranges::__get_sycl_range<
        oneapi::dpl::__par_backend_hetero::access_mode::read, Iter1>();
    auto input_buf = keep_input(first, last);

    ::std::size_t max_work_group_size =
        policy.queue()
            .get_device()
            .template get_info<sycl::info::device::max_work_group_size>();
    ::std::size_t work_group_size =
        ::std::min(max_work_group_size, ::std::size_t(1024));

    if (num_bins < __max_registers) {

      // If bins fit into registers, use register private accumulation
      __histogram_general_registers_local_reduction<32, 16, __histo_value_type>(
          ::std::forward<Policy>(policy), work_group_size, input_buf.all_view(),
          bins_buf.all_view(), num_bins, func,
          ::std::forward<Range...>(opt_range)...);
    } else if (num_bins * sizeof(__histo_value_type) < __local_mem_size) {
      // If bins fit into SLM, use local atomics

      // Experimentally determined iters per work-item
      if (N <= 524288) {
        __histogram_general_local_atomics<4, __histo_value_type>(
            ::std::forward<Policy>(policy), work_group_size,
            input_buf.all_view(), bins_buf.all_view(), num_bins, func,
            ::std::forward<Range...>(opt_range)...);
      } else {
        __histogram_general_local_atomics<32, __histo_value_type>(
            ::std::forward<Policy>(policy), work_group_size,
            input_buf.all_view(), bins_buf.all_view(), num_bins, func,
            ::std::forward<Range...>(opt_range)...);
      }
    } else // Otherwise, use global atomics (private copies per workgroup)
    {
      // Experimentally determined iters per work-item
      if (N <= 524288) {
        __histogram_general_private_global_atomics<4, __histo_value_type>(
            ::std::forward<Policy>(policy), work_group_size,
            input_buf.all_view(), bins_buf.all_view(), num_bins, func,
            ::std::forward<Range...>(opt_range)...);
      } else {
        __histogram_general_private_global_atomics<32, __histo_value_type>(
            ::std::forward<Policy>(policy), work_group_size,
            input_buf.all_view(), bins_buf.all_view(), num_bins, func,
            ::std::forward<Range...>(opt_range)...);
      }
    }
  }
  return histogram_first + num_bins;
}

template <typename Policy, typename Iter1, typename Iter2, typename Size,
          typename T>
inline ::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
        dpct::internal::is_iterator<Iter2>::value &&
        internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value,
    Iter2>
histogram(Policy &&policy, Iter1 first, Iter1 last, Iter2 histogram_first,
          const Size &num_bins, const T &first_bin_min_val,
          const T &last_bin_max_val) {
  return __histogram_general_select_best(
      ::std::forward<Policy>(policy), first, last, histogram_first, num_bins,
      __evenly_divided_binhash<T>(first_bin_min_val, last_bin_max_val,
                                  num_bins));
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline ::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
        dpct::internal::is_iterator<Iter2>::value &&
        dpct::internal::is_iterator<Iter3>::value &&
        internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value,
    Iter2>
histogram(Policy &&policy, Iter1 first, Iter1 last, Iter2 histogram_first,
          Iter3 boundary_first, Iter3 boundary_last) {
  // Temporary use of stable non-public API from oneDPL,  this function will be
  // replaced with oneDPL call in an upcoming release.
  auto keep_boundaries = oneapi::dpl::__ranges::__get_sycl_range<
      oneapi::dpl::__par_backend_hetero::access_mode::read, Iter3>();
  auto boundary_buf = keep_boundaries(boundary_first, boundary_last);

  return __histogram_general_select_best(
      ::std::forward<Policy>(policy), first, last, histogram_first,
      (boundary_last - boundary_first) - 1,
      __custom_range_binhash{boundary_buf.all_view()}, boundary_buf.all_view());
}
} // end namespace dpl_histogram

} // end namespace internal

// Evenly Divided Histogram of a 1-D array
template <typename Policy, typename Iter1, typename Iter2, typename T,
          typename Size>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
histogram_even(Policy &&policy, Iter1 d_samples, Iter2 d_histogram,
               int num_levels, T lower_level, T upper_level, Size num_samples) {
  internal::dpl_histogram::histogram(::std::forward<Policy>(policy), d_samples,
                                     d_samples + num_samples, d_histogram,
                                     num_levels - 1, lower_level, upper_level);
}

// Evenly Divided Histogram of a 2-D ROI in a flattened 2-D array
template <typename Policy, typename Iter1, typename Iter2, typename T,
          typename OffsetT>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
histogram_even_roi(Policy &&policy, Iter1 d_samples, Iter2 d_histogram,
                   int num_levels, T lower_level, T upper_level,
                   OffsetT num_row_samples, OffsetT num_rows,
                   ::std::size_t row_stride_bytes) {
  return histogram_even(
      ::std::forward<Policy>(policy),
      oneapi::dpl::permutation_iterator(
          d_samples,
          internal::__roi_2d_index_functor(
              num_row_samples,
              row_stride_bytes /
                  sizeof(typename ::std::iterator_traits<Iter1>::value_type))),
      d_histogram, num_levels, lower_level, upper_level,
      num_row_samples * num_rows);
}

// Evenly Divided Multi-Channel Histogram of a 1-D array
template <int NumChannels, int NumActiveChannels, typename Policy,
          typename Iter1, typename Iter2, typename T, typename Size>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
multi_histogram_even(Policy &&policy, Iter1 d_samples,
                     Iter2 d_histogram[NumActiveChannels],
                     int num_levels[NumActiveChannels],
                     T lower_level[NumActiveChannels],
                     T upper_level[NumActiveChannels], Size num_pixels) {
  for (int active_channel = 0; active_channel < NumActiveChannels;
       active_channel++) {
    histogram_even(
        policy,
        oneapi::dpl::permutation_iterator(
            d_samples,
            internal::__interleaved_index_functor(NumChannels, active_channel)),
        d_histogram[active_channel], num_levels[active_channel],
        lower_level[active_channel], upper_level[active_channel], num_pixels);
  }
}

// Evenly Divided Multi-Channel Histogram of a 2-D ROI in a flattened 2-D array
template <int NumChannels, int NumActiveChannels, typename Policy,
          typename Iter1, typename Iter2, typename T, typename OffsetT>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
multi_histogram_even_roi(Policy &&policy, Iter1 d_samples,
                         Iter2 d_histogram[NumActiveChannels],
                         int num_levels[NumActiveChannels],
                         T lower_level[NumActiveChannels],
                         T upper_level[NumActiveChannels],
                         OffsetT num_row_samples, OffsetT num_rows,
                         ::std::size_t row_stride_bytes) {
  for (int active_channel = 0; active_channel < NumActiveChannels;
       active_channel++) {
    histogram_even(
        policy,
        oneapi::dpl::permutation_iterator(
            d_samples,
            internal::__composition_functor(
                internal::__roi_2d_index_functor(
                    num_row_samples,
                    row_stride_bytes /
                        (NumChannels * sizeof(typename ::std::iterator_traits<
                                              Iter1>::value_type))),
                internal::__interleaved_index_functor(NumChannels,
                                                      active_channel))),
        d_histogram[active_channel], num_levels[active_channel],
        lower_level[active_channel], upper_level[active_channel],
        num_row_samples * num_rows);
  }
}

// Custom Range Histogram of a 1-D array
template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Size>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    dpct::internal::is_iterator<Iter3>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
histogram_range(Policy &&policy, Iter1 d_samples, Iter2 d_histogram,
                int num_levels, Iter3 d_levels, Size num_samples) {
  internal::dpl_histogram::histogram(::std::forward<Policy>(policy), d_samples,
                                     d_samples + num_samples, d_histogram,
                                     d_levels, d_levels + num_levels);
}

// Custom Range Histogram of a 2-D ROI in a flattened 2-D Array
template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename OffsetT>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    dpct::internal::is_iterator<Iter3>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
histogram_range_roi(Policy &&policy, Iter1 d_samples, Iter2 d_histogram,
                    int num_levels, Iter3 d_levels, OffsetT num_row_samples,
                    OffsetT num_rows, ::std::size_t row_stride_bytes) {
  return histogram_range(
      ::std::forward<Policy>(policy),
      oneapi::dpl::permutation_iterator(
          d_samples,
          internal::__roi_2d_index_functor(
              num_row_samples,
              row_stride_bytes /
                  sizeof(typename ::std::iterator_traits<Iter1>::value_type))),
      d_histogram, num_levels, d_levels, num_row_samples * num_rows);
}

// Custom Range Multi-Channel Histogram of a 1-D array
template <int NumChannels, int NumActiveChannels, typename Policy,
          typename Iter1, typename Iter2, typename Iter3, typename Size>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    dpct::internal::is_iterator<Iter3>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
multi_histogram_range(Policy &&policy, Iter1 d_samples,
                      Iter2 d_histogram[NumActiveChannels],
                      int num_levels[NumActiveChannels],
                      Iter3 d_levels[NumActiveChannels], Size num_pixels) {
  for (int active_channel = 0; active_channel < NumActiveChannels;
       active_channel++) {
    histogram_range(policy,
                    oneapi::dpl::permutation_iterator(
                        d_samples, internal::__interleaved_index_functor(
                                       NumChannels, active_channel)),
                    d_histogram[active_channel], num_levels[active_channel],
                    d_levels[active_channel], num_pixels);
  }
}

// Custom Range Multi-Channel Histogram of a 2-D ROI in a flattened 2-D array
template <int NumChannels, int NumActiveChannels, typename Policy,
          typename Iter1, typename Iter2, typename Iter3, typename OffsetT>
::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    dpct::internal::is_iterator<Iter3>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
multi_histogram_range_roi(Policy &&policy, Iter1 d_samples,
                          Iter2 d_histogram[NumActiveChannels],
                          int num_levels[NumActiveChannels],
                          Iter3 d_levels[NumActiveChannels],
                          OffsetT num_row_samples, OffsetT num_rows,
                          ::std::size_t row_stride_bytes) {
  for (int active_channel = 0; active_channel < NumActiveChannels;
       active_channel++) {
    histogram_range(
        policy,
        oneapi::dpl::permutation_iterator(
            d_samples,
            internal::__composition_functor(
                internal::__roi_2d_index_functor(
                    num_row_samples,
                    row_stride_bytes /
                        (NumChannels * sizeof(typename ::std::iterator_traits<
                                              Iter1>::value_type))),
                internal::__interleaved_index_functor(NumChannels,
                                                      active_channel))),
        d_histogram[active_channel], num_levels[active_channel],
        d_levels[active_channel], num_row_samples * num_rows);
  }
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4>
inline ::std::enable_if_t<dpct::internal::is_iterator<Iter1>::value &&
                          dpct::internal::is_iterator<Iter2>::value &&
                          dpct::internal::is_iterator<Iter3>::value &&
                          dpct::internal::is_iterator<Iter4>::value>
sort_pairs(Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter3 values_in,
           Iter4 values_out, ::std::int64_t n, bool descending, int begin_bit,
           int end_bit) {
  internal::sort_pairs_impl(std::forward<Policy>(policy), keys_in, keys_out,
                            values_in, values_out, n, descending, begin_bit,
                            end_bit);
}

template <typename Policy, typename Iter1, typename Iter2>
inline void sort_pairs(
    Policy &&policy, io_iterator_pair<Iter1> &keys,
    io_iterator_pair<Iter2> &values, ::std::int64_t n, bool descending = false,
    bool do_swap_iters = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  sort_pairs(::std::forward<Policy>(policy), keys.first(), keys.second(),
             values.first(), values.second(), n, descending, begin_bit,
             end_bit);
  if (do_swap_iters) {
    keys.swap();
    values.swap();
  }
}

template <typename Policy, typename Iter1, typename Iter2>
inline ::std::enable_if_t<dpct::internal::is_iterator<Iter1>::value &&
                          dpct::internal::is_iterator<Iter2>::value>
sort_keys(Policy &&policy, Iter1 keys_in, Iter2 keys_out, ::std::int64_t n,
          bool descending, int begin_bit, int end_bit) {
  using key_t_value_t = typename ::std::iterator_traits<Iter1>::value_type;

  int clipped_begin_bit = ::std::max(begin_bit, 0);
  int clipped_end_bit =
      ::std::min((::std::uint64_t)end_bit, sizeof(key_t_value_t) * 8);
  int num_bytes = (clipped_end_bit - clipped_begin_bit - 1) / 8 + 1;

  auto transform_and_sort_f = [&](auto x) {
    using T = typename ::std::decay_t<decltype(x)>;
    internal::transform_and_sort<decltype(policy), Iter1, Iter2, T>(
        ::std::forward<Policy>(policy), keys_in, keys_out, n, descending,
        clipped_begin_bit, clipped_end_bit);
  };

  if (clipped_end_bit - clipped_begin_bit == sizeof(key_t_value_t) * 8) {
    internal::sort_only(::std::forward<Policy>(policy), keys_in, keys_out, n,
                        descending);
  } else if (num_bytes == 1) {
    transform_and_sort_f.template operator()<uint8_t>(0);
  } else if (num_bytes == 2) {
    transform_and_sort_f.template operator()<uint16_t>(0);
  } else if (num_bytes <= 4) {
    transform_and_sort_f.template operator()<uint32_t>(0);
  } else // if (num_bytes <= 8)
  {
    transform_and_sort_f.template operator()<::std::uint64_t>(0);
  }
}

template <typename Policy, typename Iter1>
inline void sort_keys(
    Policy &&policy, io_iterator_pair<Iter1> &keys, ::std::int64_t n,
    bool descending = false, bool do_swap_iters = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  sort_keys(std::forward<Policy>(policy), keys.first(), keys.second(), n,
            descending, begin_bit, end_bit);
  if (do_swap_iters)
    keys.swap();
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline ::std::enable_if_t<dpct::internal::is_iterator<Iter1>::value &&
                          dpct::internal::is_iterator<Iter2>::value>
segmented_sort_keys(
    Policy &&policy, Iter1 keys_in, Iter2 keys_out, ::std::int64_t n,
    ::std::int64_t nsegments, Iter3 begin_offsets, Iter3 end_offsets,
    bool descending = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  int compute_units =
      policy.queue()
          .get_device()
          .template get_info<sycl::info::device::max_compute_units>();
  auto sg_sizes = policy.queue()
                      .get_device()
                      .template get_info<sycl::info::device::sub_group_sizes>();
  int subgroup_size = sg_sizes.empty() ? 1 : sg_sizes.back();
  // parallel for of serial sorts when we have sufficient number of segments for
  // load balance when number of segments is large as compared to our target
  // compute capability
  if (nsegments >
      compute_units *
          (policy.queue().get_device().is_gpu() ? subgroup_size : 1)) {
    dpct::internal::segmented_sort_keys_by_parallel_for_of_sorts(
        ::std::forward<Policy>(policy), keys_in, keys_out, n, nsegments,
        begin_offsets, end_offsets, descending, begin_bit, end_bit);
  } else
  {
    dpct::internal::segmented_sort_keys_by_parallel_sorts(
        ::std::forward<Policy>(policy), keys_in, keys_out, n, nsegments,
        begin_offsets, end_offsets, descending, begin_bit, end_bit);
  }
}

template <typename Policy, typename Iter1, typename Iter2>
inline void segmented_sort_keys(
    Policy &&policy, io_iterator_pair<Iter1> &keys, ::std::int64_t n,
    ::std::int64_t nsegments, Iter2 begin_offsets, Iter2 end_offsets,
    bool descending = false, bool do_swap_iters = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  segmented_sort_keys(::std::forward<Policy>(policy), keys.first(),
                      keys.second(), n, nsegments, begin_offsets, end_offsets,
                      descending, begin_bit, end_bit);
  if (do_swap_iters) {
    keys.swap();
  }
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3,
          typename Iter4, typename Iter5>
inline ::std::enable_if_t<dpct::internal::is_iterator<Iter1>::value &&
                          dpct::internal::is_iterator<Iter2>::value &&
                          dpct::internal::is_iterator<Iter3>::value &&
                          dpct::internal::is_iterator<Iter4>::value>
segmented_sort_pairs(
    Policy &&policy, Iter1 keys_in, Iter2 keys_out, Iter3 values_in,
    Iter4 values_out, ::std::int64_t n, ::std::int64_t nsegments,
    Iter5 begin_offsets, Iter5 end_offsets, bool descending = false,
    int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  int compute_units =
      policy.queue()
          .get_device()
          .template get_info<sycl::info::device::max_compute_units>();
  auto sg_sizes = policy.queue()
                      .get_device()
                      .template get_info<sycl::info::device::sub_group_sizes>();
  int subgroup_size = sg_sizes.empty() ? 1 : sg_sizes.back();
  // parallel for of serial sorts when we have sufficient number of segments for
  // load balance when number of segments is large as compared to our target
  // compute capability
  if (nsegments >
      compute_units *
          (policy.queue().get_device().is_gpu() ? subgroup_size : 1)) {
    dpct::internal::segmented_sort_pairs_by_parallel_for_of_sorts(
        ::std::forward<Policy>(policy), keys_in, keys_out, values_in,
        values_out, n, nsegments, begin_offsets, end_offsets, descending,
        begin_bit, end_bit);
  } else
  {
    dpct::internal::segmented_sort_pairs_by_parallel_sorts(
        ::std::forward<Policy>(policy), keys_in, keys_out, values_in,
        values_out, n, nsegments, begin_offsets, end_offsets, descending,
        begin_bit, end_bit);
  }
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline void segmented_sort_pairs(
    Policy &&policy, io_iterator_pair<Iter1> &keys,
    io_iterator_pair<Iter2> &values, ::std::int64_t n, ::std::int64_t nsegments,
    Iter3 begin_offsets, Iter3 end_offsets, bool descending = false,
    bool do_swap_iters = false, int begin_bit = 0,
    int end_bit = sizeof(typename ::std::iterator_traits<Iter1>::value_type) *
                  8) {
  segmented_sort_pairs(std::forward<Policy>(policy), keys.first(),
                       keys.second(), values.first(), values.second(), n,
                       nsegments, begin_offsets, end_offsets, descending,
                       begin_bit, end_bit);
  if (do_swap_iters) {
    keys.swap();
    values.swap();
  }
}

template <typename Policy, typename Iter1, typename Iter2>
inline void reduce_argmax(Policy &&policy, Iter1 input, Iter2 output,
                          ::std::size_t n) {
  dpct::arg_index_input_iterator<decltype(input), int> input_arg_idx(input);
  auto ret = ::std::max_element(
      policy, input_arg_idx, input_arg_idx + n,
      [](const auto &a, const auto &b) { return (a.value < b.value); });
  ::std::copy(::std::forward<Policy>(policy), ret, ret + 1, output);
}

template <typename Policy, typename Iter1, typename Iter2>
inline void reduce_argmin(Policy &&policy, Iter1 input, Iter2 output,
                          ::std::size_t n) {
  dpct::arg_index_input_iterator<decltype(input), int> input_arg_idx(input);
  auto ret = ::std::min_element(
      policy, input_arg_idx, input_arg_idx + n,
      [](const auto &a, const auto &b) { return (a.value < b.value); });
  ::std::copy(::std::forward<Policy>(policy), ret, ret + 1, output);
}

template <typename Policy, typename Iter1, typename ValueT, typename CompT>
inline ::std::pair<Iter1, Iter1> equal_range(Policy &&policy, Iter1 start,
                                             Iter1 end, const ValueT &value,
                                             CompT comp) {
  ::std::vector<::std::int64_t> res_lower(1);
  ::std::vector<::std::int64_t> res_upper(1);
  ::std::vector<ValueT> value_vec(1, value);
  ::oneapi::dpl::lower_bound(policy, start, end, value_vec.begin(),
                             value_vec.end(), res_lower.begin(), comp);
  ::oneapi::dpl::upper_bound(::std::forward<Policy>(policy), start, end,
                             value_vec.begin(), value_vec.end(),
                             res_upper.begin(), comp);
  return ::std::make_pair(start + res_lower[0], start + res_upper[0]);
}

template <typename Policy, typename Iter1, typename ValueT>
inline ::std::pair<Iter1, Iter1> equal_range(Policy &&policy, Iter1 start,
                                             Iter1 end, const ValueT &value) {
  return equal_range(::std::forward<Policy>(policy), start, end, value,
                     internal::__less());
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline ::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
segmented_reduce_argmin(Policy &&policy, Iter1 keys_in, Iter2 keys_out,
                        ::std::int64_t nsegments, Iter3 begin_offsets,
                        Iter3 end_offsets) {
  policy.queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nsegments, [=](sycl::id<1> i) {
      if (end_offsets[i] <= begin_offsets[i]) {
        keys_out[i] = dpct::key_value_pair(
            1, ::std::numeric_limits<
                   typename ::std::iterator_traits<Iter1>::value_type>::max());
      } else {
        dpct::arg_index_input_iterator<Iter1, int> arg_index(keys_in +
                                                             begin_offsets[i]);
        keys_out[i] = *::std::min_element(
            arg_index, arg_index + (end_offsets[i] - begin_offsets[i]),
            [](const auto &a, const auto &b) { return a.value < b.value; });
      }
    });
  });
  policy.queue().wait();
}

template <typename Policy, typename Iter1, typename Iter2, typename Iter3>
inline ::std::enable_if_t<
    dpct::internal::is_iterator<Iter1>::value &&
    dpct::internal::is_iterator<Iter2>::value &&
    internal::is_hetero_execution_policy<::std::decay_t<Policy>>::value>
segmented_reduce_argmax(Policy &&policy, Iter1 keys_in, Iter2 keys_out,
                        ::std::int64_t nsegments, Iter3 begin_offsets,
                        Iter3 end_offsets) {
  policy.queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(nsegments, [=](sycl::id<1> i) {
      if (end_offsets[i] <= begin_offsets[i]) {
        keys_out[i] = dpct::key_value_pair(
            1,
            ::std::numeric_limits<
                typename ::std::iterator_traits<Iter1>::value_type>::lowest());
      } else {
        dpct::arg_index_input_iterator<Iter1, int> arg_index(keys_in +
                                                             begin_offsets[i]);
        keys_out[i] = *::std::max_element(
            arg_index, arg_index + (end_offsets[i] - begin_offsets[i]),
            [](const auto &a, const auto &b) { return a.value < b.value; });
      }
    });
  });
  policy.queue().wait();
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator1, typename OutputIterator2,
          typename OutputIterator3>
void nontrivial_run_length_encode(ExecutionPolicy &&policy,
                                  InputIterator input_beg,
                                  OutputIterator1 offsets_out,
                                  OutputIterator2 lengths_out,
                                  OutputIterator3 num_runs,
                                  ::std::int64_t num_items) {
  using oneapi::dpl::make_transform_iterator;
  using oneapi::dpl::make_zip_iterator;
  using offsets_t =
      typename ::std::iterator_traits<OutputIterator1>::value_type;
  using lengths_t =
      typename ::std::iterator_traits<OutputIterator2>::value_type;

  auto input_end = input_beg + num_items;
  // First element must be nontrivial run (start of first segment)
  auto first_adj_it = oneapi::dpl::adjacent_find(policy, input_beg, input_end);
  auto first_adj_idx = ::std::distance(input_beg, first_adj_it);
  if (first_adj_it == input_end) {
    ::std::fill(policy, num_runs, num_runs + 1, 0);
    return;
  }
  auto get_prev_idx_element = [first_adj_idx](const auto &idx) {
    auto out_idx = idx + first_adj_idx;
    return (out_idx == 0) ? 0 : out_idx - 1;
  };
  auto get_next_idx_element = [first_adj_idx, num_items](const auto &idx) {
    auto out_idx = idx + first_adj_idx;
    return (out_idx == num_items - 1) ? num_items - 1 : out_idx + 1;
  };
  // TODO: Use shifted view to pad range once oneDPL ranges is non-experimental
  auto left_shifted_input_beg =
      oneapi::dpl::make_permutation_iterator(input_beg, get_prev_idx_element);
  auto right_shifted_input_beg =
      oneapi::dpl::make_permutation_iterator(input_beg, get_next_idx_element);
  // Segment type for ith idx consists of zip of iterators at (i-1, i, i+1)
  // padded at the ends
  auto zipped_keys_beg = make_zip_iterator(
      left_shifted_input_beg, input_beg, right_shifted_input_beg,
      oneapi::dpl::counting_iterator<offsets_t>(0));
  // Set flag at the beginning of new nontrivial run (ex: (2, 3, 3) -> 1)
  auto key_flags_beg =
      make_transform_iterator(zipped_keys_beg, [num_items](const auto &zipped) {
        using ::std::get;
        bool last_idx_mask = get<3>(zipped) != num_items - 1;
        return (get<0>(zipped) != get<1>(zipped) &&
                get<1>(zipped) == get<2>(zipped)) &&
               last_idx_mask;
      });
  auto count_beg = oneapi::dpl::counting_iterator<offsets_t>(0);
  auto const_it = dpct::make_constant_iterator(lengths_t(1));
  // Check for presence of nontrivial element at current index
  auto tr_nontrivial_flags = make_transform_iterator(
      make_zip_iterator(left_shifted_input_beg, input_beg),
      [](const auto &zip) {
        using ::std::get;
        return get<0>(zip) == get<1>(zip);
      });
  auto zipped_vals_beg =
      make_zip_iterator(tr_nontrivial_flags, count_beg, const_it);
  auto pred = [](bool lhs, bool rhs) { return !rhs; };
  auto op = [](auto lhs, const auto &rhs) {
    using ::std::get;

    // Update length count of run.
    // The first call to this op will use the first element of the input as lhs
    // and second element as rhs. get<0>(first_element) is ignored in favor of a
    // constant `1` in get<2>, avoiding the need for special casing the first
    // element. The constant `1` utilizes the knowledge that each segment begins
    // with a nontrivial run.
    get<2>(lhs) += get<0>(rhs);

    // A run's starting index is stored in get<1>(lhs) as the initial value in
    // the segment and is preserved throughout the segment's reduction as the
    // nontrivial run's offset.

    return ::std::move(lhs);
  };
  auto zipped_out_beg = make_zip_iterator(oneapi::dpl::discard_iterator(),
                                          offsets_out, lengths_out);
  auto [_, zipped_out_vals_end] = oneapi::dpl::reduce_by_segment(
      policy, key_flags_beg + first_adj_idx, key_flags_beg + num_items,
      zipped_vals_beg + first_adj_idx, oneapi::dpl::discard_iterator(),
      zipped_out_beg, pred, op);
  auto ret_dist = ::std::distance(zipped_out_beg, zipped_out_vals_end);
  ::std::fill(policy, num_runs, num_runs + 1, ret_dist);
}

} // end namespace dpct

#endif
