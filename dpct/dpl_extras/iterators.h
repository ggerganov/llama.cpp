//==---- iterators.h ------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_ITERATORS_H__
#define __DPCT_ITERATORS_H__

#include <oneapi/dpl/iterator>

#include "functional.h"

namespace dpct {

namespace internal {

// Wrapper class returned from a dereferenced transform_iterator which was
// created using
//  make_transform_output_iterator(). Used to apply the supplied transform
//  function when writing into an object of this class.
//
// Example:
// int a[] = {0, 1, 2, 3, 4};
// int* p = a;
// auto f = [](auto v) {return v*v;};
// auto tr_out = dpct::make_transform_output_iterator(p+1, f);
// auto wrap = *tr_out;         // wrap is a transform_output_ref_wrapper
// std::cout<<*(p+1)<<std::endl;  // '1'
// wrap = 2;                    // apply function, store 2*2=4
// std::cout<<*(p+1)<<std::endl;  // '4'
template <typename T, typename _UnaryFunc> class transform_output_ref_wrapper {
private:
  T __my_reference_;
  _UnaryFunc __my_unary_func_;

public:
  template <typename U>
  transform_output_ref_wrapper(U &&__reference, _UnaryFunc __unary_func)
      : __my_reference_(std::forward<U>(__reference)),
        __my_unary_func_(__unary_func) {}

  // When writing to an object of this type, apply the supplied unary function,
  // then write to the wrapped reference
  template <typename UnaryInputType>
  transform_output_ref_wrapper &operator=(const UnaryInputType &e) {
    __my_reference_ = __my_unary_func_(e);
    return *this;
  }
};

// Unary functor to create a transform_output_reference_wrapper when a
// transform_iterator is dereferenced, so that a
// the supplied unary function may be applied on write, resulting in a
// transform_output_iterator
template <typename _UnaryFunc> struct _Unary_Out {
  _Unary_Out(_UnaryFunc __f_) : __f(__f_) {}
  _UnaryFunc __f;
  template <typename T> auto operator()(T &&val) const {
    return transform_output_ref_wrapper<T, _UnaryFunc>(std::forward<T>(val),
                                                       __f);
  }
};

} // end namespace internal

using std::advance;

using std::distance;

template <typename T>
oneapi::dpl::counting_iterator<T> make_counting_iterator(const T &input) {
  return oneapi::dpl::counting_iterator<T>(input);
}

template <typename _Tp> class constant_iterator {
public:
  typedef std::false_type is_hetero;
  typedef std::true_type is_passed_directly;
  typedef std::ptrdiff_t difference_type;
  typedef _Tp value_type;
  typedef _Tp *pointer;
  // There is no storage behind the iterator, so we return a value instead of
  // reference.
  typedef const _Tp reference;
  typedef const _Tp const_reference;
  typedef std::random_access_iterator_tag iterator_category;

  explicit constant_iterator(_Tp __init)
      : __my_value_(__init), __my_counter_(0) {}

private:
  // used to construct iterator instances with different counter values required
  // by arithmetic operators
  constant_iterator(const _Tp &__value, const difference_type &__offset)
      : __my_value_(__value), __my_counter_(__offset) {}

public:
  // non-const variants of access operators are not provided so unintended
  // writes are caught at compile time.
  const_reference operator*() const { return __my_value_; }
  const_reference operator[](difference_type) const { return __my_value_; }

  difference_type operator-(const constant_iterator &__it) const {
    return __my_counter_ - __it.__my_counter_;
  }

  constant_iterator &operator+=(difference_type __forward) {
    __my_counter_ += __forward;
    return *this;
  }
  constant_iterator &operator-=(difference_type __backward) {
    return *this += -__backward;
  }
  constant_iterator &operator++() { return *this += 1; }
  constant_iterator &operator--() { return *this -= 1; }

  constant_iterator operator++(int) {
    constant_iterator __it(*this);
    ++(*this);
    return __it;
  }
  constant_iterator operator--(int) {
    constant_iterator __it(*this);
    --(*this);
    return __it;
  }

  constant_iterator operator-(difference_type __backward) const {
    return constant_iterator(__my_value_, __my_counter_ - __backward);
  }
  constant_iterator operator+(difference_type __forward) const {
    return constant_iterator(__my_value_, __my_counter_ + __forward);
  }
  friend constant_iterator operator+(difference_type __forward,
                                     const constant_iterator __it) {
    return __it + __forward;
  }

  bool operator==(const constant_iterator &__it) const {
    return __my_value_ == __it.__my_value_ &&
           this->__my_counter_ == __it.__my_counter_;
  }
  bool operator!=(const constant_iterator &__it) const {
    return !(*this == __it);
  }
  bool operator<(const constant_iterator &__it) const {
    return *this - __it < 0;
  }
  bool operator>(const constant_iterator &__it) const { return __it < *this; }
  bool operator<=(const constant_iterator &__it) const {
    return !(*this > __it);
  }
  bool operator>=(const constant_iterator &__it) const {
    return !(*this < __it);
  }

private:
  _Tp __my_value_;
  uint64_t __my_counter_;
};

template <typename _Tp>
constant_iterator<_Tp> make_constant_iterator(_Tp __value) {
  return constant_iterator<_Tp>(__value);
}

// key_value_pair class to represent a key and value, specifically a
// dereferenced arg_index_input_iterator
template <typename _KeyTp, typename _ValueTp> class key_value_pair {
public:
  key_value_pair() = default;

  key_value_pair(const _KeyTp &_key, const _ValueTp &_value)
      : key(_key), value(_value) {}

  bool operator==(const key_value_pair<_KeyTp, _ValueTp> &_kvp) const {
    return (key == _kvp.key) && (value == _kvp.value);
  }

  bool operator!=(const key_value_pair<_KeyTp, _ValueTp> &_kvp) const {
    return (key != _kvp.key) || (value != _kvp.value);
  }

  _KeyTp key;
  _ValueTp value;
};

namespace detail {

template <typename KeyTp, typename _ValueTp> struct make_key_value_pair {
  template <typename ValRefTp>
  key_value_pair<KeyTp, _ValueTp>
  operator()(const oneapi::dpl::__internal::tuple<KeyTp, ValRefTp> &tup) const {
    return ::dpct::key_value_pair<KeyTp, _ValueTp>(::std::get<0>(tup),
                                                   ::std::get<1>(tup));
  }
};

template <class T> struct __zip_iterator_impl;
template <class... Ts> struct __zip_iterator_impl<std::tuple<Ts...>> {
  using type = oneapi::dpl::zip_iterator<Ts...>;
};

} // end namespace detail

// dpct::zip_iterator can only accept std::tuple type as template argument for
// compatibility purpose. Please use oneapi::dpl::zip_iterator if you want to
// pass iterator's types directly.
template <typename... Ts>
using zip_iterator = typename detail::__zip_iterator_impl<Ts...>::type;

// arg_index_input_iterator is an iterator over a input iterator, with a index.
// When dereferenced, it returns a key_value_pair, which can be interrogated for
// the index key or the value from the input iterator
template <typename InputIteratorT, typename OffsetT = ptrdiff_t,
          typename OutputValueT =
              typename ::std::iterator_traits<InputIteratorT>::value_type>
class arg_index_input_iterator
    : public oneapi::dpl::transform_iterator<
          oneapi::dpl::zip_iterator<oneapi::dpl::counting_iterator<OffsetT>,
                                    InputIteratorT>,
          detail::make_key_value_pair<OffsetT, OutputValueT>> {
  using arg_index_input_iterator_wrap = oneapi::dpl::transform_iterator<
      oneapi::dpl::zip_iterator<oneapi::dpl::counting_iterator<OffsetT>,
                                InputIteratorT>,
      detail::make_key_value_pair<OffsetT, OutputValueT>>;

public:
  typedef OffsetT difference_type;

  // signal to __get_sycl_range that this iterator is as a direct pass iterator
  using is_zip = ::std::true_type;

  arg_index_input_iterator(const arg_index_input_iterator_wrap &__arg_wrap)
      : arg_index_input_iterator_wrap(__arg_wrap) {}
  arg_index_input_iterator(InputIteratorT __iter)
      : arg_index_input_iterator_wrap(
            oneapi::dpl::make_zip_iterator(
                oneapi::dpl::counting_iterator(OffsetT(0)), __iter),
            detail::make_key_value_pair<OffsetT, OutputValueT>()) {}

  arg_index_input_iterator &operator=(const arg_index_input_iterator &__input) {
    arg_index_input_iterator_wrap::operator=(__input);
    return *this;
  }
  arg_index_input_iterator &operator++() {
    arg_index_input_iterator_wrap::operator++();
    return *this;
  }
  arg_index_input_iterator &operator--() {
    arg_index_input_iterator_wrap::operator--();
    return *this;
  }
  arg_index_input_iterator operator++(int) {
    arg_index_input_iterator __it(*this);
    ++(*this);
    return __it;
  }
  arg_index_input_iterator operator--(int) {
    arg_index_input_iterator __it(*this);
    --(*this);
    return __it;
  }
  arg_index_input_iterator operator+(difference_type __forward) const {
    return arg_index_input_iterator(
        arg_index_input_iterator_wrap::operator+(__forward));
  }
  arg_index_input_iterator operator-(difference_type __backward) const {
    return arg_index_input_iterator(
        arg_index_input_iterator_wrap::operator-(__backward));
  }
  arg_index_input_iterator &operator+=(difference_type __forward) {
    arg_index_input_iterator_wrap::operator+=(__forward);
    return *this;
  }
  arg_index_input_iterator &operator-=(difference_type __backward) {
    arg_index_input_iterator_wrap::operator-=(__backward);
    return *this;
  }

  friend arg_index_input_iterator
  operator+(difference_type __forward, const arg_index_input_iterator &__it) {
    return __it + __forward;
  }

  difference_type operator-(const arg_index_input_iterator &__it) const {
    return arg_index_input_iterator_wrap::operator-(__it);
  }
  bool operator==(const arg_index_input_iterator &__it) const {
    return arg_index_input_iterator_wrap::operator==(__it);
  }
  bool operator!=(const arg_index_input_iterator &__it) const {
    return !(*this == __it);
  }
  bool operator<(const arg_index_input_iterator &__it) const {
    return *this - __it < 0;
  }
  bool operator>(const arg_index_input_iterator &__it) const {
    return __it < *this;
  }
  bool operator<=(const arg_index_input_iterator &__it) const {
    return !(*this > __it);
  }
  bool operator>=(const arg_index_input_iterator &__it) const {
    return !(*this < __it);
  }

  // returns an arg_index_input_iterator with the same iter position, but a
  // count reset to 0
  arg_index_input_iterator create_normalized() {
    return arg_index_input_iterator(
        ::std::get<1>(arg_index_input_iterator_wrap::base().base()));
  }
};

template <typename IterT> struct io_iterator_pair {
  inline io_iterator_pair() : selector(false) {}

  inline io_iterator_pair(const IterT &first, const IterT &second)
      : selector(false) {
    iter[0] = first;
    iter[1] = second;
  }

  inline IterT first() const { return selector ? iter[1] : iter[0]; }

  inline IterT second() const { return selector ? iter[0] : iter[1]; }

  inline void swap() { selector = !selector; }

  bool selector;

  IterT iter[2];
};

template <typename _Iter, typename _UnaryFunc>
auto make_transform_output_iterator(_Iter __it, _UnaryFunc __unary_func) {
  return oneapi::dpl::transform_iterator(
      __it, internal::_Unary_Out<_UnaryFunc>(__unary_func));
}

} // end namespace dpct

#endif
