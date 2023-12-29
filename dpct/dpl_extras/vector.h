//==---- vector.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_VECTOR_H__
#define __DPCT_VECTOR_H__

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <sycl/sycl.hpp>

#include "memory.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "../device.hpp"

namespace dpct {

namespace internal {
template <typename Iter, typename Void = void> // for non-iterators
struct is_iterator : std::false_type {};

template <typename Iter> // For iterators
struct is_iterator<
    Iter,
    typename std::enable_if<
        !std::is_void<typename Iter::iterator_category>::value, void>::type>
    : std::true_type {};

template <typename T> // For pointers
struct is_iterator<T *> : std::true_type {};
} // end namespace internal

#ifndef DPCT_USM_LEVEL_NONE

template <typename T,
          typename Allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>>
class device_vector {
public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename ::std::iterator_traits<iterator>::difference_type;
  using size_type = ::std::size_t;

private:
  Allocator _alloc;
  size_type _size;
  size_type _capacity;
  pointer _storage;

  size_type _min_capacity() const { return size_type(1); }

  void _set_capacity_and_alloc() {
    _capacity = ::std::max(_size * 2, _min_capacity());
    _storage = _alloc.allocate(_capacity);
  }

public:
  template <typename OtherA> operator ::std::vector<T, OtherA>() const {
    auto __tmp = ::std::vector<T, OtherA>(this->size());
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              this->begin(), this->end(), __tmp.begin());
    return __tmp;
  }
  device_vector()
      : _alloc(get_default_queue()), _size(0), _capacity(_min_capacity()) {
    _set_capacity_and_alloc();
  }
  ~device_vector() /*= default*/ { _alloc.deallocate(_storage, _capacity); };
  explicit device_vector(size_type n) : device_vector(n, T()) {}
  explicit device_vector(size_type n, const T &value)
      : _alloc(get_default_queue()), _size(n) {
    _set_capacity_and_alloc();
    if (_size > 0) {
      ::std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                begin(), end(), T(value));
    }
  }
  device_vector(const device_vector &other) : _alloc(get_default_queue()) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = _alloc.allocate(_capacity);
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              other.begin(), other.end(), begin());
  }
  device_vector(device_vector &&other)
      : _alloc(get_default_queue()), _size(other.size()),
        _capacity(other.capacity()), _storage(other._storage) {
    other._size = 0;
    other._capacity = 0; 
    other._storage = nullptr;
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename ::std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !::std::is_pointer<InputIterator>::value &&
                        ::std::is_same<typename ::std::iterator_traits<
                                         InputIterator>::iterator_category,
                                     ::std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _alloc(get_default_queue()) {
    _size = ::std::distance(first, last);
    _set_capacity_and_alloc();
    if (_size > 0) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                first, last, begin());
    }
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename ::std::enable_if<::std::is_pointer<InputIterator>::value,
                                        InputIterator>::type last)
      : _alloc(get_default_queue()) {
    _size = ::std::distance(first, last);
    _set_capacity_and_alloc();
    if (_size > 0) {
      auto ptr_type = sycl::get_pointer_type(first, get_default_context());
      if (ptr_type != sycl::usm::alloc::host &&
          ptr_type != sycl::usm::alloc::unknown) {
        ::std::copy(
            oneapi::dpl::execution::make_device_policy(get_default_queue()),
            first, last, begin());
      } else {
        sycl::buffer<typename ::std::iterator_traits<InputIterator>::value_type,
                     1>
            buf(first, last);
        auto buf_first = oneapi::dpl::begin(buf);
        auto buf_last = oneapi::dpl::end(buf);
        ::std::copy(
            oneapi::dpl::execution::make_device_policy(get_default_queue()),
            buf_first, buf_last, begin());
      }
    }
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename ::std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !::std::is_pointer<InputIterator>::value &&
                        !::std::is_same<typename ::std::iterator_traits<
                                          InputIterator>::iterator_category,
                                      ::std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _alloc(get_default_queue()), _size(::std::distance(first, last)) {
    _set_capacity_and_alloc();
    ::std::vector<T> _tmp(first, last);
    if (_size > 0) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                _tmp.begin(), _tmp.end(), this->begin());
    }
  }

  template <typename OtherAllocator>
  device_vector(const device_vector<T, OtherAllocator> &v)
      : _alloc(get_default_queue()), _storage(v.real_begin()), _size(v.size()),
        _capacity(v.capacity()) {}

  template <typename OtherAllocator>
  device_vector(::std::vector<T, OtherAllocator> &v)
      : _alloc(get_default_queue()), _size(v.size()) {
    _set_capacity_and_alloc();
    if (_size > 0) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                v.begin(), v.end(), this->begin());
    }
  }

  template <typename OtherAllocator>
  device_vector &operator=(const ::std::vector<T, OtherAllocator> &v) {
    resize(v.size());
    if (_size > 0) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                v.begin(), v.end(), begin());
    }
    return *this;
  }
  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    resize(other.size());
    if (_size > 0) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                other.begin(), other.end(), begin());
    }
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    device_vector dummy(::std::move(other));
    this->swap(dummy);
    return *this;
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(_storage, 0); }
  iterator end() { return device_iterator<T>(_storage, size()); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(_storage, 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(_storage, size()); }
  const_iterator cend() const { return end(); }
  T *real_begin() { return _storage; }
  const T *real_begin() const { return _storage; }
  void swap(device_vector &v) {
    ::std::swap(_size, v._size);
    ::std::swap(_capacity, v._capacity);
    ::std::swap(_storage, v._storage);
    ::std::swap(_alloc, v._alloc);
  }
  reference operator[](size_type n) { return _storage[n]; }
  const_reference operator[](size_type n) const { return _storage[n]; }
  void reserve(size_type n) {
    if (n > capacity()) {
      // allocate buffer for new size
      auto tmp = _alloc.allocate(2 * n);
      // copy content (old buffer to new buffer)
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                begin(), end(), tmp);
      // deallocate old memory
      _alloc.deallocate(_storage, _capacity);
      _storage = tmp;
      _capacity = 2 * n;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (_size < new_size) {
      ::std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                begin() + _size, begin() + new_size, x);
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return ::std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const { return _capacity; }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return _storage; }
  const_pointer data(void) const { return _storage; }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      size_type tmp_capacity = ::std::max(_size, _min_capacity());
      auto tmp = _alloc.allocate(tmp_capacity);
      if (_size > 0) {
        ::std::copy(
            oneapi::dpl::execution::make_device_policy(get_default_queue()),
            begin(), end(), tmp);
      }
      _alloc.deallocate(_storage, _capacity);
      _storage = tmp;
      _capacity = tmp_capacity;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    if (_size > 0) {
      ::std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                begin(), begin() + n, x);
    }
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename ::std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = ::std::distance(first, last);
    resize(n);
    if (_size > 0) {
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                first, last, begin());
    }
  }
  void clear(void) { _size = 0; }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = ::std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    auto m = ::std::distance(last, end());
    if (m <= 0) {
      return end();
    }
    auto tmp = _alloc.allocate(m);
    // copy remainder to temporary buffer.
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              last, end(), tmp);
    // override (erase) subsequence in storage.
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              tmp, tmp + m, first);
    _alloc.deallocate(tmp, m);
    _size -= n;
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = ::std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      ::std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                end() - n, end(), x);
    } else {
      auto i_n = ::std::distance(begin(), position);
      // allocate temporary storage
      auto m = ::std::distance(position, end());
      // will throw if position is not inside active vector
      auto tmp = _alloc.allocate(m);
      // copy remainder
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                position, end(), tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      ::std::fill(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                position, position + n, x);

      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                tmp, tmp + m, position + n);
      _alloc.deallocate(tmp, m);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename ::std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = ::std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                first, last, end());
    } else {
      auto m = ::std::distance(position, end());
      // will throw if position is not inside active vector
      auto tmp = _alloc.allocate(m);

      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                position, end(), tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                first, last, position);
      ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                tmp, tmp + m, position + n);
      _alloc.deallocate(tmp, m);
    }
  }
  Allocator get_allocator() const { return _alloc; }
};

#else

template <typename T, typename Allocator = detail::__buffer_allocator<T>>
class device_vector {
  static_assert(
      std::is_same<Allocator, detail::__buffer_allocator<T>>::value,
      "device_vector doesn't support custom allocator when USM is not used.");

public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

private:
  using Buffer = sycl::buffer<T, 1>;
  using Range = sycl::range<1>;
  // Using mem_mgr to handle memory allocation
  void *_storage;
  size_type _size;

  size_type _min_capacity() const { return size_type(1); }

  void *alloc_store(size_type num_bytes) {
    return detail::mem_mgr::instance().mem_alloc(num_bytes);
  }

public:
  template <typename OtherA> operator std::vector<T, OtherA>() const {
    auto __tmp = std::vector<T, OtherA>(this->size());
    std::copy(oneapi::dpl::execution::dpcpp_default, this->begin(), this->end(),
              __tmp.begin());
    return __tmp;
  }
  device_vector()
      : _storage(alloc_store(_min_capacity() * sizeof(T))), _size(0) {}
  ~device_vector() = default;
  explicit device_vector(size_type n) : device_vector(n, T()) {}
  explicit device_vector(size_type n, const T &value)
      : _storage(alloc_store(std::max(n, _min_capacity()) * sizeof(T))),
        _size(n) {
    auto buf = get_buffer();
    std::fill(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf),
              oneapi::dpl::begin(buf) + n, T(value));
  }
  device_vector(const device_vector &other)
      : _storage(other._storage), _size(other.size()) {}
  device_vector(device_vector &&other)
      : _storage(std::move(other._storage)), _size(other.size()) {}

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !std::is_pointer<InputIterator>::value &&
                        std::is_same<typename std::iterator_traits<
                                         InputIterator>::iterator_category,
                                     std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              first, last, dst);
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<std::is_pointer<InputIterator>::value,
                                        InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    Buffer tmp_buf(first, last);
    auto start = oneapi::dpl::begin(tmp_buf);
    auto end = oneapi::dpl::end(tmp_buf);
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              start, end, dst);
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !std::is_same<typename std::iterator_traits<
                                          InputIterator>::iterator_category,
                                      std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    std::vector<T> tmp(first, last);
    Buffer tmp_buf(tmp);
    auto start = oneapi::dpl::begin(tmp_buf);
    auto end = oneapi::dpl::end(tmp_buf);
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              start, end, dst);
  }

  template <typename OtherAllocator>
  device_vector(const device_vector<T, OtherAllocator> &v)
      : _storage(alloc_store(v.size() * sizeof(T))), _size(v.size()) {
    auto buf = get_buffer();
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              v.real_begin(), v.real_begin() + v.size(), dst);
  }

  template <typename OtherAllocator>
  device_vector(std::vector<T, OtherAllocator> &v)
      : _storage(alloc_store(v.size() * sizeof(T))), _size(v.size()) {
    std::copy(oneapi::dpl::execution::dpcpp_default, v.begin(), v.end(),
              oneapi::dpl::begin(get_buffer()));
  }

  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    _size = other.size();
    void *tmp = alloc_store(_size * sizeof(T));
    auto tmp_buf =
        detail::mem_mgr::instance()
            .translate_ptr(tmp)
            .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
    std::copy(oneapi::dpl::execution::dpcpp_default,
              oneapi::dpl::begin(other.get_buffer()),
              oneapi::dpl::end(other.get_buffer()),
              oneapi::dpl::begin(tmp_buf));
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = tmp;
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    _size = other.size();
    this->_storage = std::move(other._storage);
    return *this;
  }
  template <typename OtherAllocator>
  device_vector &operator=(const std::vector<T, OtherAllocator> &v) {
    Buffer data(v.begin(), v.end());
    _size = v.size();
    void *tmp = alloc_store(_size * sizeof(T));
    auto tmp_buf =
        detail::mem_mgr::instance()
            .translate_ptr(tmp)
            .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
    std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(data),
              oneapi::dpl::end(data), oneapi::dpl::begin(tmp_buf));
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = tmp;

    return *this;
  }
  Buffer get_buffer() const {
    return detail::mem_mgr::instance()
        .translate_ptr(_storage)
        .buffer.template reinterpret<T, 1>(sycl::range<1>(capacity()));
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(get_buffer(), 0); }
  iterator end() { return device_iterator<T>(get_buffer(), _size); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(get_buffer(), 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(get_buffer(), _size); }
  const_iterator cend() const { return end(); }
  T *real_begin() {
    return (detail::mem_mgr::instance()
                .translate_ptr(_storage)
                .buffer.template get_access<sycl::access_mode::read_write>())
        .get_pointer();
  }
  const T *real_begin() const {
    return const_cast<device_vector *>(this)
        ->detail::mem_mgr::instance()
        .translate_ptr(_storage)
        .buffer.template get_access<sycl::access_mode::read_write>()
        .get_pointer();
  }
  void swap(device_vector &v) {
    void *temp = v._storage;
    v._storage = this->_storage;
    this->_storage = temp;
    std::swap(_size, v._size);
  }
  reference operator[](size_type n) { return *(begin() + n); }
  const_reference operator[](size_type n) const { return *(begin() + n); }
  void reserve(size_type n) {
    if (n > capacity()) {
      // create new buffer (allocate for new size)
      void *a = alloc_store(n * sizeof(T));

      // copy content (old buffer to new buffer)
      if (_storage != nullptr) {
        auto tmp = detail::mem_mgr::instance()
                       .translate_ptr(a)
                       .buffer.template reinterpret<T, 1>(sycl::range<1>(n));
        auto src_buf = get_buffer();
        std::copy(oneapi::dpl::execution::dpcpp_default,
                  oneapi::dpl::begin(src_buf), oneapi::dpl::end(src_buf),
                  oneapi::dpl::begin(tmp));

        // deallocate old memory
        detail::mem_mgr::instance().mem_free(_storage);
      }
      _storage = a;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (_size < new_size) {
      auto src_buf = get_buffer();
      std::fill(oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::begin(src_buf) + _size,
                oneapi::dpl::begin(src_buf) + new_size, x);
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const {
    return _storage != nullptr ? detail::mem_mgr::instance()
                                         .translate_ptr(_storage)
                                         .buffer.size() /
                                     sizeof(T)
                               : 0;
  }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return reinterpret_cast<pointer>(_storage); }
  const_pointer data(void) const {
    return reinterpret_cast<const_pointer>(_storage);
  }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      void *a = alloc_store(_size * sizeof(T));
      auto tmp = detail::mem_mgr::instance()
                     .translate_ptr(a)
                     .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
      std::copy(oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::begin(get_buffer()),
                oneapi::dpl::begin(get_buffer()) + _size,
                oneapi::dpl::begin(tmp));
      detail::mem_mgr::instance().mem_free(_storage);
      _storage = a;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    std::fill(oneapi::dpl::execution::dpcpp_default, begin(), begin() + n, x);
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    resize(n);
    if (internal::is_iterator<InputIterator>::value &&
        !std::is_pointer<InputIterator>::value)
      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, begin());
    else {
      Buffer tmp(first, last);
      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), begin());
    }
  }
  void clear(void) {
    _size = 0;
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = nullptr;
  }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    Buffer tmp{Range(std::distance(last, end()))};
    // copy remainder to temporary buffer.
    std::copy(oneapi::dpl::execution::dpcpp_default, last, end(),
              oneapi::dpl::begin(tmp));
    // override (erase) subsequence in storage.
    std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
              oneapi::dpl::end(tmp), first);
    resize(_size - n);
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      std::fill(oneapi::dpl::execution::dpcpp_default, end() - n, end(), x);
    } else {
      auto i_n = std::distance(begin(), position);
      // allocate temporary storage
      Buffer tmp{Range(std::distance(position, end()))};
      // copy remainder
      std::copy(oneapi::dpl::execution::dpcpp_default, position, end(),
                oneapi::dpl::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::fill(oneapi::dpl::execution::dpcpp_default, position, position + n,
                x);

      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), position + n);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, end());
    } else {
      Buffer tmp{Range(std::distance(position, end()))};

      std::copy(oneapi::dpl::execution::dpcpp_default, position, end(),
                oneapi::dpl::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, position);
      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), position + n);
    }
  }
};

#endif

} // end namespace dpct

#endif
