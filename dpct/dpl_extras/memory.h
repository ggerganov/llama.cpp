//==---- memory.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MEMORY_H__
#define __DPCT_MEMORY_H__

#include <sycl/sycl.hpp>
#include <oneapi/dpl/memory>
#include "functional.h"

// Memory management section:
// device_pointer, device_reference, swap, device_iterator, malloc_device,
// device_new, free_device, device_delete
namespace dpct {
namespace detail {
template <typename T>
struct make_allocatable
{
  using type = T;
};

template <>
struct make_allocatable<void>
{
  using type = dpct::byte_t;
};

#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) &&    \
    defined(__LIBSYCL_PATCH_VERSION)
#define _DPCT_LIBSYCL_VERSION                                                  \
  (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 +           \
   __LIBSYCL_PATCH_VERSION)
#else
#define _DPCT_LIBSYCL_VERSION 0
#endif

template <typename _DataT>
using __buffer_allocator =
#if _DPCT_LIBSYCL_VERSION >= 60000
    sycl::buffer_allocator<typename make_allocatable<_DataT>::type>;
#else
    sycl::buffer_allocator;
#endif
} // namespace detail

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access_mode Mode = sycl::access_mode::read_write,
          typename Allocator = detail::__buffer_allocator<T>>
class device_pointer;
#else
template <typename T> class device_pointer;
#endif

template <typename T> struct device_reference {
  using pointer = device_pointer<T>;
  using value_type = T;
  template <typename OtherT>
  device_reference(const device_reference<OtherT> &input)
      : value(input.value) {}
  device_reference(const pointer &input) : value((*input).value) {}
  device_reference(value_type &input) : value(input) {}
  template <typename OtherT>
  device_reference &operator=(const device_reference<OtherT> &input) {
    value = input;
    return *this;
  };
  device_reference &operator=(const device_reference &input) {
    T val = input.value;
    value = val;
    return *this;
  };
  device_reference &operator=(const value_type &x) {
    value = x;
    return *this;
  };
  pointer operator&() const { return pointer(&value); };
  operator value_type() const { return T(value); }
  device_reference &operator++() {
    ++value;
    return *this;
  };
  device_reference &operator--() {
    --value;
    return *this;
  };
  device_reference operator++(int) {
    device_reference ref(*this);
    ++(*this);
    return ref;
  };
  device_reference operator--(int) {
    device_reference ref(*this);
    --(*this);
    return ref;
  };
  device_reference &operator+=(const T &input) {
    value += input;
    return *this;
  };
  device_reference &operator-=(const T &input) {
    value -= input;
    return *this;
  };
  device_reference &operator*=(const T &input) {
    value *= input;
    return *this;
  };
  device_reference &operator/=(const T &input) {
    value /= input;
    return *this;
  };
  device_reference &operator%=(const T &input) {
    value %= input;
    return *this;
  };
  device_reference &operator&=(const T &input) {
    value &= input;
    return *this;
  };
  device_reference &operator|=(const T &input) {
    value |= input;
    return *this;
  };
  device_reference &operator^=(const T &input) {
    value ^= input;
    return *this;
  };
  device_reference &operator<<=(const T &input) {
    value <<= input;
    return *this;
  };
  device_reference &operator>>=(const T &input) {
    value >>= input;
    return *this;
  };
  void swap(device_reference &input) {
    T tmp = (*this);
    *this = (input);
    input = (tmp);
  }
  T &value;
};

template <typename T>
void swap(device_reference<T> &x, device_reference<T> &y) {
  x.swap(y);
}

template <typename T> void swap(T &x, T &y) {
  T tmp = x;
  x = y;
  y = tmp;
}

template <typename T>
::std::ostream &operator<<(::std::ostream &out,
                           const device_reference<T> &ref) {
  return out << T(ref);
}

namespace internal {
// struct for checking if iterator is heterogeneous or not
template <typename Iter,
          typename Void = void> // for non-heterogeneous iterators
struct is_hetero_iterator : std::false_type {};

template <typename Iter> // for heterogeneous iterators
struct is_hetero_iterator<
    Iter, typename std::enable_if<Iter::is_hetero::value, void>::type>
    : std::true_type {};
} // namespace internal

#ifdef DPCT_USM_LEVEL_NONE
// Must be forward declared due to default argument
template <typename T>
device_pointer<T> device_new(device_pointer<void>, const T &,
                             const std::size_t = 1);

template <typename T, sycl::access_mode Mode, typename Allocator>
class device_iterator;

template <typename ValueType, typename Allocator, typename Derived>
class device_pointer_base {
protected:
  sycl::buffer<ValueType, 1, Allocator> buffer;
  std::size_t idx;

  // Declare friend to give access to protected buffer and idx members
  template <typename T>
  friend device_pointer<T> device_new(device_pointer<void>, const T &,
                                      const std::size_t);

public:
  using pointer = ValueType *;
  using difference_type = std::make_signed<std::size_t>::type;

  device_pointer_base(sycl::buffer<ValueType, 1> in, std::size_t i = 0)
      : buffer(in), idx(i) {}
#ifdef __USE_DPCT
  template <typename OtherT>
  device_pointer_base(OtherT *ptr)
      : buffer(
            dpct::detail::mem_mgr::instance()
                .translate_ptr(ptr)
                .buffer.template reinterpret<ValueType, 1>(sycl::range<1>(
                    dpct::detail::mem_mgr::instance().translate_ptr(ptr).size /
                    sizeof(ValueType)))),
        idx(ptr - (ValueType*)dpct::detail::mem_mgr::instance()
                .translate_ptr(ptr).alloc_ptr) {}
#endif
  device_pointer_base(const std::size_t count)
      : buffer(sycl::range<1>(count / sizeof(ValueType))), idx() {}
  // buffer has no default ctor we pass zero-range to create an empty buffer
  device_pointer_base() : buffer(sycl::range<1>(0)) {}
  device_pointer_base(const device_pointer_base &in)
      : buffer(in.buffer), idx(in.idx) {}
  pointer get() const {
    auto res =
        (const_cast<device_pointer_base *>(this)
             ->buffer.template get_access<sycl::access_mode::read_write>())
            .get_pointer();
    return res + idx;
  }
  operator ValueType *() {
    auto res = (buffer.template get_access<sycl::access_mode::read_write>())
                   .get_pointer();
    return res + idx;
  }
  operator ValueType *() const {
    auto res =
        (const_cast<device_pointer_base *>(this)
             ->buffer.template get_access<sycl::access_mode::read_write>())
            .get_pointer();
    return res + idx;
  }
  Derived operator+(difference_type forward) const {
    return Derived{buffer, idx + forward};
  }
  Derived operator-(difference_type backward) const {
    return Derived{buffer, idx - backward};
  }
  Derived operator++(int) {
    Derived p(buffer, idx);
    idx += 1;
    return p;
  }
  Derived operator--(int) {
    Derived p(buffer, idx);
    idx -= 1;
    return p;
  }
  difference_type operator-(const Derived &it) const { return idx - it.idx; }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - std::distance(oneapi::dpl::begin(buffer), it);
  }

  std::size_t get_idx() const { return idx; } // required

  sycl::buffer<ValueType, 1, Allocator> get_buffer() {
    return buffer;
  } // required
};

template <sycl::access_mode Mode, typename Allocator>
class device_pointer<void, Mode, Allocator>
    : public device_pointer_base<dpct::byte_t, Allocator,
                                 device_pointer<void, Mode, Allocator>> {
private:
  using base_type =
      device_pointer_base<dpct::byte_t, Allocator, device_pointer>;

public:
  using value_type = dpct::byte_t;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = void *;
  using reference = value_type &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type; // required
  using is_passed_directly = std::false_type;
  static constexpr sycl::access_mode mode = Mode; // required

  device_pointer(sycl::buffer<value_type, 1> in, std::size_t i = 0)
      : base_type(in, i) {}
#ifdef __USE_DPCT
  template <typename OtherT> device_pointer(OtherT *ptr) : base_type(ptr) {}
#endif
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer(const device_pointer &in) : base_type(in) {}
  device_pointer &operator+=(difference_type forward) {
    this->idx += forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->idx -= backward;
    return *this;
  }
  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    this->idx += 1;
    return *this;
  }
  device_pointer &operator--() {
    this->idx -= 1;
    return *this;
  }
};

template <typename T, sycl::access_mode Mode, typename Allocator>
class device_pointer
    : public device_pointer_base<T, Allocator,
                                 device_pointer<T, Mode, Allocator>> {
private:
  using base_type = device_pointer_base<T, Allocator, device_pointer>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type; // required
  using is_passed_directly = std::false_type;
  static constexpr sycl::access_mode mode = Mode; // required

  device_pointer(sycl::buffer<T, 1> in, std::size_t i = 0) : base_type(in, i) {}
#ifdef __USE_DPCT
  template <typename OtherT> device_pointer(OtherT *ptr) : base_type(ptr) {}
#endif
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer(const device_pointer &in) : base_type(in) {}
  device_pointer &operator+=(difference_type forward) {
    this->idx += forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->idx -= backward;
    return *this;
  }
  operator device_pointer<void>() {
    auto converted_buf = (this->buffer)
                             .template reinterpret<dpct::byte_t>(sycl::range<1>(
                                 sizeof(value_type) * this->buffer.size()));
    return device_pointer<void>(converted_buf, this->idx);
  }
  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    this->idx += 1;
    return *this;
  }
  device_pointer &operator--() {
    this->idx -= 1;
    return *this;
  }
};
#else
template <typename T> class device_iterator;

template <typename ValueType, typename Derived> class device_pointer_base {
protected:
  ValueType *ptr;

public:
  using pointer = ValueType *;
  using difference_type = std::make_signed<std::size_t>::type;

  device_pointer_base(ValueType *p) : ptr(p) {}
  device_pointer_base(const std::size_t count) {
    sycl::queue default_queue = dpct::get_default_queue();
    ptr = static_cast<ValueType *>(sycl::malloc_shared(
        count, default_queue.get_device(), default_queue.get_context()));
  }
  device_pointer_base() {}
  pointer get() const { return ptr; }
  operator ValueType *() { return ptr; }
  operator ValueType *() const { return ptr; }

  ValueType &operator[](difference_type idx) { return ptr[idx]; }
  ValueType &operator[](difference_type idx) const { return ptr[idx]; }

  Derived operator+(difference_type forward) const {
    return Derived{ptr + forward};
  }
  Derived operator-(difference_type backward) const {
    return Derived{ptr - backward};
  }
  Derived operator++(int) {
    Derived p(ptr);
    ++ptr;
    return p;
  }
  Derived operator--(int) {
    Derived p(ptr);
    --ptr;
    return p;
  }
  difference_type operator-(const Derived &it) const { return ptr - it.ptr; }
};

template <>
class device_pointer<void>
    : public device_pointer_base<dpct::byte_t, device_pointer<void>> {
private:
  using base_type = device_pointer_base<dpct::byte_t, device_pointer<void>>;

public:
  using value_type = dpct::byte_t;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = void *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(void *p) : base_type(static_cast<value_type *>(p)) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  pointer get() const { return static_cast<pointer>(this->ptr); }
  operator void *() { return this->ptr; }
  operator void *() const { return this->ptr; }

  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    ++(this->ptr);
    return *this;
  }
  device_pointer &operator--() {
    --(this->ptr);
    return *this;
  }
  device_pointer &operator+=(difference_type forward) {
    this->ptr = this->ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->ptr = this->ptr - backward;
    return *this;
  }
};

template <typename T>
class device_pointer : public device_pointer_base<T, device_pointer<T>> {
private:
  using base_type = device_pointer_base<T, device_pointer<T>>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required

  device_pointer(T *p) : base_type(p) {}
  // needed for malloc_device, count is number of bytes to allocate
  device_pointer(const std::size_t count) : base_type(count) {}
  device_pointer() : base_type() {}
  device_pointer &operator=(const device_iterator<T> &in) {
    this->ptr = static_cast<device_pointer<T>>(in).ptr;
    return *this;
  }
  operator device_pointer<void>() {
    return device_pointer<void>(static_cast<void *>(this->ptr));
  }
  // include operators from base class
  using base_type::operator++;
  using base_type::operator--;
  device_pointer &operator++() {
    ++(this->ptr);
    return *this;
  }
  device_pointer &operator--() {
    --(this->ptr);
    return *this;
  }
  device_pointer &operator+=(difference_type forward) {
    this->ptr = this->ptr + forward;
    return *this;
  }
  device_pointer &operator-=(difference_type backward) {
    this->ptr = this->ptr - backward;
    return *this;
  }
};
#endif

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, sycl::access_mode Mode = sycl::access_mode::read_write,
          typename Allocator = detail::__buffer_allocator<T>>
class device_iterator : public device_pointer<T, Mode, Allocator> {
  using Base = device_pointer<T, Mode, Allocator>;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::true_type;                // required
  using is_passed_directly = std::false_type;      // required
  static constexpr sycl::access_mode mode = Mode; // required

  device_iterator() : Base() {}
  device_iterator(sycl::buffer<T, 1, Allocator> vec, std::size_t index)
      : Base(vec, index) {}
  device_iterator(const Base &dev_ptr) : Base(dev_ptr) {}
  template <sycl::access_mode inMode>
  device_iterator(const device_iterator<T, inMode, Allocator> &in)
      : Base(in.buffer, in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::buffer = in.buffer;
    Base::idx = in.idx;
    return *this;
  }

  reference operator*() const {
    return const_cast<device_iterator *>(this)
        ->buffer.template get_access<mode>()[Base::idx];
  }

  reference operator[](difference_type i) const { return *(*this + i); }
  device_iterator &operator++() {
    ++Base::idx;
    return *this;
  }
  device_iterator &operator--() {
    --Base::idx;
    return *this;
  }
  device_iterator operator++(int) {
    device_iterator it(*this);
    ++(*this);
    return it;
  }
  device_iterator operator--(int) {
    device_iterator it(*this);
    --(*this);
    return it;
  }
  device_iterator operator+(difference_type forward) const {
    const auto new_idx = Base::idx + forward;
    return {Base::buffer, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    Base::idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::buffer, Base::idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    Base::idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return Base::idx - it.idx;
  }
  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return Base::idx - std::distance(oneapi::dpl::begin(Base::buffer), it);
  }
  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() const { return Base::idx; } // required

  sycl::buffer<T, 1, Allocator> get_buffer() {
    return Base::buffer;
  } // required
};
#else
template <typename T> class device_iterator : public device_pointer<T> {
  using Base = device_pointer<T>;

protected:
  std::size_t idx;

public:
  using value_type = T;
  using difference_type = std::make_signed<std::size_t>::type;
  using pointer = typename Base::pointer;
  using reference = typename Base::reference;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = std::false_type;         // required
  using is_passed_directly = std::true_type; // required
  static constexpr sycl::access_mode mode =
      sycl::access_mode::read_write; // required

  device_iterator() : Base(nullptr), idx(0) {}
  device_iterator(T *vec, std::size_t index) : Base(vec), idx(index) {}
  device_iterator(const Base &dev_ptr) : Base(dev_ptr), idx(0) {}
  template <sycl::access_mode inMode>
  device_iterator(const device_iterator<T> &in)
      : Base(in.ptr), idx(in.idx) {} // required for iter_mode
  device_iterator &operator=(const device_iterator &in) {
    Base::operator=(in);
    idx = in.idx;
    return *this;
  }

  reference operator*() const { return *(Base::ptr + idx); }

  reference operator[](difference_type i) { return Base::ptr[idx + i]; }
  reference operator[](difference_type i) const { return Base::ptr[idx + i]; }
  device_iterator &operator++() {
    ++idx;
    return *this;
  }
  device_iterator &operator--() {
    --idx;
    return *this;
  }
  device_iterator operator++(int) {
    device_iterator it(*this);
    ++(*this);
    return it;
  }
  device_iterator operator--(int) {
    device_iterator it(*this);
    --(*this);
    return it;
  }
  device_iterator operator+(difference_type forward) const {
    const auto new_idx = idx + forward;
    return {Base::ptr, new_idx};
  }
  device_iterator &operator+=(difference_type forward) {
    idx += forward;
    return *this;
  }
  device_iterator operator-(difference_type backward) const {
    return {Base::ptr, idx - backward};
  }
  device_iterator &operator-=(difference_type backward) {
    idx -= backward;
    return *this;
  }
  friend device_iterator operator+(difference_type forward,
                                   const device_iterator &it) {
    return it + forward;
  }
  difference_type operator-(const device_iterator &it) const {
    return idx - it.idx;
  }

  template <typename OtherIterator>
  typename std::enable_if<internal::is_hetero_iterator<OtherIterator>::value,
                          difference_type>::type
  operator-(const OtherIterator &it) const {
    return idx - it.get_idx();
  }

  bool operator==(const device_iterator &it) const { return *this - it == 0; }
  bool operator!=(const device_iterator &it) const { return !(*this == it); }
  bool operator<(const device_iterator &it) const { return *this - it < 0; }
  bool operator>(const device_iterator &it) const { return it < *this; }
  bool operator<=(const device_iterator &it) const { return !(*this > it); }
  bool operator>=(const device_iterator &it) const { return !(*this < it); }

  std::size_t get_idx() const { return idx; } // required

  device_iterator &get_buffer() { return *this; } // required

  std::size_t size() const { return idx; }
};
#endif

struct sys_tag {};
struct device_sys_tag : public sys_tag {};
struct host_sys_tag : public sys_tag {};

#ifdef DPCT_USM_LEVEL_NONE
template <typename T, typename Tag> class tagged_pointer {
  static_assert(false,
                "tagged_pointer is not supported with DPCT_USM_LEVEL_NONE");
};
template <typename PolicyOrTag, typename Pointer>
void release_temporary_allocation(PolicyOrTag &&policy_or_tag, Pointer ptr) {
  static_assert(
      false,
      "release_temporary_allocation is not supported with DPCT_USM_LEVEL_NONE");
}
template <typename T, typename PolicyOrTag, typename SizeType>
auto get_temporary_allocation(PolicyOrTag &&policy_or_tag,
                              SizeType num_elements) {
  static_assert(
      false,
      "get_temporary_allocation is not supported with DPCT_USM_LEVEL_NONE");
}
template <typename PolicyOrTag>
auto malloc(PolicyOrTag &&policy_or_tag, const ::std::size_t num_bytes) {
  static_assert(false, "malloc is not supported with DPCT_USM_LEVEL_NONE");
}
template <typename T, typename PolicyOrTag>
auto malloc(PolicyOrTag &&policy_or_tag, const ::std::size_t num_elements) {
  static_assert(false, "malloc<T> is not supported with DPCT_USM_LEVEL_NONE");
}
template <typename PolicyOrTag, typename Pointer>
void free(PolicyOrTag &&policy_or_tag, Pointer ptr) {
  static_assert(false, "free is not supported with DPCT_USM_LEVEL_NONE");
}
#else
namespace internal {

// Utility that converts a policy to a tag or reflects a provided tag
template <typename PolicyOrTag> struct policy_or_tag_to_tag {
private:
  using decayed_policy_or_tag_t = ::std::decay_t<PolicyOrTag>;
  using policy_conversion = ::std::conditional_t<
      !is_hetero_execution_policy<decayed_policy_or_tag_t>::value, host_sys_tag,
      device_sys_tag>;
  static constexpr bool is_policy_v =
      oneapi::dpl::execution::is_execution_policy_v<decayed_policy_or_tag_t>;
  static constexpr bool is_sys_tag_v = ::std::disjunction_v<
      ::std::is_same<decayed_policy_or_tag_t, host_sys_tag>,
      ::std::is_same<decayed_policy_or_tag_t, device_sys_tag>>;
  static_assert(is_policy_v || is_sys_tag_v,
                "Only oneDPL policies or system tags may be provided");

public:
  using type = ::std::conditional_t<is_policy_v, policy_conversion,
                                    decayed_policy_or_tag_t>;
};

template <typename PolicyOrTag>
using policy_or_tag_to_tag_t = typename policy_or_tag_to_tag<PolicyOrTag>::type;

template <typename PolicyOrTag> struct is_host_policy_or_tag {
private:
  using tag_t = policy_or_tag_to_tag_t<PolicyOrTag>;

public:
  static constexpr bool value = ::std::is_same_v<tag_t, host_sys_tag>;
};

template <typename PolicyOrTag>
inline constexpr bool is_host_policy_or_tag_v =
    is_host_policy_or_tag<PolicyOrTag>::value;

} // namespace internal

// TODO: Make this class an iterator adaptor.
// tagged_pointer provides a wrapper around a raw pointer type with a tag of the
// location of the allocated memory. Standard pointer operations are supported
// with this class.
template <typename T, typename Tag> class tagged_pointer {
public:
  using value_type = T;
  using difference_type = ::std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;
  using is_hetero = ::std::false_type;
  using is_passed_directly = std::true_type;

  tagged_pointer() : m_ptr(nullptr) {}
  tagged_pointer(T *ptr) : m_ptr(ptr) {}
  T &operator[](difference_type idx) { return this->m_ptr[idx]; }
  const T &operator[](difference_type idx) const { return this->m_ptr[idx]; }
  tagged_pointer operator+(difference_type forward) const {
    return tagged_pointer{this->m_ptr + forward};
  }
  tagged_pointer operator-(difference_type backward) const {
    return tagged_pointer{this->m_ptr - backward};
  }
  operator const T *() const { return m_ptr; }
  operator T *() { return m_ptr; }
  T &operator*() { return *this->m_ptr; }
  const T &operator*() const { return *this->m_ptr; }
  T *operator->() { return this->m_ptr; }
  const T *operator->() const { return this->m_ptr; }
  tagged_pointer operator++(int) {
    tagged_pointer p(this->m_ptr);
    ++this->m_ptr;
    return p;
  }
  tagged_pointer operator--(int) {
    tagged_pointer p(this->m_ptr);
    --this->m_ptr;
    return p;
  }
  tagged_pointer &operator++() {
    ++this->m_ptr;
    return *this;
  }
  tagged_pointer &operator--() {
    --this->m_ptr;
    return *this;
  }
  difference_type operator-(const tagged_pointer &it) const {
    return this->m_ptr - it.m_ptr;
  }
  tagged_pointer &operator+=(difference_type forward) {
    this->m_ptr = this->m_ptr + forward;
    return *this;
  }
  tagged_pointer &operator-=(difference_type backward) {
    this->m_ptr = this->m_ptr - backward;
    return *this;
  }

private:
  T *m_ptr;
};

// Void specialization for tagged pointers. Iterator traits are not provided but
// conversion to other non-void tagged pointers is allowed. Pointer arithmetic
// is disallowed with this specialization.
template <typename Tag> class tagged_pointer<void, Tag> {
public:
  using difference_type = ::std::ptrdiff_t;
  using pointer = void *;
  tagged_pointer() : m_ptr(nullptr) {}
  tagged_pointer(pointer ptr) : m_ptr(ptr) {}
  operator const void *() const { return m_ptr; }
  operator void *() { return m_ptr; }
  // Enable tagged void pointer to convert to all other raw pointer types.
  template <typename OtherPtr> operator OtherPtr *() const {
    return static_cast<OtherPtr *>(this->m_ptr);
  }

private:
  void *m_ptr;
};

namespace internal {

// Internal utility to return raw pointer to allocated memory. Note that host
// allocations are not device accessible (not pinned).
template <typename PolicyOrTag>
void *malloc_base(PolicyOrTag &&policy_or_tag, const ::std::size_t num_bytes) {
  using decayed_policy_or_tag_t = ::std::decay_t<PolicyOrTag>;
  if constexpr (internal::is_host_policy_or_tag_v<PolicyOrTag>) {
    return ::std::malloc(num_bytes);
  } else {
    sycl::queue q;
    // Grab the associated queue if a device policy is provided. Otherwise, use
    // default constructed.
    if constexpr (oneapi::dpl::execution::is_execution_policy_v<
                      decayed_policy_or_tag_t>) {
      q = policy_or_tag.queue();
    } else {
      q = get_default_queue();
    }
    return sycl::malloc_shared(num_bytes, q);
  }
}

} // namespace internal

template <typename PolicyOrTag>
auto malloc(PolicyOrTag &&policy_or_tag, const ::std::size_t num_bytes) {
  return tagged_pointer<void, internal::policy_or_tag_to_tag_t<PolicyOrTag>>(
      internal::malloc_base(::std::forward<PolicyOrTag>(policy_or_tag),
                            num_bytes));
}

template <typename T, typename PolicyOrTag>
auto malloc(PolicyOrTag &&policy_or_tag, const ::std::size_t num_elements) {
  return tagged_pointer<T, internal::policy_or_tag_to_tag_t<PolicyOrTag>>(
      static_cast<T *>(
          internal::malloc_base(::std::forward<PolicyOrTag>(policy_or_tag),
                                num_elements * sizeof(T))));
}

template <typename PolicyOrTag, typename Pointer>
void free(PolicyOrTag &&policy_or_tag, Pointer ptr) {
  using decayed_policy_or_tag_t = ::std::decay_t<PolicyOrTag>;
  if constexpr (internal::is_host_policy_or_tag_v<PolicyOrTag>) {
    ::std::free(ptr);
  } else {
    sycl::queue q;
    // Grab the associated queue if a device policy is provided. Otherwise, use
    // default constructed.
    if constexpr (oneapi::dpl::execution::is_execution_policy_v<
                      decayed_policy_or_tag_t>) {
      q = policy_or_tag.queue();
    } else {
      q = get_default_queue();
    }
    sycl::free(ptr, q);
  }
}

template <typename T, typename PolicyOrTag, typename SizeType>
auto get_temporary_allocation(PolicyOrTag &&policy_or_tag,
                              SizeType num_elements) {
  auto allocation_ptr =
      dpct::malloc<T>(::std::forward<PolicyOrTag>(policy_or_tag), num_elements);
  if (allocation_ptr == nullptr)
    return ::std::make_pair(allocation_ptr, SizeType(0));
  return ::std::make_pair(allocation_ptr, num_elements);
}

template <typename PolicyOrTag, typename Pointer>
void release_temporary_allocation(PolicyOrTag &&policy_or_tag, Pointer ptr) {
  dpct::free(::std::forward<PolicyOrTag>(policy_or_tag), ptr);
}
#endif

template <typename T>
device_pointer<T> malloc_device(const std::size_t num_elements) {
  return device_pointer<T>(num_elements * sizeof(T));
}
static inline device_pointer<void> malloc_device(const std::size_t num_bytes) {
  return device_pointer<void>(num_bytes);
}
#ifdef DPCT_USM_LEVEL_NONE
template <typename T>
device_pointer<T> device_new(device_pointer<void> p, const T &value,
                             const std::size_t count) {
  auto converted_buf = p.buffer.template reinterpret<T>(sycl::range<1>(count));
  ::std::uninitialized_fill(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      oneapi::dpl::begin(converted_buf),
      oneapi::dpl::end(converted_buf), value);
  return device_pointer<T>(converted_buf, p.idx);
}
// buffer manages lifetime
template <typename T> void free_device(device_pointer<T> ptr) {}
#else
template <typename T>
device_pointer<T> device_new(device_pointer<void> p, const T &value,
                             const std::size_t count = 1) {
  dpct::device_pointer<T> converted_p(static_cast<T *>(p.get()));
  ::std::uninitialized_fill(
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      converted_p, converted_p + count, value);
  return converted_p;
}
template <typename T> void free_device(device_pointer<T> ptr) {
  sycl::free(ptr.get(), dpct::get_default_queue());
}
#endif
template <typename T>
device_pointer<T> device_new(device_pointer<void> p,
                             const std::size_t count = 1) {
  return device_new(p, T{}, count);
}
template <typename T>
device_pointer<T> device_new(const std::size_t count = 1) {
  return device_new(device_pointer<void>(sizeof(T) * count), T{}, count);
}

template <typename T>
typename std::enable_if<!std::is_trivially_destructible<T>::value, void>::type
device_delete(device_pointer<T> p, const std::size_t count = 1) {
  ::std::destroy(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                 p, p + count);
  free_device(p);
}
template <typename T>
typename std::enable_if<std::is_trivially_destructible<T>::value, void>::type
device_delete(device_pointer<T> p, const std::size_t count = 1) {
  free_device(p);
}

template <typename T> device_pointer<T> get_device_pointer(T *ptr) {
  return device_pointer<T>(ptr);
}

template <typename T>
device_pointer<T> get_device_pointer(const device_pointer<T> &ptr) {
  return device_pointer<T>(ptr);
}

template <typename T> T *get_raw_pointer(const device_pointer<T> &ptr) {
  return ptr.get();
}

template <typename Pointer> Pointer get_raw_pointer(const Pointer &ptr) {
  return ptr;
}

template <typename T> const T &get_raw_reference(const device_reference<T> &ref) {
  return ref.value;
}

template <typename T> T &get_raw_reference(device_reference<T> &ref) {
  return ref.value;
}

template <typename T> const T &get_raw_reference(const T &ref) {
  return ref;
}

template <typename T> T &get_raw_reference(T &ref) {
  return ref;
}

} // namespace dpct

#endif
