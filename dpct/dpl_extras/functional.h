//==---- functional.h -----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_FUNCTIONAL_H__
#define __DPCT_FUNCTIONAL_H__

#include <functional>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/iterator>

#if ONEDPL_USE_DPCPP_BACKEND
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#endif

#include <tuple>
#include <utility>

#include "../dpct.hpp"
#define _DPCT_GCC_VERSION                                                      \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

// Portability "#pragma" definition
#ifdef _MSC_VER
#define _DPCT_PRAGMA(x) __pragma(x)
#else
#define _DPCT_PRAGMA(x) _Pragma(#x)
#endif

// Enable loop unrolling pragmas where supported
#if (__INTEL_COMPILER ||                                                       \
     (!defined(__INTEL_COMPILER) && _DPCT_GCC_VERSION >= 80000))
#define _DPCT_PRAGMA_UNROLL _DPCT_PRAGMA(unroll)
#else // no pragma unroll
#define _DPCT_PRAGMA_UNROLL
#endif

namespace dpct {

struct null_type {};

// Function object to wrap user defined functors to provide compile time "const"
// workaround for user function objects.
// The SYCL spec (4.12) states that writing to a function object during a SYCL
// kernel is undefined behavior.  This wrapper is provided as a compile-time
// work around, but functors used in SYCL kernels must be `const` in practice.
template <typename _Op> struct mark_functor_const {
  mutable _Op op;
  mark_functor_const() : op() {}
  mark_functor_const(const _Op &__op) : op(__op) {}
  mark_functor_const(_Op &&__op) : op(::std::move(__op)) {}
  template <typename... _T> auto operator()(_T &&...x) const {
    return op(std::forward<_T>(x)...);
  }
};

namespace internal {

template <class _ExecPolicy, class _T>
using enable_if_execution_policy =
    typename std::enable_if<oneapi::dpl::execution::is_execution_policy<
                                typename std::decay<_ExecPolicy>::type>::value,
                            _T>::type;

template <typename _T> struct is_hetero_execution_policy : ::std::false_type {};

template <typename... PolicyParams>
struct is_hetero_execution_policy<
    oneapi::dpl::execution::device_policy<PolicyParams...>> : ::std::true_type {
};

template <typename _T> struct is_fpga_execution_policy : ::std::false_type {};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int unroll_factor, typename... PolicyParams>
struct is_hetero_execution_policy<
    execution::fpga_policy<unroll_factor, PolicyParams...>> : ::std::true_type {
};
#endif

template <class _ExecPolicy, class _T>
using enable_if_hetero_execution_policy = typename std::enable_if<
    is_hetero_execution_policy<typename std::decay<_ExecPolicy>::type>::value,
    _T>::type;

#if _ONEDPL_CPP14_INTEGER_SEQUENCE_PRESENT

template <std::size_t... _Sp>
using index_sequence = std::index_sequence<_Sp...>;
template <std::size_t _Np>
using make_index_sequence = std::make_index_sequence<_Np>;

#else

template <std::size_t... _Sp> class index_sequence {};

template <std::size_t _Np, std::size_t... _Sp>
struct make_index_sequence_impl
    : make_index_sequence_impl<_Np - 1, _Np - 1, _Sp...> {};

template <std::size_t... _Sp> struct make_index_sequence_impl<0, _Sp...> {
  using type = index_sequence<_Sp...>;
};

template <std::size_t _Np>
using make_index_sequence = typename make_index_sequence_impl<_Np>::type;
#endif

// Minimal buffer implementations for temporary storage in mapping rules
// Some of our algorithms need to start with raw memory buffer,
// not an initialized array, because initialization/destruction
// would make the span be at least O(N).
#if ONEDPL_USE_DPCPP_BACKEND
template <typename _Tp> class __buffer {
  sycl::buffer<_Tp, 1> __buf;

  __buffer(const __buffer &) = delete;

  void operator=(const __buffer &) = delete;

public:
  // Try to obtain buffer of given size to store objects of _Tp type
  __buffer(std::size_t __n) : __buf(sycl::range<1>(__n)) {}

  // Return pointer to buffer, or  NULL if buffer could not be obtained.
  auto get() -> decltype(oneapi::dpl::begin(__buf)) const {
    return oneapi::dpl::begin(__buf);
  }
};
#else
template <typename _Tp> class __buffer {
  std::unique_ptr<_Tp> _M_ptr;

  __buffer(const __buffer &) = delete;

  void operator=(const __buffer &) = delete;

public:
  // Try to obtain buffer of given size to store objects of _Tp type
  __buffer(const std::size_t __n) : _M_ptr(new _Tp[__n]) {}

  // Return pointer to buffer, or  NULL if buffer could not be obtained.
  _Tp *get() const { return _M_ptr.get(); }
};
#endif

// Implements C++14 std::less<void> specialization to allow parameter type
// deduction.
class __less {
public:
  template <typename _Xp, typename _Yp>
  bool operator()(_Xp &&__x, _Yp &&__y) const {
    return std::forward<_Xp>(__x) < std::forward<_Yp>(__y);
  }
};

template <typename Policy, typename NewName> struct rebind_policy {
  using type = Policy;
};

template <typename KernelName, typename NewName>
struct rebind_policy<oneapi::dpl::execution::device_policy<KernelName>,
                     NewName> {
  using type = oneapi::dpl::execution::device_policy<NewName>;
};

#if _ONEDPL_FPGA_DEVICE
template <unsigned int factor, typename KernelName, typename NewName>
struct rebind_policy<oneapi::dpl::execution::fpga_policy<factor, KernelName>,
                     NewName> {
  using type = oneapi::dpl::execution::fpga_policy<factor, NewName>;
};
#endif

template <typename T1, typename T2,
          typename R1 = typename std::iterator_traits<T1>::reference,
          typename R2 = typename std::iterator_traits<T2>::reference>
struct perm_fun {
  typedef R2 result_of;
  perm_fun(T1 input) : source(input) {}

  R2 operator()(R1 x) const { return *(source + x); }

private:
  T1 source;
};

// Functor compares first element (key) from tied sequence.
template <typename Compare = class internal::__less> struct compare_key_fun {
  typedef bool result_of;
  compare_key_fun(Compare _comp = internal::__less()) : comp(_comp) {}

  template <typename _T1, typename _T2>
  result_of operator()(_T1 &&a, _T2 &&b) const {
    using std::get;
    return comp(get<0>(a), get<0>(b));
  }

private:
  mutable Compare comp;
};

// Functor evaluates second element of tied sequence with predicate.
// Used by: copy_if, remove_copy_if, stable_partition_copy
// Lambda:
template <typename Predicate> struct predicate_key_fun {
  typedef bool result_of;
  predicate_key_fun(Predicate _pred) : pred(_pred) {}

  template <typename _T1> result_of operator()(_T1 &&a) const {
    using std::get;
    return pred(get<1>(a));
  }

private:
  mutable Predicate pred;
};

// Used by: remove_if
template <typename Predicate> struct negate_predicate_key_fun {
  typedef bool result_of;
  negate_predicate_key_fun(Predicate _pred) : pred(_pred) {}

  template <typename _T1> result_of operator()(_T1 &&a) const {
    using std::get;
    return !pred(get<1>(a));
  }

private:
  mutable Predicate pred;
};

template <typename T> struct sequence_fun {
  using result_type = T;
  sequence_fun(T _init, T _step) : init(_init), step(_step) {}

  template <typename _T> result_type operator()(_T &&i) const {
    return static_cast<T>(init + step * i);
  }

private:
  const T init;
  const T step;
};

//[binary_pred](Ref a, Ref b){ return(binary_pred(get<0>(a),get<0>(b)));
template <typename Predicate> struct unique_fun {
  typedef bool result_of;
  unique_fun(Predicate _pred) : pred(_pred) {}
  template <typename _T> result_of operator()(_T &&a, _T &&b) const {
    using std::get;
    return pred(get<0>(a), get<0>(b));
  }

private:
  mutable Predicate pred;
};

// Lambda: [pred, &new_value](Ref1 a, Ref2 s) {return pred(s) ? new_value : a;
// });
template <typename T, typename Predicate> struct replace_if_fun {
public:
  typedef T result_of;
  replace_if_fun(Predicate _pred, T _new_value)
      : pred(_pred), new_value(_new_value) {}

  template <typename _T1, typename _T2> T operator()(_T1 &&a, _T2 &&s) const {
    return pred(s) ? new_value : a;
  }

private:
  mutable Predicate pred;
  const T new_value;
};

//[pred,op](Ref a){return pred(a) ? op(a) : a; }
template <typename T, typename Predicate, typename Operator>
struct transform_if_fun {
  transform_if_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
  template <typename _T>
  void operator()(_T&& t) const {
    using std::get;
    if (pred(get<0>(t)))
      get<1>(t) = op(get<0>(t));
  }

private:
  mutable Predicate pred;
  mutable Operator op;
};

//[pred, op](Ref1 a, Ref2 s) { return pred(s) ? op(a) : a; });
template <typename T, typename Predicate, typename Operator>
struct transform_if_unary_zip_mask_fun {
  transform_if_unary_zip_mask_fun(Predicate _pred, Operator _op) : pred(_pred), op(_op) {}
  template <typename _T>
  void operator()(_T&& t) const {
    using std::get;
    if (pred(get<1>(t)))
      get<2>(t) = op(get<0>(t));
  }

private:
  mutable Predicate pred;
  mutable Operator op;
};

template <typename T, typename Predicate, typename BinaryOperation>
class transform_if_zip_mask_fun {
public:
  transform_if_zip_mask_fun(Predicate _pred = oneapi::dpl::identity(),
                            BinaryOperation _op = oneapi::dpl::identity())
      : pred(_pred), op(_op) {}
  template <typename _T> void operator()(_T &&t) const {
    using std::get;
    if (pred(get<2>(t)))
      get<3>(t) = op(get<0>(t), get<1>(t));
  }

private:
  mutable Predicate pred;
  mutable BinaryOperation op;
};

// This following code is similar to a section of code in
// oneDPL/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort.h
// It has a similar approach, and could be consolidated.
// Outside of some differences in approach, there are two significant
// differences in function.
//
// 1) This code allows the output type of the bit range translation to be fit
// into to the minimal type required to provide that many bits. The code in
// oneDPL to calculate the bucket for the radix is similar but its output is
// always std::uint32_t.  The assumption that the bit range desired will fit in
// 32 bits is not true for this code.
//
// 2) This code ensures that for floating point type, -0.0f and 0.0f map to the
// same value.  This allows the output of this translation to be used to provide
// a sort which ensures the stability of these values for floating point types.

template <int N> struct uint_byte_map {};
template <> struct uint_byte_map<1> { using type = uint8_t; };
template <> struct uint_byte_map<2> { using type = uint16_t; };
template <> struct uint_byte_map<4> { using type = uint32_t; };
template <> struct uint_byte_map<8> { using type = uint64_t; };

template <typename T> struct uint_map {
  using type = typename uint_byte_map<sizeof(T)>::type;
};

template <typename T, typename OutKeyT> class translate_key {
  using uint_type_t = typename uint_map<T>::type;

public:
  translate_key(int begin_bit, int end_bit) {
    shift = begin_bit;
    mask = ~OutKeyT(0); // all ones
    mask = mask >> (sizeof(OutKeyT) * 8 -
                    (end_bit - begin_bit));           // setup appropriate mask
    flip_sign = uint_type_t(1) << (sizeof(uint_type_t) * 8 - 1); // sign bit
    flip_key = ~uint_type_t(0);                       // 0xF...F
  }

  inline OutKeyT operator()(const T &key) const {
    uint_type_t intermediate;
    if constexpr (std::is_floating_point<T>::value) {
        // normal case (both -0.0f and 0.0f equal -0.0f)
        if (key != T(-0.0f)) {
        uint_type_t is_negative = reinterpret_cast<const uint_type_t &>(key) >>
              (sizeof(uint_type_t) * 8 - 1);
          intermediate = reinterpret_cast<const uint_type_t &>(key) ^
                         ((is_negative * flip_key) | flip_sign);
        } else // special case for -0.0f to keep stability with 0.0f
        {
          T negzero = T(-0.0f);
          intermediate = reinterpret_cast<const uint_type_t &>(negzero);
        }
    } else if constexpr (std::is_signed<T>::value) {
        intermediate = reinterpret_cast<const uint_type_t &>(key) ^ flip_sign;
    } else {
      intermediate = key;
    }

    return static_cast<OutKeyT>(intermediate >> shift) &
           mask; // shift, cast, and mask
  }

private:
  uint8_t shift;
  OutKeyT mask;
  uint_type_t flip_sign;
  uint_type_t flip_key;
};

// Unary operator that returns reference to its argument. Ported from
// oneDPL: oneapi/dpl/pstl/utils.h
struct no_op_fun {
  template <typename Tp> Tp &&operator()(Tp &&a) const {
    return ::std::forward<Tp>(a);
  }
};

// Unary functor which composes a pair of functors by calling them in succession
// on an input
template <typename FunctorInner, typename FunctorOuter>
struct __composition_functor {
  __composition_functor(FunctorInner in, FunctorOuter out)
      : _in(in), _out(out) {}
  template <typename T> T operator()(const T &i) const {
    return _out(_in(i));
  }
  FunctorInner _in;
  FunctorOuter _out;
};

// Unary functor which maps an index of a ROI into a 2D flattened array
template <typename OffsetT> struct __roi_2d_index_functor {
  __roi_2d_index_functor(const OffsetT &num_cols,
                         const ::std::size_t &row_stride)
      : _num_cols(num_cols), _row_stride(row_stride) {}

  template <typename Index> Index operator()(const Index &i) const {
    return _row_stride * (i / _num_cols) + (i % _num_cols);
  }

  OffsetT _num_cols;
  ::std::size_t _row_stride;
};

// Unary functor which maps and index into an interleaved array by its active
// channel
template <typename OffsetT> struct __interleaved_index_functor {
  __interleaved_index_functor(const OffsetT &total_channels,
                              const OffsetT &active_channel)
      : _total_channels(total_channels), _active_channel(active_channel) {}

  template <typename Index> Index operator()(const Index &i) const {
    return i * _total_channels + _active_channel;
  }

  OffsetT _total_channels;
  OffsetT _active_channel;
};

} // end namespace internal

} // end namespace dpct

#endif
