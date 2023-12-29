//==---- blas_utils.hpp----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_BLAS_UTILS_HPP__
#define __DPCT_BLAS_UTILS_HPP__

#include "memory.hpp"
#include "util.hpp"
#include "lib_common_utils.hpp"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <utility>
#include <vector>
#include <thread>

namespace dpct {

/// Get the value of \p s.
/// Copy the data to host synchronously, then return the data.
/// \param [in] p The pointer points the data.
/// \param [in] q The queue where the memory copy should be executed.
template <typename T>
inline auto get_value(const T *s, sycl::queue &q) {
  return detail::get_value(s, q);
}

namespace detail {
inline void mem_free(sycl::queue *exec_queue,
                     std::vector<void *> pointers_array, sycl::event e) {
  e.wait();
  for (auto p : pointers_array)
    sycl::free(p, *exec_queue);
}

inline int stride_for(int num_elems, int mem_align_in_elems) {
  return ((num_elems - 1) / mem_align_in_elems + 1) * mem_align_in_elems;
}

#ifndef DPCT_USM_LEVEL_NONE
template<typename T>
class working_memory {
  T *_input_ptr;
  T *_temp_ptr;
  bool _is_sycl_malloced = false;
  bool _is_scalar_value = false;
  sycl::queue _q;
  sycl::event _e;

public:
  working_memory(size_t size, sycl::queue q) : _q(q) {
    _is_scalar_value = false;
    _temp_ptr = (T *)sycl::malloc_device(size, q);
  }
  working_memory(T *result_ptr, sycl::queue q) : _input_ptr(result_ptr), _q(q) {
    _is_scalar_value = true;
    _is_sycl_malloced = sycl::get_pointer_type(_input_ptr, _q.get_context()) !=
                        sycl::usm::alloc::unknown;
    if (!_is_sycl_malloced)
      _temp_ptr = sycl::malloc_shared<T>(1, _q);
  }
  auto get_ptr() {
    if (_is_scalar_value && _is_sycl_malloced)
      return _input_ptr;
    return _temp_ptr;
  }
  void set_event(sycl::event e) { _e = e; }
  ~working_memory() {
    if (_is_scalar_value) {
      if (!_is_sycl_malloced) {
        _q.memcpy(_input_ptr, _temp_ptr, sizeof(T)).wait();
        sycl::free(_temp_ptr, _q);
      }
    } else {
      std::vector<void *> ptrs{_temp_ptr};
      dpct::async_dpct_free(ptrs, {_e});
    }
  }
};
#endif

template <typename Tx, typename Tr>
inline void nrm2_impl(sycl::queue &q, int n, const void *x, int incx,
                         void *result) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
#ifdef DPCT_USM_LEVEL_NONE
  auto x_buffer = dpct::get_buffer<Tx>(x);
  auto r_buffer =
      sycl::buffer<Tr, 1>(reinterpret_cast<Tr *>(result), sycl::range<1>(1));
  if (dpct::is_device_ptr(result))
    r_buffer = dpct::get_buffer<Tr>(result);
  oneapi::mkl::blas::column_major::nrm2(q, n, x_buffer, incx, r_buffer);
#else
  working_memory<Tr> res_mem(reinterpret_cast<Tr *>(result), q);
  oneapi::mkl::blas::column_major::nrm2(q, n, reinterpret_cast<const Tx *>(x),
                                        incx, res_mem.get_ptr());
#endif
#endif
}

template <bool is_conjugate, class Txy, class Tr>
inline void dotuc_impl(sycl::queue &q, int n, const Txy *x, int incx,
                          const Txy *y, int incy, Tr *result) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
#ifdef DPCT_USM_LEVEL_NONE
  auto x_buffer = dpct::get_buffer<Txy>(x);
  auto y_buffer = dpct::get_buffer<Txy>(y);
  auto r_buffer = sycl::buffer<Tr, 1>((Tr *)result, sycl::range<1>(1));
  if (dpct::is_device_ptr(result))
    r_buffer = dpct::get_buffer<Tr>(result);
  if constexpr (std::is_same_v<Txy, std::complex<float>> ||
                std::is_same_v<Txy, std::complex<double>>) {
    if constexpr (is_conjugate)
      oneapi::mkl::blas::column_major::dotc(q, n, x_buffer, incx, y_buffer,
                                            incy, r_buffer);
    else
      oneapi::mkl::blas::column_major::dotu(q, n, x_buffer, incx, y_buffer,
                                            incy, r_buffer);
  } else
    oneapi::mkl::blas::column_major::dot(q, n, x_buffer, incx, y_buffer, incy,
                                         r_buffer);
#else
  working_memory<Tr> res_mem(result, q);
  if constexpr (std::is_same_v<Txy, std::complex<float>> ||
                std::is_same_v<Txy, std::complex<double>>) {
    if constexpr (is_conjugate)
      oneapi::mkl::blas::column_major::dotc(q, n, x, incx, y, incy, res_mem.get_ptr());
    else
      oneapi::mkl::blas::column_major::dotu(q, n, x, incx, y, incy, res_mem.get_ptr());
  } else
    oneapi::mkl::blas::column_major::dot(q, n, x, incx, y, incy, res_mem.get_ptr());
#endif
#endif
}

template <bool is_conjugate>
inline void dotuc(sycl::queue &q, int n, const void *x,
                     library_data_t x_type, int incx, const void *y,
                     library_data_t y_type, int incy, void *result,
                     library_data_t result_type) {
  std::uint64_t key = detail::get_type_combination_id(x_type, y_type, result_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float, library_data_t::real_float,
                       library_data_t::real_float): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const float *>(x), incx,
        reinterpret_cast<const float *>(y), incy,
        reinterpret_cast<float *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double, library_data_t::real_double,
                       library_data_t::real_double): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const double *>(x), incx,
        reinterpret_cast<const double *>(y), incy,
        reinterpret_cast<double *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                       library_data_t::complex_float,
                       library_data_t::complex_float): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const std::complex<float> *>(x), incx,
        reinterpret_cast<const std::complex<float> *>(y), incy,
        reinterpret_cast<std::complex<float> *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                       library_data_t::complex_double,
                       library_data_t::complex_double): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const std::complex<double> *>(x), incx,
        reinterpret_cast<const std::complex<double> *>(y), incy,
        reinterpret_cast<std::complex<double> *>(result));
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_half, library_data_t::real_half,
                       library_data_t::real_half): {
    detail::dotuc_impl<is_conjugate>(
        q, n, reinterpret_cast<const sycl::half *>(x), incx,
        reinterpret_cast<const sycl::half *>(y), incy,
        reinterpret_cast<sycl::half *>(result));
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

template <class Tx, class Te>
inline void scal_impl(sycl::queue &q, int n, const void *alpha, void *x,
                         int incx) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  Te alpha_val = dpct::get_value(reinterpret_cast<const Te *>(alpha), q);
  auto data_x = get_memory<Tx>(x);
  oneapi::mkl::blas::column_major::scal(q, n, alpha_val,
                                        data_x, incx);
#endif
}

template <class Txy, class Te>
inline void axpy_impl(sycl::queue &q, int n, const void *alpha, const void *x,
                        int incx, void *y, int incy) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  Te alpha_val = dpct::get_value(reinterpret_cast<const Te *>(alpha), q);
  auto data_x = get_memory<const Txy>(x);
  auto data_y = get_memory<Txy>(y);
  oneapi::mkl::blas::column_major::axpy(q, n, alpha_val,
                                        data_x, incx,
                                        data_y, incy);
#endif
}

template <class Txy, class Tc, class Ts>
inline void rot_impl(sycl::queue &q, int n, void *x, int incx, void *y,
                        int incy, const void *c, const void *s) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  Tc c_value = dpct::get_value(reinterpret_cast<const Tc *>(c), q);
  Ts s_value = dpct::get_value(reinterpret_cast<const Ts *>(s), q);
  auto data_x = get_memory<Txy>(x);
  auto data_y = get_memory<Txy>(y);
  oneapi::mkl::blas::column_major::rot(q, n, data_x, incx,
                                       data_y, incy, c_value,
                                       s_value);
#endif
}

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                         oneapi::mkl::transpose b_trans, int m, int n, int k,
                         const void *alpha, const void *a, int lda, const void *b,
                         int ldb, const void *beta, void *c, int ldc) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm(
      q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda,
      data_b, ldb, beta_value, data_c, ldc);
#endif
}

template <class Ta, class Tb, class Tc, class Ts>
inline void gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                            oneapi::mkl::transpose b_trans, int m, int n, int k,
                            const void *alpha, const void **a, int lda,
                            const void **b, int ldb, const void *beta, void **c,
                            int ldc, int batch_size) {
  struct matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts value_info[2];
    std::int64_t size_info[3];
    std::int64_t ld_info[3];
    std::int64_t groupsize_info;
  };

  Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);

  matrix_info_t *matrix_info =
      (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
  matrix_info->transpose_info[0] = a_trans;
  matrix_info->transpose_info[1] = b_trans;
  matrix_info->value_info[0] = alpha_value;
  matrix_info->value_info[1] = beta_value;
  matrix_info->size_info[0] = m;
  matrix_info->size_info[1] = n;
  matrix_info->size_info[2] = k;
  matrix_info->ld_info[0] = lda;
  matrix_info->ld_info[1] = ldb;
  matrix_info->ld_info[2] = ldc;
  matrix_info->groupsize_info = batch_size;

  sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
      q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
      matrix_info->size_info, matrix_info->size_info + 1,
      matrix_info->size_info + 2, matrix_info->value_info,
      reinterpret_cast<const Ta **>(a), matrix_info->ld_info,
      reinterpret_cast<const Tb **>(b), matrix_info->ld_info + 1,
      matrix_info->value_info + 1, reinterpret_cast<Tc **>(c),
      matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));

  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); });
  });
}

template <class Ta, class Tb, class Tc, class Ts>
inline void
gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                    oneapi::mkl::transpose b_trans, int m, int n,
                    int k, const void *alpha, const void *a, int lda,
                    long long int stride_a, const void *b, int ldb,
                    long long int stride_b, const void *beta, void *c,
                    int ldc, long long int stride_c, int batch_size) {
  Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
  Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
  auto data_a = get_memory<const Ta>(a);
  auto data_b = get_memory<const Tb>(b);
  auto data_c = get_memory<Tc>(c);
  oneapi::mkl::blas::column_major::gemm_batch(
      q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda,
      stride_a, data_b, ldb, stride_b, beta_value,
      data_c, ldc, stride_c, batch_size);
}

template <bool is_hermitian, class T, class Tbeta>
inline void rk_impl(sycl::queue &q, oneapi::mkl::uplo uplo,
                          oneapi::mkl::transpose trans, int n, int k,
                          const T *alpha, const T *a, int lda, const T *b,
                          int ldb, const Tbeta *beta, T *c, int ldc) {
  // For symmetric matrix, this function performs: C = alpha*OP(A)*(OP(B))^T + beta*C
  // For Hermitian matrix, this function performs: C = alpha*OP(A)*(OP(B))^H + beta*C
  // The gemmt() function performs: C = alpha*OPA(A)*OPB(B) + beta*C
  // So the OPB need be updated before we call gemmt().
  using Ty = typename dpct::DataType<T>::T2;
  using Ts = typename dpct::DataType<Tbeta>::T2;
  Ty alpha_value = dpct::get_value(reinterpret_cast<const Ty *>(alpha), q);
  Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
  oneapi::mkl::transpose trans_A = trans, trans_B = trans;
  int origin_b_rows = trans == oneapi::mkl::transpose::nontrans ? n : k;
  int origin_b_cols = trans == oneapi::mkl::transpose::nontrans ? k : n;

  if ((is_hermitian && trans == oneapi::mkl::transpose::trans) ||
      (!is_hermitian && !std::is_floating_point_v<Ty> && trans == oneapi::mkl::transpose::conjtrans)) {
    // In this case, OPB need be a conjugate operation,
    // but only notrans, conjtrans and trans are available.
    // So we need do a conjtrans operation first, then do a trans operation.
    trans_B = oneapi::mkl::transpose::trans;
    auto data_a = get_memory<const Ty>(a);
    auto data_c = get_memory<Ty>(c);
#ifdef DPCT_USM_LEVEL_NONE
    auto new_B_buffer = sycl::buffer<Ty, 1>(sycl::range<1>(origin_b_rows * origin_b_cols));
    auto from_buffer = dpct::get_buffer<Ty>(b);
    oneapi::mkl::blas::column_major::omatcopy_batch(
          q, oneapi::mkl::transpose::conjtrans, origin_b_rows, origin_b_cols,
          Ts(1.0), from_buffer, ldb, origin_b_rows * ldb, new_B_buffer,
          origin_b_cols, origin_b_rows * origin_b_cols, 1);
    oneapi::mkl::blas::column_major::gemmt(
        q, uplo, trans_A, trans_B, n, k, alpha_value,
        data_a, lda, new_B_buffer, origin_b_cols, beta_value, data_c, ldc);
#else
    working_memory<T> new_B(origin_b_rows * origin_b_cols * sizeof(T), q);
    oneapi::mkl::blas::column_major::omatcopy_batch(
        q, oneapi::mkl::transpose::conjtrans, origin_b_rows, origin_b_cols,
        Ts(1.0), reinterpret_cast<const Ty *>(b), ldb, origin_b_rows * ldb,
        reinterpret_cast<Ty *>(new_B.get_ptr()), origin_b_cols,
        origin_b_rows * origin_b_cols, 1);
    sycl::event e = oneapi::mkl::blas::column_major::gemmt(
        q, uplo, trans_A, trans_B, n, k, alpha_value,
        data_a, lda, reinterpret_cast<Ty *>(new_B.get_ptr()), origin_b_cols,
        beta_value, data_c, ldc);
    new_B.set_event(e);
#endif
  } else {
    if constexpr (is_hermitian) {
      trans_B = trans == oneapi::mkl::transpose::nontrans
                  ? oneapi::mkl::transpose::conjtrans
                  : oneapi::mkl::transpose::nontrans;
    } else {
      trans_B = trans == oneapi::mkl::transpose::nontrans
                  ? oneapi::mkl::transpose::trans
                  : oneapi::mkl::transpose::nontrans;
    }
    auto data_a = get_memory<const Ty>(a);
    auto data_b = get_memory<const Ty>(b);
    auto data_c = get_memory<Ty>(c);
    oneapi::mkl::blas::column_major::gemmt(
        q, uplo, trans_A, trans_B, n, k, alpha_value,
        data_a, lda, data_b, ldb, beta_value, data_c, ldc);
  }
}

template <class Ta, class Tb, class Ts>
inline void
trsm_batch_impl(sycl::queue &q, oneapi::mkl::side left_right,
                oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                oneapi::mkl::diag unit_diag, int m, int n, const void *alpha,
                const void **a, int lda, void **b, int ldb, int batch_size) {
  struct matrix_info_t {
    matrix_info_t(oneapi::mkl::side side_info, oneapi::mkl::uplo uplo_info,
                  oneapi::mkl::transpose transpose_info,
                  oneapi::mkl::diag diag_info, Ts value_info, std::int64_t m,
                  std::int64_t n, std::int64_t lda, std::int64_t ldb,
                  std::int64_t groupsize_info)
        : side_info(side_info), uplo_info(uplo_info),
          transpose_info(transpose_info), diag_info(diag_info),
          value_info(value_info), groupsize_info(groupsize_info) {
      size_info[0] = m;
      size_info[1] = n;
      ld_info[0] = lda;
      ld_info[1] = ldb;
    }
    oneapi::mkl::side side_info;
    oneapi::mkl::uplo uplo_info;
    oneapi::mkl::transpose transpose_info;
    oneapi::mkl::diag diag_info;
    Ts value_info;
    std::int64_t size_info[2];
    std::int64_t ld_info[2];
    std::int64_t groupsize_info;
  };

  Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);

  matrix_info_t *matrix_info =
      new matrix_info_t(left_right, upper_lower, trans, unit_diag, alpha_value,
                        m, n, lda, ldb, batch_size);

  sycl::event e = oneapi::mkl::blas::column_major::trsm_batch(
      q, &(matrix_info->side_info), &(matrix_info->uplo_info),
      &(matrix_info->transpose_info), &(matrix_info->diag_info),
      matrix_info->size_info, matrix_info->size_info + 1,
      &(matrix_info->value_info), reinterpret_cast<const Ta **>(a),
      matrix_info->ld_info, reinterpret_cast<Tb **>(b),
      matrix_info->ld_info + 1, 1, &(matrix_info->groupsize_info));

  q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.host_task([=] { delete matrix_info; });
  });
}

template <typename T>
inline void getrfnp_batch_wrapper(sycl::queue &exec_queue, int n, T *a[],
                                  int lda, int *info, int batch_size) {
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                           "Project does not support this API.");
#else
  using Ty = typename DataType<T>::T2;
  // Set the info array value to 0
  detail::dpct_memset<unsigned char>(exec_queue, info, 0, sizeof(int) * batch_size);
  std::int64_t stride_a = n * lda;
  std::int64_t scratchpad_size =
      oneapi::mkl::lapack::getrfnp_batch_scratchpad_size<Ty>(
          exec_queue, n, n, lda, stride_a, batch_size);

  Ty *a_strided_mem =
      (Ty *)dpct::dpct_malloc(stride_a * batch_size * sizeof(Ty), exec_queue);
  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  dpct::dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i)
    dpct::dpct_memcpy(a_strided_mem + i * stride_a, host_a[i],
                      n * lda * sizeof(T));

#ifdef DPCT_USM_LEVEL_NONE
  {
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    auto a_buffer = get_buffer<Ty>(a_strided_mem);
    oneapi::mkl::lapack::getrfnp_batch(exec_queue, n, n, a_buffer, lda,
                                       stride_a, batch_size, scratchpad,
                                       scratchpad_size);
  }
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_a[i],
                                         a_strided_mem + i * stride_a,
                                         n * lda * sizeof(T), automatic));
#else
  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  sycl::event e = oneapi::mkl::lapack::getrfnp_batch(
      exec_queue, n, n, a_strided_mem, lda, stride_a, batch_size, scratchpad,
      scratchpad_size);
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_a[i],
                                         a_strided_mem + i * stride_a,
                                         n * lda * sizeof(T), automatic, {e}));

  std::vector<void *> ptrs{scratchpad, a_strided_mem};
  dpct::async_dpct_free(ptrs, events, exec_queue);
#endif

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.host_task([=] { std::free(host_a); });
  });
#endif
}

} // namespace detail

inline oneapi::mkl::transpose get_transpose(int t) {
  if (t == 0) {
    return oneapi::mkl::transpose::nontrans;
  } else if (t == 1) {
    return oneapi::mkl::transpose::trans;
  } else {
    return oneapi::mkl::transpose::conjtrans;
  }
}

/// Computes the LU factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in, out] a Array of pointers to matrices. These matrices will be
/// overwritten by lower triangulars with unit diagonal elements and upper
/// triangulars.
/// \param [in] lda The leading dimension of the matrices.
/// \param [out] ipiv An array stores the pivot indices. If \p ipiv is nullptr,
/// non-pivoting LU factorization is computed.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrf_batch_wrapper(sycl::queue &exec_queue, int n, T *a[],
                                int lda, int *ipiv, int *info, int batch_size) {
  if (ipiv == nullptr) {
    detail::getrfnp_batch_wrapper(exec_queue, n, a, lda, info, batch_size);
    return;
  }
  using Ty = typename DataType<T>::T2;
  // Set the info array value to 0
  detail::dpct_memset<unsigned char>(exec_queue, info, 0, sizeof(int) * batch_size);
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_a = n * lda;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
      exec_queue, n, n, lda, stride_a, stride_ipiv, batch_size);

  T *a_buffer_ptr;
  a_buffer_ptr = (T *)dpct_malloc(stride_a * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));

  {
    sycl::buffer<std::int64_t, 1> ipiv_buf(
        sycl::range<1>(batch_size * stride_ipiv));
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    oneapi::mkl::lapack::getrf_batch(exec_queue, n, n, a_buffer, lda, stride_a,
                             ipiv_buf, stride_ipiv, batch_size, scratchpad,
                             scratchpad_size);

    auto to_buffer = get_buffer<int>(ipiv);
    exec_queue.submit([&](sycl::handler &cgh) {
      auto from_acc = ipiv_buf.get_access<sycl::access_mode::read>(cgh);
      auto to_acc = to_buffer.get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<dpct_kernel_name<class getrf_device_int64_to_int, T>>(
          sycl::range<2>(batch_size, n), [=](sycl::id<2> id) {
            to_acc[id.get(0) * n + id.get(1)] =
                static_cast<int>(from_acc[id.get(0) * stride_ipiv + id.get(1)]);
          });
    });
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_a[i],
                                         a_buffer_ptr + i * stride_a,
                                         n * lda * sizeof(T), automatic));

  std::vector<void *> ptrs{host_a};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptrs, events);
  mem_free_thread.detach();
#else
  std::int64_t m_int64 = n;
  std::int64_t n_int64 = n;
  std::int64_t lda_int64 = lda;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_batch_scratchpad_size<Ty>(
      exec_queue, &m_int64, &n_int64, &lda_int64, 1, &group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);
  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *)).wait();
  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  oneapi::mkl::lapack::getrf_batch(exec_queue, &m_int64, &n_int64, (Ty **)a_shared, &lda_int64,
                           ipiv_int64_ptr, 1, &group_sizes, scratchpad,
                           scratchpad_size);

  sycl::event e = exec_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<dpct_kernel_name<class getrf_device_int64_to_int, T>>(
        sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
          ipiv[idx] = static_cast<int>(ipiv_int64[idx]);
        });
  });

  std::vector<void *> ptrs{scratchpad, ipiv_int64, ipiv_int64_ptr, a_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}

/// Solves a system of linear equations with a batch of LU-factored square
/// coefficient matrices, with multiple right-hand sides.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] trans Indicates the form of the linear equations.
/// \param [in] n The order of the matrices.
/// \param [in] nrhs The number of right hand sides.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [in] ipiv An array stores the pivots.
/// \param [in, out] b Array of pointers to matrices, whose columns are
/// the right-hand sides for the systems of equations.
/// \param [in] ldb The leading dimension of the matrices in \p b.
/// \param [out] info A value stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getrs_batch_wrapper(sycl::queue &exec_queue,
                                oneapi::mkl::transpose trans, int n, int nrhs,
                                const T *a[], int lda, const int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info value to 0
  *info = 0;
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_a = n * lda;
  std::int64_t stride_b = nrhs * ldb;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
      exec_queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b,
      batch_size);

  T *a_buffer_ptr, *b_buffer_ptr;
  a_buffer_ptr = (T *)dpct_malloc(stride_a * batch_size * sizeof(T));
  b_buffer_ptr = (T *)dpct_malloc(stride_b * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  T **host_b = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));
  for (std::int64_t i = 0; i < batch_size; ++i) {
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
    dpct_memcpy(b_buffer_ptr + i * stride_b, host_b[i], nrhs * ldb * sizeof(T));
  }

  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    sycl::buffer<std::int64_t, 1> ipiv_buf(
        sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<sycl::access_mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<dpct_kernel_name<class getrs_device_int64_to_int, T>>(
          sycl::range<2>(batch_size, n), [=](sycl::id<2> id) {
            to_acc[id.get(0) * stride_ipiv + id.get(1)] =
                static_cast<std::int64_t>(from_acc[id.get(0) * n + id.get(1)]);
          });
    });

    oneapi::mkl::lapack::getrs_batch(exec_queue, trans, n, nrhs, a_buffer, lda,
                             stride_a, ipiv_buf, stride_ipiv, b_buffer, ldb,
                             stride_b, batch_size, scratchpad, scratchpad_size);
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_b[i],
                                         b_buffer_ptr + i * stride_b,
                                         nrhs * ldb * sizeof(T), automatic));
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptrs, events);
  mem_free_thread.detach();
#else
  std::int64_t n_int64 = n;
  std::int64_t nrhs_int64 = nrhs;
  std::int64_t lda_int64 = lda;
  std::int64_t ldb_int64 = ldb;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getrs_batch_scratchpad_size<Ty>(
      exec_queue, &trans, &n_int64, &nrhs_int64, &lda_int64, &ldb_int64, 1,
      &group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);
  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **b_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(b_shared, b, batch_size * sizeof(T *));

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<dpct_kernel_name<class getrs_device_int64_to_int, T>>(
        sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  }).wait();

  for (std::int64_t i = 0; i < batch_size; ++i)
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;

  sycl::event e = oneapi::mkl::lapack::getrs_batch(
      exec_queue, &trans, &n_int64, &nrhs_int64, (Ty **)a_shared, &lda_int64,
      ipiv_int64_ptr, (Ty **)b_shared, &ldb_int64, 1, &group_sizes, scratchpad,
      scratchpad_size);

  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64, a_shared, b_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}

/// Computes the inverses of a batch of LU-factored matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] n The order of the matrices.
/// \param [in] a Array of pointers to matrices.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [in] ipiv An array stores the pivots.
/// \param [out] b Array of pointers to inverse matrices.
/// \param [in] ldb The leading dimension of the matrices in \p b.
/// \param [out] info An array stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void getri_batch_wrapper(sycl::queue &exec_queue, int n,
                                const T *a[], int lda, int *ipiv, T *b[],
                                int ldb, int *info, int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info array value to 0
  detail::dpct_memset<unsigned char>(exec_queue, info, 0, sizeof(int) * batch_size);
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_b = n * ldb;
  std::int64_t stride_ipiv = n;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
      exec_queue, n, ldb, stride_b, stride_ipiv, batch_size);

  T *b_buffer_ptr;
  b_buffer_ptr = (T *)dpct_malloc(stride_b * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  T **host_b = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_b, b, batch_size * sizeof(T *));

  for (std::int64_t i = 0; i < batch_size; ++i) {
    // Need to create a copy of input matrices "a" to keep them unchanged.
    // Matrices "b" (copy of matrices "a") will be used as input and output
    // parameter in oneapi::mkl::lapack::getri_batch call.
    matrix_mem_copy(b_buffer_ptr + i * stride_b, host_a[i], ldb, lda, n, n,
                    dpct::device_to_device, exec_queue);
  }

  {
    auto b_buffer = get_buffer<Ty>(b_buffer_ptr);
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    sycl::buffer<std::int64_t, 1> ipiv_buf(
        sycl::range<1>(batch_size * stride_ipiv));
    auto from_buf = get_buffer<int>(ipiv);
    exec_queue.submit([&](sycl::handler &cgh) {
      auto from_acc = from_buf.get_access<sycl::access_mode::read>(cgh);
      auto to_acc = ipiv_buf.get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<dpct_kernel_name<class getri_device_int64_to_int, T>>(
          sycl::range<2>(batch_size, n), [=](sycl::id<2> id) {
            to_acc[id.get(0) * stride_ipiv + id.get(1)] =
                static_cast<std::int64_t>(from_acc[id.get(0) * n + id.get(1)]);
          });
    });

    oneapi::mkl::lapack::getri_batch(exec_queue, n, b_buffer, ldb, stride_b, ipiv_buf,
                             stride_ipiv, batch_size, scratchpad,
                             scratchpad_size);
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events;
  for (std::int64_t i = 0; i < batch_size; ++i)
    events.push_back(detail::dpct_memcpy(exec_queue, host_b[i],
                                         b_buffer_ptr + i * stride_b,
                                         n * ldb * sizeof(T), automatic));
  std::vector<void *> ptrs{host_a, host_b};
  std::thread mem_free_thread(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptrs, events);
  mem_free_thread.detach();
#else
  std::int64_t n_int64 = n;
  std::int64_t ldb_int64 = ldb;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::getri_batch_scratchpad_size<Ty>(
      exec_queue, &n_int64, &ldb_int64, 1, &group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  std::int64_t *ipiv_int64 =
      sycl::malloc_device<std::int64_t>(batch_size * n, exec_queue);
  std::int64_t **ipiv_int64_ptr =
      sycl::malloc_shared<std::int64_t *>(batch_size, exec_queue);

  exec_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<dpct_kernel_name<class getri_device_int64_to_int, T>>(
        sycl::range<1>(batch_size * n), [=](sycl::id<1> idx) {
          ipiv_int64[idx] = static_cast<std::int64_t>(ipiv[idx]);
        });
  });

  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **b_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(b_shared, b, batch_size * sizeof(T *)).wait();
  for (std::int64_t i = 0; i < batch_size; ++i) {
    ipiv_int64_ptr[i] = ipiv_int64 + n * i;
    // Need to create a copy of input matrices "a" to keep them unchanged.
    // Matrices "b" (copy of matrices "a") will be used as input and output
    // parameter in oneapi::mkl::lapack::getri_batch call.
    matrix_mem_copy(b_shared[i], a_shared[i], ldb, lda, n, n, dpct::device_to_device,
                    exec_queue);
  }

  sycl::event e = oneapi::mkl::lapack::getri_batch(
      exec_queue, &n_int64, (Ty **)b_shared, &ldb_int64, ipiv_int64_ptr, 1,
      &group_sizes, scratchpad, scratchpad_size);

  std::vector<void *> ptrs{scratchpad, ipiv_int64_ptr, ipiv_int64, a_shared, b_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}

/// Computes the QR factorizations of a batch of general matrices.
/// \param [in] exec_queue The queue where the routine should be executed.
/// \param [in] m The number of rows in the matrices.
/// \param [in] n The number of columns in the matrices.
/// \param [in, out] a Array of pointers to matrices. These
/// matrices will be overwritten by the factorization data.
/// \param [in] lda The leading dimension of the matrices in \p a.
/// \param [out] tau An array stores the scalars.
/// \param [out] info A value stores the error information.
/// \param [in] batch_size The size of the batch.
template <typename T>
inline void geqrf_batch_wrapper(sycl::queue exec_queue, int m, int n,
                                T *a[], int lda, T *tau[], int *info,
                                int batch_size) {
  using Ty = typename DataType<T>::T2;
  // Set the info value to 0
  *info = 0;
#ifdef DPCT_USM_LEVEL_NONE
  std::int64_t stride_a = n * lda;
  std::int64_t stride_tau = std::max(1, std::min(m, n));
  std::int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
      exec_queue, m, n, lda, stride_a, stride_tau, batch_size);

  T *a_buffer_ptr, *tau_buffer_ptr;
  a_buffer_ptr = (T *)dpct_malloc(stride_a * batch_size * sizeof(T));
  tau_buffer_ptr = (T *)dpct_malloc(stride_tau * batch_size * sizeof(T));

  T **host_a = (T **)std::malloc(batch_size * sizeof(T *));
  T **host_tau = (T **)std::malloc(batch_size * sizeof(T *));
  dpct_memcpy(host_a, a, batch_size * sizeof(T *));
  dpct_memcpy(host_tau, tau, batch_size * sizeof(T *));

  for (std::int64_t i = 0; i < batch_size; ++i)
    dpct_memcpy(a_buffer_ptr + i * stride_a, host_a[i], n * lda * sizeof(T));
  {
    auto a_buffer = get_buffer<Ty>(a_buffer_ptr);
    auto tau_buffer = get_buffer<Ty>(tau_buffer_ptr);
    sycl::buffer<Ty, 1> scratchpad{sycl::range<1>(scratchpad_size)};
    oneapi::mkl::lapack::geqrf_batch(exec_queue, m, n, a_buffer, lda, stride_a,
                             tau_buffer, stride_tau, batch_size, scratchpad,
                             scratchpad_size);
  }

  // Copy back to the original buffers
  std::vector<sycl::event> events_a;
  std::vector<sycl::event> events_tau;
  for (std::int64_t i = 0; i < batch_size; ++i) {
    events_a.push_back(detail::dpct_memcpy(exec_queue, host_a[i],
                                           a_buffer_ptr + i * stride_a,
                                           n * lda * sizeof(T), automatic));
    events_tau.push_back(detail::dpct_memcpy(
        exec_queue, host_tau[i], tau_buffer_ptr + i * stride_tau,
        std::max(1, std::min(m, n)) * sizeof(T), automatic));
  }
  std::vector<void *> ptr_a{host_a};
  std::vector<void *> ptr_tau{host_tau};
  std::thread mem_free_thread_a(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptr_a, events_a);
  std::thread mem_free_thread_tau(
      [=](std::vector<void *> pointers_array,
          std::vector<sycl::event> events_array) {
        sycl::event::wait(events_array);
        for (auto p : pointers_array)
          std::free(p);
      },
      ptr_tau, events_tau);
  mem_free_thread_a.detach();
  mem_free_thread_tau.detach();
#else
  std::int64_t m_int64 = n;
  std::int64_t n_int64 = n;
  std::int64_t lda_int64 = lda;
  std::int64_t group_sizes = batch_size;
  std::int64_t scratchpad_size = oneapi::mkl::lapack::geqrf_batch_scratchpad_size<Ty>(
      exec_queue, &m_int64, &n_int64, &lda_int64, 1, &group_sizes);

  Ty *scratchpad = sycl::malloc_device<Ty>(scratchpad_size, exec_queue);
  T **a_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  T **tau_shared = sycl::malloc_shared<T *>(batch_size, exec_queue);
  exec_queue.memcpy(a_shared, a, batch_size * sizeof(T *));
  exec_queue.memcpy(tau_shared, tau, batch_size * sizeof(T *)).wait();

  sycl::event e = oneapi::mkl::lapack::geqrf_batch(
      exec_queue, &m_int64, &n_int64, (Ty **)a_shared, &lda_int64, (Ty **)tau_shared, 1,
      &group_sizes, scratchpad, scratchpad_size);

  std::vector<void *> ptrs{scratchpad, a_shared, tau_shared};
  async_dpct_free(ptrs, {e}, exec_queue);
#endif
}

/// Computes the Euclidean norm of a vector.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
inline void nrm2(sycl::queue &q, int n, const void *x, library_data_t x_type,
                    int incx, void *result, library_data_t result_type) {
  std::uint64_t key = detail::get_type_combination_id(x_type, result_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float,
                       library_data_t::real_float): {
    detail::nrm2_impl<float, float>(q, n, x, incx, result);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double,
                       library_data_t::real_double): {
    detail::nrm2_impl<double, double>(q, n, x, incx, result);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                       library_data_t::real_float): {
    detail::nrm2_impl<std::complex<float>, float>(
        q, n, x, incx, result);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                       library_data_t::real_double): {
    detail::nrm2_impl<std::complex<double>, double>(
        q, n, x, incx, result);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_half,
                       library_data_t::real_half): {
    detail::nrm2_impl<sycl::half, sycl::half>(
        q, n, x, incx, result);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes the dot product of two vectors.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in] y Input vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
inline void dot(sycl::queue &q, int n, const void *x, library_data_t x_type,
                   int incx, const void *y, library_data_t y_type, int incy,
                   void *result, library_data_t result_type) {
  detail::dotuc<false>(q, n, x, x_type, incx, y, y_type, incy, result,
                          result_type);
}

/// Computes the dot product of two vectors, conjugating the first vector.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in] y Input vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [out] result The result scalar.
/// \param [in] result_type Data type of the result.
inline void dotc(sycl::queue &q, int n, const void *x, library_data_t x_type,
                    int incx, const void *y, library_data_t y_type, int incy,
                    void *result, library_data_t result_type) {
  detail::dotuc<true>(q, n, x, x_type, incx, y, y_type, incy, result,
                         result_type);
}

/// Computes the product of a vector by a scalar.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] alpha The scale factor alpha.
/// \param [in] alpha_type The data type of alpha.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
inline void scal(sycl::queue &q, int n, const void *alpha,
                    library_data_t alpha_type, void *x, library_data_t x_type,
                    int incx) {
  std::uint64_t key = detail::get_type_combination_id(x_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float): {
    detail::scal_impl<float, float>(q, n, alpha, x, incx);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double): {
    detail::scal_impl<double, double>(q, n, alpha, x, incx);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float): {
    detail::scal_impl<std::complex<float>, std::complex<float>>(q, n, alpha,
                                                                   x, incx);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double): {
    detail::scal_impl<std::complex<double>, std::complex<double>>(
        q, n, alpha, x, incx);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_half): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    sycl::half alaph_half(alpha_value);
    detail::scal_impl<sycl::half, sycl::half>(q, n, &alaph_half, x, incx);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a vector-scalar product and adds the result to a vector.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in] alpha The scale factor alpha.
/// \param [in] alpha_type The data type of alpha.
/// \param [in] x Input vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
inline void axpy(sycl::queue &q, int n, const void *alpha,
                    library_data_t alpha_type, const void *x, library_data_t x_type,
                    int incx, void *y, library_data_t y_type, int incy) {
  std::uint64_t key = detail::get_type_combination_id(x_type, alpha_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float,
                       library_data_t::real_float): {
    detail::axpy_impl<float, float>(q, n, alpha, x, incx, y, incy);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double,
                       library_data_t::real_double): {
    detail::axpy_impl<double, double>(q, n, alpha, x, incx, y, incy);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                       library_data_t::complex_float): {
    detail::axpy_impl<std::complex<float>, std::complex<float>>(
        q, n, alpha, x, incx, y, incy);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                       library_data_t::complex_double): {
    detail::axpy_impl<std::complex<double>, std::complex<double>>(
        q, n, alpha, x, incx, y, incy);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_half,
                       library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    sycl::half alaph_half(alpha_value);
    detail::axpy_impl<sycl::half, sycl::half>(q, n, &alaph_half, x, incx, y, incy);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Performs rotation of points in the plane.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] n Number of elements in vector x.
/// \param [in, out] x Input/Output vector x.
/// \param [in] x_type Data type of the vector x.
/// \param [in] incx Stride of vector x.
/// \param [in, out] y Input/Output vector y.
/// \param [in] y_type Data type of the vector y.
/// \param [in] incy Stride of vector y.
/// \param [in] c Scaling factor.
/// \param [in] s Scaling factor.
/// \param [in] cs_type Data type of the scaling factors.
inline void rot(sycl::queue &q, int n, void *x, library_data_t x_type,
                   int incx, void *y, library_data_t y_type, int incy,
                   const void *c, const void *s, library_data_t cs_type) {
  std::uint64_t key = detail::get_type_combination_id(x_type, cs_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float,
                       library_data_t::real_float): {
    detail::rot_impl<float, float, float>(q, n, x, incx, y, incy, c, s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double,
                       library_data_t::real_double): {
    detail::rot_impl<double, double, double>(q, n, x, incx, y, incy, c, s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                       library_data_t::real_float): {
    detail::rot_impl<std::complex<float>, float, float>(q, n, x, incx, y, incy, c,
                                                    s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                       library_data_t::real_double): {
    detail::rot_impl<std::complex<double>, double, double>(q, n, x, incx, y, incy, c,
                                                      s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                       library_data_t::complex_float): {
    detail::rot_impl<std::complex<float>, float, std::complex<float>>(q, n, x, incx, y, incy, c, s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                       library_data_t::complex_double): {
    detail::rot_impl<std::complex<double>, double, std::complex<double>>(q, n, x, incx, y, incy, c, s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_half,
                       library_data_t::real_half): {
    detail::rot_impl<sycl::half, sycl::half, sycl::half>(q, n, x, incx, y, incy, c, s);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_bfloat16,
                       library_data_t::real_bfloat16): {
    detail::rot_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16>(q, n, x, incx, y, incy, c, s);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] scaling_type Data type of the scaling factors.
inline void gemm(sycl::queue &q, oneapi::mkl::transpose a_trans,
                 oneapi::mkl::transpose b_trans, int m, int n, int k,
                 const void *alpha, const void *a, library_data_t a_type,
                 int lda, const void *b, library_data_t b_type, int ldb,
                 const void *beta, void *c, library_data_t c_type, int ldc,
                 library_data_t scaling_type) {
  bool matched = false;
  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (scaling_type == library_data_t::real_double &&
             c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
  switch (key) {
  case detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_impl<float, float, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double): {
    detail::gemm_impl<double, double, double, double>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float): {
    detail::gemm_impl<std::complex<float>, std::complex<float>,
                      std::complex<float>, std::complex<float>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double): {
    detail::gemm_impl<std::complex<double>, std::complex<double>,
                      std::complex<double>, std::complex<double>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_half): {
    detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                      sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                      lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                      float>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_impl<sycl::half, sycl::half, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value =
        dpct::get_value(reinterpret_cast<const float *>(beta), q);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                      sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                      a, lda, b, ldb, &beta_half, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_impl<std::int8_t, std::int8_t, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_bfloat16, library_data_t::real_float): {
    detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                      oneapi::mkl::bfloat16, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_int32, library_data_t::real_int32): {
    float alpha_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
    float beta_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
    detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, float>(
        q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
inline void gemm_batch(sycl::queue &q, oneapi::mkl::transpose a_trans,
                       oneapi::mkl::transpose b_trans, int m, int n, int k,
                       const void *alpha, const void *a[],
                       library_data_t a_type, int lda, const void *b[],
                       library_data_t b_type, int ldb, const void *beta,
                       void *c[], library_data_t c_type, int ldc,
                       int batch_size, library_data_t scaling_type) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  bool matched = false;
  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (scaling_type == library_data_t::real_double &&
             c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
  switch (key) {
  case detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<float, float, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double): {
    detail::gemm_batch_impl<double, double, double, double>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float): {
    detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                            std::complex<float>, std::complex<float>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double): {
    detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                            std::complex<double>, std::complex<double>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_half): {
    detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                            sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                            a, lda, b, ldb, beta, c, ldc,
                                            batch_size);
    break;
  }
#ifdef __INTEL_MKL__
  case detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_bfloat16, library_data_t::real_float): {
    detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                            oneapi::mkl::bfloat16, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                            float>(q, a_trans, b_trans, m, n, k, alpha, a, lda,
                                   b, ldb, beta, c, ldc, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_int32, library_data_t::real_int32): {
    float alpha_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
    float beta_float =
        dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
    detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                            float>(q, a_trans, b_trans, m, n, k, &alpha_float,
                                          a, lda, b, ldb, &beta_float, c, ldc,
                                          batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        batch_size);
    break;
  }
#endif
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value =
        dpct::get_value(reinterpret_cast<const float *>(beta), q);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
        q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc,
        batch_size);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#endif
}

/// Computes a batch of matrix-matrix product with general matrices.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] a_trans Specifies the operation applied to A.
/// \param [in] b_trans Specifies the operation applied to B.
/// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
/// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
/// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrix A.
/// \param [in] a_type Data type of the matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] stride_a Stride between the different A matrices.
/// \param [in] b Input matrix B.
/// \param [in] b_type Data type of the matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] stride_b Stride between the different B matrices.
/// \param [in] beta Scaling factor for matrix C.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] c_type Data type of the matrix C.
/// \param [in] ldc Leading dimension of C.
/// \param [in] stride_c Stride between the different C matrices.
/// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
inline void gemm_batch(sycl::queue &q, oneapi::mkl::transpose a_trans,
                       oneapi::mkl::transpose b_trans, int m, int n, int k,
                       const void *alpha, const void *a, library_data_t a_type,
                       int lda, long long int stride_a, const void *b,
                       library_data_t b_type, int ldb, long long int stride_b,
                       const void *beta, void *c, library_data_t c_type,
                       int ldc, long long int stride_c, int batch_size,
                       library_data_t scaling_type) {
  bool matched = false;
  if (scaling_type == library_data_t::real_float &&
      c_type == library_data_t::complex_float) {
    scaling_type = library_data_t::complex_float;
  } else if (scaling_type == library_data_t::real_double &&
             c_type == library_data_t::complex_double) {
    scaling_type = library_data_t::complex_double;
  }

  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
  switch (key) {
  case detail::get_type_combination_id(
      library_data_t::real_float, library_data_t::real_float,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<float, float, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_double, library_data_t::real_double,
      library_data_t::real_double, library_data_t::real_double): {
    detail::gemm_batch_impl<double, double, double, double>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::complex_float, library_data_t::complex_float,
      library_data_t::complex_float, library_data_t::complex_float): {
    detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                            std::complex<float>, std::complex<float>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::complex_double, library_data_t::complex_double,
      library_data_t::complex_double, library_data_t::complex_double): {
    detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                            std::complex<double>, std::complex<double>>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_half): {
    detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                            sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                            a, lda, stride_a, b, ldb, stride_b,
                                            beta, c, ldc, stride_c, batch_size);
    break;
  }
#ifdef __INTEL_MKL__
  case detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_bfloat16, library_data_t::real_float): {
    detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                            oneapi::mkl::bfloat16, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_bfloat16, library_data_t::real_bfloat16,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                            float>(q, a_trans, b_trans, m, n, k, alpha, a, lda,
                                   stride_a, b, ldb, stride_b, beta, c, ldc,
                                   stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_int32, library_data_t::real_int32): {
    detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                            std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                          a, lda, stride_a, b, ldb, stride_b,
                                          beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_int8, library_data_t::real_int8,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_float, library_data_t::real_float): {
    detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
        q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
        beta, c, ldc, stride_c, batch_size);
    break;
  }
#endif
  case detail::get_type_combination_id(
      library_data_t::real_half, library_data_t::real_half,
      library_data_t::real_half, library_data_t::real_float): {
    float alpha_value =
        dpct::get_value(reinterpret_cast<const float *>(alpha), q);
    float beta_value =
        dpct::get_value(reinterpret_cast<const float *>(beta), q);
    sycl::half alpha_half(alpha_value);
    sycl::half beta_half(beta_value);
    detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
        q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
        &beta_half, c, ldc, stride_c, batch_size);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
}

/// This routines perform a special rank-k update of a symmetric matrix C by
/// general matrices A and B.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] uplo Specifies whether C's data is stored in its upper or lower triangle.
/// \param [in] trans Specifies the operation to apply.
/// \param [in] n The number of rows and columns in C.
/// \param [in] k The inner dimension of matrix multiplications.
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] ldc Leading dimension of C.
template <class T>
inline void syrk(sycl::queue &q, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, int n, int k, const T *alpha,
                  const T *a, int lda, const T *b, int ldb, const T *beta, T *c,
                  int ldc) {
  detail::rk_impl<false, T, T>(q, uplo, trans, n, k, alpha, a, lda, b,
                                     ldb, beta, c, ldc);
}

/// This routines perform a special rank-k update of a Hermitian matrix C by
/// general matrices A and B.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] uplo Specifies whether C's data is stored in its upper or lower triangle.
/// \param [in] trans Specifies the operation to apply.
/// \param [in] n The number of rows and columns in C.
/// \param [in] k The inner dimension of matrix multiplications.
/// \param [in] alpha Scaling factor for the rank-k update.
/// \param [in] a Input matrix A.
/// \param [in] lda Leading dimension of A.
/// \param [in] b Input matrix B.
/// \param [in] ldb Leading dimension of B.
/// \param [in] beta Scaling factor for the rank-k update.
/// \param [in, out] c Input/Output matrix C.
/// \param [in] ldc Leading dimension of C.
template <class T, class Tbeta>
inline void herk(sycl::queue &q, oneapi::mkl::uplo uplo,
                 oneapi::mkl::transpose trans, int n, int k, const T *alpha,
                 const T *a, int lda, const T *b, int ldb, const Tbeta *beta,
                 T *c, int ldc) {
  detail::rk_impl<true, T, Tbeta>(q, uplo, trans, n, k, alpha, a, lda, b,
                                        ldb, beta, c, ldc);
}

/// This routine performs a group of trsm operations. Each trsm solves an
/// equation of the form op(A) * X = alpha * B or X * op(A) = alpha * B.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] left_right Specifies A multiplies X on the left or on the right.
/// \param [in] upper_lower Specifies A is upper or lower triangular.
/// \param [in] trans Specifies the operation applied to A.
/// \param [in] unit_diag Specifies whether A is unit triangular.
/// \param [in] m Number of rows of the B matrices.
/// \param [in] n Number of columns of the B matrices.
/// \param [in] alpha Scaling factor for the solutions.
/// \param [in] a Input matrices A.
/// \param [in] a_type Data type of the matrices A.
/// \param [in] lda Leading dimension of the matrices A.
/// \param [in, out] b Input and output matrices B.
/// \param [in] b_type Data type of the matrices B.
/// \param [in] ldb Leading dimension of the matrices B.
/// \param [in] batch_size Specifies the number of trsm operations to perform.
/// \param [in] scaling_type Data type of the scaling factors.
inline void trsm_batch(sycl::queue &q, oneapi::mkl::side left_right,
                       oneapi::mkl::uplo upper_lower,
                       oneapi::mkl::transpose trans,
                       oneapi::mkl::diag unit_diag, int m, int n,
                       const void *alpha, const void **a, library_data_t a_type,
                       int lda, void **b, library_data_t b_type, int ldb,
                       int batch_size, library_data_t scaling_type) {
#ifdef DPCT_USM_LEVEL_NONE
  throw std::runtime_error("this API is unsupported when USM level is none");
#else
  std::uint64_t key =
      detail::get_type_combination_id(a_type, b_type, scaling_type);
  switch (key) {
  case detail::get_type_combination_id(library_data_t::real_float,
                                       library_data_t::real_float,
                                       library_data_t::real_float): {
    detail::trsm_batch_impl<float, float, float>(q, left_right, upper_lower,
                                                 trans, unit_diag, m, n, alpha,
                                                 a, lda, b, ldb, batch_size);
    break;
  }
  case detail::get_type_combination_id(library_data_t::real_double,
                                       library_data_t::real_double,
                                       library_data_t::real_double): {
    detail::trsm_batch_impl<double, double, double>(
        q, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
        ldb, batch_size);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_float,
                                       library_data_t::complex_float,
                                       library_data_t::complex_float): {
    detail::trsm_batch_impl<std::complex<float>, std::complex<float>,
                            std::complex<float>>(q, left_right, upper_lower,
                                                 trans, unit_diag, m, n, alpha,
                                                 a, lda, b, ldb, batch_size);
    break;
  }
  case detail::get_type_combination_id(library_data_t::complex_double,
                                       library_data_t::complex_double,
                                       library_data_t::complex_double): {
    detail::trsm_batch_impl<std::complex<double>, std::complex<double>,
                            std::complex<double>>(q, left_right, upper_lower,
                                                  trans, unit_diag, m, n, alpha,
                                                  a, lda, b, ldb, batch_size);
    break;
  }
  default:
    throw std::runtime_error("the combination of data type is unsupported");
  }
#endif
}

/// Computes a triangular matrix-general matrix product.
/// \param [in] q The queue where the routine should be executed.
/// \param [in] left_right Specifies A is on the left or right side of the
/// multiplication.
/// \param [in] upper_lower Specifies A is upper or lower triangular.
/// \param [in] trans Specifies the operation applied to A.
/// \param [in] unit_diag Specifies whether A is unit triangular.
/// \param [in] m Number of rows of B.
/// \param [in] n Number of columns of B.
/// \param [in] alpha Scaling factor for the matrix-matrix product.
/// \param [in] a Input matrices A.
/// \param [in] lda Leading dimension of the matrices A.
/// \param [in] b Input matrices B.
/// \param [in] ldb Leading dimension of the matrices B.
/// \param [out] c Output matrices C.
/// \param [in] ldc Leading dimension of the matrices C.
template <class T>
inline void trmm(sycl::queue &q, oneapi::mkl::side left_right,
                 oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, int m, int n, const T *alpha,
                 const T *a, int lda, const T *b, int ldb, T *c, int ldc) {
  using Ty = typename DataType<T>::T2;
  auto alpha_val = dpct::get_value(alpha, q);
  if (b != c) {
    dpct::matrix_mem_copy(c, b, ldc, ldb, m, n, dpct::device_to_device, q);
  }
  auto data_a = detail::get_memory<const Ty>(a);
  auto data_c = detail::get_memory<Ty>(c);
  oneapi::mkl::blas::column_major::trmm(q, left_right, upper_lower, trans,
                                        unit_diag, m, n, alpha_val, data_a, lda,
                                        data_c, ldc);
}

} // namespace dpct
#endif // __DPCT_BLAS_UTILS_HPP__
