//==---- fft_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_FFT_UTILS_HPP__
#define __DPCT_FFT_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <optional>
#include <utility>
#include "lib_common_utils.hpp"

namespace dpct {
namespace fft {
/// An enumeration type to describe the FFT direction is forward or backward.
enum fft_direction : int {
  forward = 0,
  backward
};
/// An enumeration type to describe the types of FFT input and output data.
enum fft_type : int {
  real_float_to_complex_float = 0,
  complex_float_to_real_float,
  real_double_to_complex_double,
  complex_double_to_real_double,
  complex_float_to_complex_float,
  complex_double_to_complex_double,
};

/// A class to perform FFT calculation.
class fft_engine {
public:
  /// Default constructor.
  fft_engine() {}
  /// Commit the configuration to calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] input_type Input data type.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] output_type Output data type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int dim, long long *n,
              long long *inembed, long long istride, long long idist,
              library_data_t input_type, long long *onembed, long long ostride,
              long long odist, library_data_t output_type, long long batch,
              size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    _q = exec_queue;
    init<long long>(dim, n, inembed, istride, idist, input_type, onembed,
                    ostride, odist, output_type, batch,
                    direction_and_placement);
    if (scratchpad_size) {
      if (_is_estimate_call)
        *scratchpad_size = _workspace_estimate_bytes;
      else
        *scratchpad_size = _workspace_bytes;
    }
  }
  /// Commit the configuration to calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] input_type Input data type.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] output_type Output data type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int dim, int *n, int *inembed,
              int istride, int idist, library_data_t input_type, int *onembed,
              int ostride, int odist, library_data_t output_type, int batch,
              size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    _q = exec_queue;
    init<int>(dim, n, inembed, istride, idist, input_type, onembed, ostride,
              odist, output_type, batch, direction_and_placement);
    if (scratchpad_size) {
      if (_is_estimate_call)
        *scratchpad_size = _workspace_estimate_bytes;
      else
        *scratchpad_size = _workspace_bytes;
    }
  }
  /// Commit the configuration to calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int dim, long long *n,
              long long *inembed, long long istride, long long idist,
              long long *onembed, long long ostride, long long odist,
              fft_type type, long long batch, size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    commit(exec_queue, dim, n, inembed, istride, idist,
           fft_type_to_data_type(type).first, onembed, ostride, odist,
           fft_type_to_data_type(type).second, batch, scratchpad_size,
           direction_and_placement);
  }
  /// Commit the configuration to calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int dim, int *n, int *inembed,
              int istride, int idist, int *onembed, int ostride, int odist,
              fft_type type, int batch, size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    commit(exec_queue, dim, n, inembed, istride, idist,
           fft_type_to_data_type(type).first, onembed, ostride, odist,
           fft_type_to_data_type(type).second, batch, scratchpad_size,
           direction_and_placement);
  }
  /// Commit the configuration to calculate 1-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] n1 The size of the dimension of the data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int n1, fft_type type, int batch,
              size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    _q = exec_queue;
    _n.resize(1);
    _n[0] = n1;
    std::tie(_input_type, _output_type) = fft_type_to_data_type(type);
    _dim = 1;
    _batch = batch;
    _is_basic = true;
    if (direction_and_placement.has_value()) {
      _is_user_specified_dir_and_placement = true;
      _direction = direction_and_placement->first;
      _is_inplace = direction_and_placement->second;
    }
    config_and_commit_basic();
    if (scratchpad_size) {
      if (_is_estimate_call)
        *scratchpad_size = _workspace_estimate_bytes;
      else
        *scratchpad_size = _workspace_bytes;
    }
  }
  /// Commit the configuration to calculate 2-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] n2 The size of the 2nd dimension (outermost) of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int n2, int n1, fft_type type,
              size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    _q = exec_queue;
    _n.resize(2);
    _n[0] = n2;
    _n[1] = n1;
    std::tie(_input_type, _output_type) = fft_type_to_data_type(type);
    _dim = 2;
    _is_basic = true;
    if (direction_and_placement.has_value()) {
      _is_user_specified_dir_and_placement = true;
      _direction = direction_and_placement->first;
      _is_inplace = direction_and_placement->second;
    }
    config_and_commit_basic();
    if (scratchpad_size) {
      if (_is_estimate_call)
        *scratchpad_size = _workspace_estimate_bytes;
      else
        *scratchpad_size = _workspace_bytes;
    }
  }
  /// Commit the configuration to calculate 3-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] n3 The size of the 3rd dimension (outermost) of the data.
  /// \param [in] n2 The size of the 2nd dimension of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int n3, int n2, int n1, fft_type type,
              size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    _q = exec_queue;
    _n.resize(3);
    _n[0] = n3;
    _n[1] = n2;
    _n[2] = n1;
    std::tie(_input_type, _output_type) = fft_type_to_data_type(type);
    _dim = 3;
    _is_basic = true;
    if (direction_and_placement.has_value()) {
      _is_user_specified_dir_and_placement = true;
      _direction = direction_and_placement->first;
      _is_inplace = direction_and_placement->second;
    }
    config_and_commit_basic();
    if (scratchpad_size) {
      if (_is_estimate_call)
        *scratchpad_size = _workspace_estimate_bytes;
      else
        *scratchpad_size = _workspace_bytes;
    }
  }

  /// Create the class for calculate 1-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] n1 The size of the dimension of the data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  static fft_engine *
  create(sycl::queue *exec_queue, int n1, fft_type type, int batch,
         std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
             direction_and_placement = std::nullopt) {
    fft_engine *engine = new fft_engine();
    engine->commit(exec_queue, n1, type, batch, nullptr,
                   direction_and_placement);
    return engine;
  }
  /// Create the class for calculate 2-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] n2 The size of the 2nd dimension (outermost) of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  static fft_engine *
  create(sycl::queue *exec_queue, int n2, int n1, fft_type type,
         std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
             direction_and_placement = std::nullopt) {
    fft_engine *engine = new fft_engine();
    engine->commit(exec_queue, n2, n1, type, nullptr, direction_and_placement);
    return engine;
  }
  /// Create the class for calculate 3-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] n3 The size of the 3rd dimension (outermost) of the data.
  /// \param [in] n2 The size of the 2nd dimension of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  static fft_engine *
  create(sycl::queue *exec_queue, int n3, int n2, int n1, fft_type type,
         std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
             direction_and_placement = std::nullopt) {
    fft_engine *engine = new fft_engine();
    engine->commit(exec_queue, n3, n2, n1, type, nullptr,
                   direction_and_placement);
    return engine;
  }
  /// Create the class for calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  static fft_engine *
  create(sycl::queue *exec_queue, int dim, int *n, int *inembed, int istride,
         int idist, int *onembed, int ostride, int odist, fft_type type,
         int batch,
         std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
             direction_and_placement = std::nullopt) {
    fft_engine *engine = new fft_engine();
    engine->commit(exec_queue, dim, n, inembed, istride, idist, onembed,
                   ostride, odist, type, batch, nullptr,
                   direction_and_placement);
    return engine;
  }
  /// Create the class for calculate FFT without commit any config.
  static fft_engine *create() {
    fft_engine *engine = new fft_engine();
    return engine;
  }
  /// Destroy the class for calculate FFT.
  /// \param [in] engine Pointer returned from fft_engine::craete.
  static void destroy(fft_engine *engine) { delete engine; }

#ifdef __INTEL_MKL__
  /// Estimates the workspace size for calculating n-D FFT.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] estimated_scratchpad_size The estimated workspace size
  /// required for this FFT. If this value is used to allocate memory,
  /// \p direction_and_placement need to be specified explicitly to get correct
  /// result.
  /// \param [in] direction_and_placement Explicitly specify the FFT
  /// direction and placement info. If it is not set, forward direction(if
  /// current FFT is complex-to-complex) and out-of-place (false) are set by default.
  static void
  estimate_size(int dim, long long *n, long long *inembed, long long istride,
                long long idist, long long *onembed, long long ostride,
                long long odist, fft_type type, long long batch,
                size_t *estimated_scratchpad_size,
                std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                    direction_and_placement = std::nullopt) {
    fft_engine *engine = fft_engine::create();
    engine->_is_estimate_call = true;
    engine->commit(&dpct::get_default_queue(), dim, n, inembed, istride, idist,
                   fft_type_to_data_type(type).first, onembed, ostride, odist,
                   fft_type_to_data_type(type).second, batch,
                   estimated_scratchpad_size, direction_and_placement);
    fft_engine::destroy(engine);
  }
  /// Estimates the workspace size for calculating n-D FFT.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] estimated_scratchpad_size The estimated workspace size
  /// required for this FFT. If this value is used to allocate memory,
  /// \p direction_and_placement need to be specified explicitly to get correct
  /// result.
  /// \param [in] direction_and_placement Explicitly specify the FFT
  /// direction and placement info. If it is not set, forward direction(if
  /// current FFT is complex-to-complex) and out-of-place (false) are set by default.
  static void
  estimate_size(int dim, int *n, int *inembed, int istride, int idist,
                int *onembed, int ostride, int odist, fft_type type, int batch,
                size_t *estimated_scratchpad_size,
                std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                    direction_and_placement = std::nullopt) {
    fft_engine *engine = fft_engine::create();
    engine->_is_estimate_call = true;
    engine->commit(&dpct::get_default_queue(), dim, n, inembed, istride, idist,
                   fft_type_to_data_type(type).first, onembed, ostride, odist,
                   fft_type_to_data_type(type).second, batch,
                   estimated_scratchpad_size, direction_and_placement);
    fft_engine::destroy(engine);
  }
  /// Estimates the workspace size for calculating 1-D FFT.
  /// \param [in] n1 The size of the dimension of the data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] estimated_scratchpad_size The estimated workspace size
  /// required for this FFT. If this value is used to allocate memory,
  /// \p direction_and_placement need to be specified explicitly to get correct
  /// result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If it is not set, forward direction(if current FFT is
  /// complex-to-complex) and out-of-place (false) are set by default.
  static void
  estimate_size(int n1, fft_type type, int batch,
                size_t *estimated_scratchpad_size,
                std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                    direction_and_placement = std::nullopt) {
    fft_engine *engine = fft_engine::create();
    engine->_is_estimate_call = true;
    engine->commit(&dpct::get_default_queue(), n1, type, batch,
                   estimated_scratchpad_size, direction_and_placement);
    fft_engine::destroy(engine);
  }
  /// Estimates the workspace size for calculating 2-D FFT.
  /// \param [in] n2 The size of the 2nd dimension (outermost) of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  /// \param [out] estimated_scratchpad_size The estimated workspace size
  /// required for this FFT. If this value is used to allocate memory,
  /// \p direction_and_placement need to be specified explicitly to get correct
  /// result.
  /// \param [in] direction_and_placement Explicitly specify the FFT
  /// direction and placement info. If it is not set, forward direction(if
  /// current FFT is complex-to-complex) and out-of-place (false) are set by default.
  static void
  estimate_size(int n2, int n1, fft_type type,
                size_t *estimated_scratchpad_size,
                std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                    direction_and_placement = std::nullopt) {
    fft_engine *engine = fft_engine::create();
    engine->_is_estimate_call = true;
    engine->commit(&dpct::get_default_queue(), n2, n1, type,
                   estimated_scratchpad_size, direction_and_placement);
    fft_engine::destroy(engine);
  }
  /// Estimates the workspace size for calculating 3-D FFT.
  /// \param [in] n3 The size of the 3rd dimension (outermost) of the data.
  /// \param [in] n2 The size of the 2nd dimension of the data.
  /// \param [in] n1 The size of the 1st dimension (innermost) of the data.
  /// \param [in] type The FFT type.
  /// \param [out] estimated_scratchpad_size The estimated workspace size
  /// required for this FFT. If this value is used to allocate memory,
  /// \p direction_and_placement need to be specified explicitly to get correct
  /// result.
  /// \param [in] direction_and_placement Explicitly specify the FFT
  /// direction and placement info. If it is not set, forward direction(if
  /// current FFT is complex-to-complex) and out-of-place (false) are set by default.
  static void
  estimate_size(int n3, int n2, int n1, fft_type type,
                size_t *estimated_scratchpad_size,
                std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                    direction_and_placement = std::nullopt) {
    fft_engine *engine = fft_engine::create();
    engine->_is_estimate_call = true;
    engine->commit(&dpct::get_default_queue(), n3, n2, n1, type,
                   estimated_scratchpad_size, direction_and_placement);
    fft_engine::destroy(engine);
  }
#endif

  /// Execute the FFT calculation.
  /// \param [in] input Pointer to the input data.
  /// \param [out] output Pointer to the output data.
  /// \param [in] direction The FFT direction.
  template <typename input_t, typename output_t>
  void compute(input_t *input, output_t *output, fft_direction direction) {
    if (_input_type == library_data_t::complex_float &&
        _output_type == library_data_t::complex_float) {
      compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
          (float *)input, (float *)output, direction);
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::complex_double) {
      compute_complex<double, oneapi::mkl::dft::precision::DOUBLE>(
          (double *)input, (double *)output, direction);
    } else if (_input_type == library_data_t::real_float &&
               _output_type == library_data_t::complex_float) {
      _direction = direction;
      compute_real<float, oneapi::mkl::dft::precision::SINGLE>((float *)input,
                                                               (float *)output);
    } else if (_input_type == library_data_t::complex_float &&
               _output_type == library_data_t::real_float) {
      _direction = direction;
      compute_real<float, oneapi::mkl::dft::precision::SINGLE>((float *)input,
                                                               (float *)output);
    } else if (_input_type == library_data_t::real_double &&
               _output_type == library_data_t::complex_double) {
      _direction = direction;
      compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
          (double *)input, (double *)output);
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::real_double) {
      _direction = direction;
      compute_real<double, oneapi::mkl::dft::precision::DOUBLE>(
          (double *)input, (double *)output);
    }
  }
  template <>
  void compute(float *input, sycl::float2 *output, fft_direction direction) {
    _direction = direction;
    compute_real<float, oneapi::mkl::dft::precision::SINGLE>((float *)input,
                                                             (float *)output);
  }
  template <>
  void compute(sycl::float2 *input, float *output, fft_direction direction) {
    _direction = direction;
    compute_real<float, oneapi::mkl::dft::precision::SINGLE>((float *)input,
                                                             (float *)output);
  }
  template <>
  void compute(double *input, sycl::double2 *output, fft_direction direction) {
    _direction = direction;
    compute_real<double, oneapi::mkl::dft::precision::DOUBLE>((double *)input,
                                                              (double *)output);
  }
  template <>
  void compute(sycl::double2 *input, double *output, fft_direction direction) {
    _direction = direction;
    compute_real<double, oneapi::mkl::dft::precision::DOUBLE>((double *)input,
                                                              (double *)output);
  }
  template <>
  void compute(sycl::float2 *input, sycl::float2 *output,
               fft_direction direction) {
    compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
        (float *)input, (float *)output, direction);
  }
  template <>
  void compute(sycl::double2 *input, sycl::double2 *output,
               fft_direction direction) {
    compute_complex<double, oneapi::mkl::dft::precision::DOUBLE>(
        (double *)input, (double *)output, direction);
  }
  /// Setting the user's SYCL queue for calculation.
  /// \param [in] q Pointer to the SYCL queue.
  void set_queue(sycl::queue *q) { _q = q; }
#ifdef __INTEL_MKL__
  /// Setting whether to use external or internal workspace.
  /// \param [in] flag True means using internal workspace. False means using
  /// external workspace.
  void use_internal_workspace(bool flag = true) {
    _use_external_workspace = !flag;
  }
  /// Specify the external workspace.
  /// \param [in] ptr Pointer to the workspace.
  void set_workspace(void *ptr) {
    if (!_use_external_workspace) {
      return;
    }
    if (_input_type == library_data_t::complex_float &&
        _output_type == library_data_t::complex_float) {
      if (_q->get_device().is_gpu()) {
        auto data = dpct::detail::get_memory<float>(ptr);
        _desc_sc->set_workspace(data);
      }
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::complex_double) {
      if (_q->get_device().is_gpu()) {
        auto data = dpct::detail::get_memory<double>(ptr);
        _desc_dc->set_workspace(data);
      }
    } else if ((_input_type == library_data_t::real_float &&
                _output_type == library_data_t::complex_float) ||
               (_input_type == library_data_t::complex_float &&
                _output_type == library_data_t::real_float)) {
      if (_q->get_device().is_gpu()) {
        auto data = dpct::detail::get_memory<float>(ptr);
        _desc_sr->set_workspace(data);
      }
    } else if ((_input_type == library_data_t::real_double &&
                _output_type == library_data_t::complex_double) ||
               (_input_type == library_data_t::complex_double &&
                _output_type == library_data_t::real_double)) {
      if (_q->get_device().is_gpu()) {
        auto data = dpct::detail::get_memory<double>(ptr);
        _desc_dr->set_workspace(data);
      }
    } else {
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "invalid fft type");
    }
  }
#endif
  /// Get the workspace size.
  /// \param [out] scratchpad_size Workspace size in bytes.
  void get_workspace_size(size_t *scratchpad_size) {
    if (scratchpad_size) {
      *scratchpad_size = _workspace_bytes;
    }
  }

private:
  static std::pair<library_data_t, library_data_t>
  fft_type_to_data_type(fft_type type) {
    switch (type) {
    case fft_type::real_float_to_complex_float: {
      return std::make_pair(library_data_t::real_float,
                            library_data_t::complex_float);
    }
    case fft_type::complex_float_to_real_float: {
      return std::make_pair(library_data_t::complex_float,
                            library_data_t::real_float);
    }
    case fft_type::real_double_to_complex_double: {
      return std::make_pair(library_data_t::real_double,
                            library_data_t::complex_double);
    }
    case fft_type::complex_double_to_real_double: {
      return std::make_pair(library_data_t::complex_double,
                            library_data_t::real_double);
    }
    case fft_type::complex_float_to_complex_float: {
      return std::make_pair(library_data_t::complex_float,
                            library_data_t::complex_float);
    }
    case fft_type::complex_double_to_complex_double: {
      return std::make_pair(library_data_t::complex_double,
                            library_data_t::complex_double);
    }
    }
  }

  void config_and_commit_basic() {
    if (_input_type == library_data_t::complex_float &&
        _output_type == library_data_t::complex_float) {
      _desc_sc = std::make_shared<
          oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                       oneapi::mkl::dft::domain::COMPLEX>>(_n);
      std::int64_t distance = 1;
      for (auto i : _n)
        distance = distance * i;
      _fwd_dist = distance;
      _bwd_dist = distance;
      _desc_sc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                          distance);
      _desc_sc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                          distance);
      _desc_sc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                          _batch);
#ifdef __INTEL_MKL__
      if (_is_user_specified_dir_and_placement && _is_inplace)
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_INPLACE);
      else
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
      if (_use_external_workspace) {
        if (_q->get_device().is_gpu()) {
          _desc_sc->set_value(
              oneapi::mkl::dft::config_param::WORKSPACE,
              oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
        }
      }
      if (_is_estimate_call) {
        if (_q->get_device().is_gpu()) {
          _desc_sc->get_value(
              oneapi::mkl::dft::config_param::WORKSPACE_ESTIMATE_BYTES,
              &_workspace_estimate_bytes);
        }
      } else {
        _desc_sc->commit(*_q);
        if (_q->get_device().is_gpu()) {
          _desc_sc->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES,
                              &_workspace_bytes);
        }
      }
#else
      if (_is_user_specified_dir_and_placement && _is_inplace)
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::INPLACE);
      else
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::NOT_INPLACE);
      _desc_sc->commit(*_q);
#endif
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::complex_double) {
      _desc_dc = std::make_shared<
          oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                       oneapi::mkl::dft::domain::COMPLEX>>(_n);
      std::int64_t distance = 1;
      for (auto i : _n)
        distance = distance * i;
      _fwd_dist = distance;
      _bwd_dist = distance;
      _desc_dc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                          distance);
      _desc_dc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                          distance);
      _desc_dc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                          _batch);
#ifdef __INTEL_MKL__
      if (_is_user_specified_dir_and_placement && _is_inplace)
        _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_INPLACE);
      else
        _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
      if (_use_external_workspace) {
        if (_q->get_device().is_gpu()) {
          _desc_dc->set_value(
              oneapi::mkl::dft::config_param::WORKSPACE,
              oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
        }
      }
      if (_is_estimate_call) {
        if (_q->get_device().is_gpu()) {
          _desc_dc->get_value(
              oneapi::mkl::dft::config_param::WORKSPACE_ESTIMATE_BYTES,
              &_workspace_estimate_bytes);
        }
      } else {
        _desc_dc->commit(*_q);
        if (_q->get_device().is_gpu()) {
          _desc_dc->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES,
                              &_workspace_bytes);
        }
      }
#else
      if (_is_user_specified_dir_and_placement && _is_inplace)
        _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::INPLACE);
      else
        _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::NOT_INPLACE);
      _desc_dc->commit(*_q);
#endif
    } else if ((_input_type == library_data_t::real_float &&
                _output_type == library_data_t::complex_float) ||
               (_input_type == library_data_t::complex_float &&
                _output_type == library_data_t::real_float)) {
      _desc_sr = std::make_shared<oneapi::mkl::dft::descriptor<
          oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>(
          _n);
      if (_input_type == library_data_t::real_float &&
          _output_type == library_data_t::complex_float)
        _direction = fft_direction::forward;
      else
        _direction = fft_direction::backward;
      _desc_sr->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                          _batch);
#ifdef __INTEL_MKL__
      if (_is_user_specified_dir_and_placement && _is_inplace) {
        _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_INPLACE);
        set_stride_and_distance_basic<true>(_desc_sr);
      } else {
        _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
        set_stride_and_distance_basic<false>(_desc_sr);
      }
      if (_use_external_workspace) {
        if (_q->get_device().is_gpu()) {
          _desc_sr->set_value(
              oneapi::mkl::dft::config_param::WORKSPACE,
              oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
        }
      }
      if (_is_estimate_call) {
        if (_q->get_device().is_gpu()) {
          _desc_sr->get_value(
              oneapi::mkl::dft::config_param::WORKSPACE_ESTIMATE_BYTES,
              &_workspace_estimate_bytes);
        }
      } else {
        _desc_sr->commit(*_q);
        if (_q->get_device().is_gpu()) {
          _desc_sr->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES,
                              &_workspace_bytes);
        }
      }
#else
      if (_is_user_specified_dir_and_placement && _is_inplace) {
        _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::INPLACE);
        set_stride_and_distance_basic<true>(_desc_sr);
      } else {
        _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::NOT_INPLACE);
        set_stride_and_distance_basic<false>(_desc_sr);
      }
      _desc_sr->commit(*_q);
#endif
    } else if ((_input_type == library_data_t::real_double &&
                _output_type == library_data_t::complex_double) ||
               (_input_type == library_data_t::complex_double &&
                _output_type == library_data_t::real_double)) {
      _desc_dr = std::make_shared<oneapi::mkl::dft::descriptor<
          oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>(
          _n);
      if (_input_type == library_data_t::real_double &&
          _output_type == library_data_t::complex_double)
        _direction = fft_direction::forward;
      else
        _direction = fft_direction::backward;
      _desc_dr->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                          _batch);
#ifdef __INTEL_MKL__
      if (_is_user_specified_dir_and_placement && _is_inplace) {
        _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_INPLACE);
        set_stride_and_distance_basic<true>(_desc_dr);
      } else {
        _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
        set_stride_and_distance_basic<false>(_desc_dr);
      }
      if (_use_external_workspace) {
        if (_q->get_device().is_gpu()) {
          _desc_dr->set_value(
              oneapi::mkl::dft::config_param::WORKSPACE,
              oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
        }
      }
      if (_is_estimate_call) {
        if (_q->get_device().is_gpu()) {
          _desc_dr->get_value(
              oneapi::mkl::dft::config_param::WORKSPACE_ESTIMATE_BYTES,
              &_workspace_estimate_bytes);
        }
      } else {
        _desc_dr->commit(*_q);
        if (_q->get_device().is_gpu()) {
          _desc_dr->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES,
                              &_workspace_bytes);
        }
      }
#else
      if (_is_user_specified_dir_and_placement && _is_inplace) {
        _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::INPLACE);
        set_stride_and_distance_basic<true>(_desc_dr);
      } else {
        _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::NOT_INPLACE);
        set_stride_and_distance_basic<false>(_desc_dr);
      }
      _desc_dr->commit(*_q);
#endif
    } else {
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "invalid fft type");
    }
  }

  void config_and_commit_advanced() {
#ifdef __INTEL_MKL__
#define CONFIG_AND_COMMIT(DESC, PREC, DOM, TYPE)                               \
  {                                                                            \
    DESC = std::make_shared<oneapi::mkl::dft::descriptor<                      \
        oneapi::mkl::dft::precision::PREC, oneapi::mkl::dft::domain::DOM>>(    \
        _n);                                                                   \
    set_stride_advanced(DESC);                                                 \
    DESC->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, _fwd_dist);  \
    DESC->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, _bwd_dist);  \
    DESC->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,      \
                    _batch);                                                   \
    if (_is_user_specified_dir_and_placement && _is_inplace)                   \
      DESC->set_value(oneapi::mkl::dft::config_param::PLACEMENT,               \
                      DFTI_CONFIG_VALUE::DFTI_INPLACE);                        \
    else                                                                       \
      DESC->set_value(oneapi::mkl::dft::config_param::PLACEMENT,               \
                      DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);                    \
    if (_use_external_workspace) {                                             \
      DESC->set_value(oneapi::mkl::dft::config_param::WORKSPACE,               \
                      oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);     \
    }                                                                          \
    if (_is_estimate_call) {                                                   \
      if (_q->get_device().is_gpu()) {                                         \
        DESC->get_value(                                                       \
            oneapi::mkl::dft::config_param::WORKSPACE_ESTIMATE_BYTES,          \
            &_workspace_estimate_bytes);                                       \
      }                                                                        \
    } else {                                                                   \
      DESC->commit(*_q);                                                       \
      if (_is_estimate_call) {                                                 \
        DESC->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES,       \
                        &_workspace_bytes);                                    \
      }                                                                        \
    }                                                                          \
  }
#else
#define CONFIG_AND_COMMIT(DESC, PREC, DOM, TYPE)                               \
  {                                                                            \
    DESC = std::make_shared<oneapi::mkl::dft::descriptor<                      \
        oneapi::mkl::dft::precision::PREC, oneapi::mkl::dft::domain::DOM>>(    \
        _n);                                                                   \
    set_stride_advanced(DESC);                                                 \
    DESC->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, _fwd_dist);  \
    DESC->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, _bwd_dist);  \
    DESC->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,      \
                    _batch);                                                   \
    if (_is_user_specified_dir_and_placement && _is_inplace)                   \
      DESC->set_value(oneapi::mkl::dft::config_param::PLACEMENT,               \
                      oneapi::mkl::dft::config_value::INPLACE);                \
    else                                                                       \
      DESC->set_value(oneapi::mkl::dft::config_param::PLACEMENT,               \
                      oneapi::mkl::dft::config_value::NOT_INPLACE);            \
    DESC->commit(*_q);                                                         \
  }
#endif

    if (_input_type == library_data_t::complex_float &&
        _output_type == library_data_t::complex_float) {
      CONFIG_AND_COMMIT(_desc_sc, SINGLE, COMPLEX, float);
    } else if (_input_type == library_data_t::complex_double &&
               _output_type == library_data_t::complex_double) {
      CONFIG_AND_COMMIT(_desc_dc, DOUBLE, COMPLEX, double);
    } else if ((_input_type == library_data_t::real_float &&
                _output_type == library_data_t::complex_float) ||
               (_input_type == library_data_t::complex_float &&
                _output_type == library_data_t::real_float)) {
      CONFIG_AND_COMMIT(_desc_sr, SINGLE, REAL, float);
    } else if ((_input_type == library_data_t::real_double &&
                _output_type == library_data_t::complex_double) ||
               (_input_type == library_data_t::complex_double &&
                _output_type == library_data_t::real_double)) {
      CONFIG_AND_COMMIT(_desc_dr, DOUBLE, REAL, double);
    } else {
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "invalid fft type");
    }
#undef CONFIG_AND_COMMIT
  }

  template <typename T>
  void init(int dim, T *n, T *inembed, T istride, T idist,
            library_data_t input_type, T *onembed, T ostride, T odist,
            library_data_t output_type, T batch,
            std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                direction_and_placement) {
    if (direction_and_placement.has_value()) {
      _is_user_specified_dir_and_placement = true;
      _direction = direction_and_placement->first;
      _is_inplace = direction_and_placement->second;
    }
    _n.resize(dim);
    _inembed.resize(dim);
    _onembed.resize(dim);
    _input_type = input_type;
    _output_type = output_type;
    for (int i = 0; i < dim; i++) {
      _n[i] = n[i];
    }
    if (inembed && onembed) {
      for (int i = 0; i < dim; i++) {
        _inembed[i] = inembed[i];
        _onembed[i] = onembed[i];
      }
      _istride = istride;
      _ostride = ostride;

      if ((_input_type == library_data_t::real_float &&
           _output_type == library_data_t::complex_float) ||
          (_input_type == library_data_t::real_double &&
           _output_type == library_data_t::complex_double)) {
        _fwd_dist = idist;
        _bwd_dist = odist;
      } else if ((_output_type == library_data_t::real_float &&
                  _input_type == library_data_t::complex_float) ||
                 (_output_type == library_data_t::real_double &&
                  _input_type == library_data_t::complex_double)) {
        _fwd_dist = odist;
        _bwd_dist = idist;
      } else {
        if (_is_user_specified_dir_and_placement &&
            (_direction == fft_direction::backward)) {
          _fwd_dist = odist;
          _bwd_dist = idist;
        } else {
          _fwd_dist = idist;
          _bwd_dist = odist;
        }
      }
    } else {
      _is_basic = true;
    }
    _batch = batch;
    _dim = dim;

    if (_is_basic)
      config_and_commit_basic();
    else
      config_and_commit_advanced();
  }
  template <class Desc_t>
  void set_stride_advanced(std::shared_ptr<Desc_t> desc) {
    if (_dim == 1) {
      std::int64_t input_stride[2] = {0, _istride};
      std::int64_t output_stride[2] = {0, _ostride};
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                      input_stride);
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                      output_stride);
    } else if (_dim == 2) {
      std::int64_t input_stride[3] = {0, _inembed[1] * _istride, _istride};
      std::int64_t output_stride[3] = {0, _onembed[1] * _ostride, _ostride};
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                      input_stride);
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                      output_stride);
    } else if (_dim == 3) {
      std::int64_t input_stride[4] = {0, _inembed[2] * _inembed[1] * _istride,
                                      _inembed[2] * _istride, _istride};
      std::int64_t output_stride[4] = {0, _onembed[2] * _onembed[1] * _ostride,
                                       _onembed[2] * _ostride, _ostride};
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                      input_stride);
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                      output_stride);
    }
  }

  template <class Desc_t> void swap_distance(std::shared_ptr<Desc_t> desc) {
    desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, _bwd_dist);
    desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, _fwd_dist);
    std::int64_t temp = _bwd_dist;
    _bwd_dist = _fwd_dist;
    _fwd_dist = temp;
  }

  template <bool Is_inplace, class Desc_t>
  void set_stride_and_distance_basic(std::shared_ptr<Desc_t> desc) {
    std::int64_t forward_distance = 0;
    std::int64_t backward_distance = 0;

#define SET_STRIDE                                                             \
  {                                                                            \
    if (_direction == fft_direction::forward) {                                \
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,           \
                      real_stride);                                            \
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,          \
                      complex_stride);                                         \
    } else {                                                                   \
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,           \
                      complex_stride);                                         \
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,          \
                      real_stride);                                            \
    }                                                                          \
  }
    if (_dim == 1) {
      if constexpr (Is_inplace) {
        std::int64_t real_stride[2] = {0, 1};
        std::int64_t complex_stride[2] = {0, 1};
        SET_STRIDE;
        forward_distance = 2 * (_n[0] / 2 + 1);
        backward_distance = _n[0] / 2 + 1;
      } else {
        std::int64_t real_stride[2] = {0, 1};
        std::int64_t complex_stride[2] = {0, 1};
        SET_STRIDE;
        forward_distance = _n[0];
        backward_distance = _n[0] / 2 + 1;
      }
    } else if (_dim == 2) {
      if constexpr (Is_inplace) {
        std::int64_t complex_stride[3] = {0, _n[1] / 2 + 1, 1};
        std::int64_t real_stride[3] = {0, 2 * (_n[1] / 2 + 1), 1};
        SET_STRIDE;
        forward_distance = _n[0] * 2 * (_n[1] / 2 + 1);
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      } else {
        std::int64_t complex_stride[3] = {0, _n[1] / 2 + 1, 1};
        std::int64_t real_stride[3] = {0, _n[1], 1};
        SET_STRIDE;
        forward_distance = _n[0] * _n[1];
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      }
    } else if (_dim == 3) {
      if constexpr (Is_inplace) {
        std::int64_t complex_stride[4] = {0, _n[1] * (_n[2] / 2 + 1),
                                          _n[2] / 2 + 1, 1};
        std::int64_t real_stride[4] = {0, _n[1] * 2 * (_n[2] / 2 + 1),
                                       2 * (_n[2] / 2 + 1), 1};
        SET_STRIDE;
        forward_distance = _n[0] * _n[1] * 2 * (_n[2] / 2 + 1);
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      } else {
        std::int64_t complex_stride[4] = {0, _n[1] * (_n[2] / 2 + 1),
                                          _n[2] / 2 + 1, 1};
        std::int64_t real_stride[4] = {0, _n[1] * _n[2], _n[2], 1};
        SET_STRIDE;
        forward_distance = _n[0] * _n[1] * _n[2];
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      }
    }
#undef SET_STRIDE
    desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                    forward_distance);
    desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                    backward_distance);
  }

#define COMPUTE(DESC)                                                          \
  {                                                                            \
    if (_is_inplace) {                                                         \
      auto data_input = dpct::detail::get_memory<T>(input);                    \
      if (_direction == fft_direction::forward) {                              \
        oneapi::mkl::dft::compute_forward<                                     \
            std::remove_reference_t<decltype(*DESC)>, T>(*DESC, data_input);   \
      } else {                                                                 \
        oneapi::mkl::dft::compute_backward<                                    \
            std::remove_reference_t<decltype(*DESC)>, T>(*DESC, data_input);   \
      }                                                                        \
    } else {                                                                   \
      auto data_input = dpct::detail::get_memory<T>(input);                    \
      auto data_output = dpct::detail::get_memory<T>(output);                  \
      if (_direction == fft_direction::forward) {                              \
        oneapi::mkl::dft::compute_forward<                                     \
            std::remove_reference_t<decltype(*DESC)>, T, T>(*DESC, data_input, \
                                                            data_output);      \
      } else {                                                                 \
        oneapi::mkl::dft::compute_backward<                                    \
            std::remove_reference_t<decltype(*DESC)>, T, T>(*DESC, data_input, \
                                                            data_output);      \
      }                                                                        \
    }                                                                          \
  }

  template <class T, oneapi::mkl::dft::precision Precision>
  void compute_complex(T *input, T *output, fft_direction direction) {
    bool is_this_compute_inplace = input == output;

    if (!_is_user_specified_dir_and_placement) {
      // The complex domain descriptor need different config values if the
      // FFT direction or placement is different.
      // Here we check the conditions, and new config values are set and
      // re-committed if needed.
      if (direction != _direction || is_this_compute_inplace != _is_inplace) {
        if constexpr (Precision == oneapi::mkl::dft::precision::SINGLE) {
          if (direction != _direction) {
            swap_distance(_desc_sc);
            _direction = direction;
          }
          if (is_this_compute_inplace != _is_inplace) {
            _is_inplace = is_this_compute_inplace;
#ifdef __INTEL_MKL__
            if (_is_inplace) {
              _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  DFTI_CONFIG_VALUE::DFTI_INPLACE);
            } else {
              _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
            }
#else
            if (_is_inplace) {
              _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::INPLACE);
            } else {
              _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::NOT_INPLACE);
            }
#endif
          }
          _desc_sc->commit(*_q);
        } else {
          if (direction != _direction) {
            swap_distance(_desc_dc);
            _direction = direction;
          }
          if (is_this_compute_inplace != _is_inplace) {
            _is_inplace = is_this_compute_inplace;
#ifdef __INTEL_MKL__
            if (_is_inplace) {
              _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  DFTI_CONFIG_VALUE::DFTI_INPLACE);
            } else {
              _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
            }
#else
            if (_is_inplace) {
              _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::INPLACE);
            } else {
              _desc_dc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::NOT_INPLACE);
            }
#endif
          }
          _desc_dc->commit(*_q);
        }
      }
    }

    if constexpr (Precision == oneapi::mkl::dft::precision::SINGLE) {
      COMPUTE(_desc_sc);
    } else {
      COMPUTE(_desc_dc);
    }
  }

  template <class T, oneapi::mkl::dft::precision Precision>
  void compute_real(T *input, T *output) {
    bool is_this_compute_inplace = input == output;

    if (!_is_user_specified_dir_and_placement) {
      // The real domain descriptor need different config values if the
      // FFT placement is different.
      // Here we check the condition, and new config values are set and
      // re-committed if needed.
      if (is_this_compute_inplace != _is_inplace) {
        if constexpr (Precision == oneapi::mkl::dft::precision::SINGLE) {
          _is_inplace = is_this_compute_inplace;
          if (_is_inplace) {
#ifdef __INTEL_MKL__
            _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                DFTI_CONFIG_VALUE::DFTI_INPLACE);
#else
            _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                oneapi::mkl::dft::config_value::INPLACE);
#endif
            if (_is_basic)
              set_stride_and_distance_basic<true>(_desc_sr);
          } else {
#ifdef __INTEL_MKL__
            _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
#else
            _desc_sr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                oneapi::mkl::dft::config_value::NOT_INPLACE);
#endif
            if (_is_basic)
              set_stride_and_distance_basic<false>(_desc_sr);
          }
          _desc_sr->commit(*_q);
        } else {
          _is_inplace = is_this_compute_inplace;
          if (_is_inplace) {
#ifdef __INTEL_MKL__
            _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                DFTI_CONFIG_VALUE::DFTI_INPLACE);
#else
            _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                oneapi::mkl::dft::config_value::INPLACE);
#endif
            if (_is_basic)
              set_stride_and_distance_basic<true>(_desc_dr);
          } else {
#ifdef __INTEL_MKL__
            _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
#else
            _desc_dr->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                oneapi::mkl::dft::config_value::NOT_INPLACE);
#endif
            if (_is_basic)
              set_stride_and_distance_basic<false>(_desc_dr);
          }
          _desc_dr->commit(*_q);
        }
      }
    }

    if constexpr (Precision == oneapi::mkl::dft::precision::SINGLE) {
      COMPUTE(_desc_sr);
    } else {
      COMPUTE(_desc_dr);
    }
  }
#undef COMPUTE

private:
  sycl::queue *_q = nullptr;
  int _dim;
  std::vector<std::int64_t> _n;
  std::vector<std::int64_t> _inembed;
  std::int64_t _istride;
  std::int64_t _fwd_dist;
  library_data_t _input_type;
  std::vector<std::int64_t> _onembed;
  std::int64_t _ostride;
  std::int64_t _bwd_dist;
  library_data_t _output_type;
  std::int64_t _batch = 1;
  bool _is_basic = false;
  bool _is_inplace = false;
  fft_direction _direction = fft_direction::forward;
  bool _is_user_specified_dir_and_placement = false;
  bool _use_external_workspace = false;
  void *_external_workspace_ptr = nullptr;
  size_t _workspace_bytes = 0;
  bool _is_estimate_call = false;
  size_t _workspace_estimate_bytes = 0;
  std::shared_ptr<oneapi::mkl::dft::descriptor<
      oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>>
      _desc_sr;
  std::shared_ptr<oneapi::mkl::dft::descriptor<
      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>>
      _desc_dr;
  std::shared_ptr<oneapi::mkl::dft::descriptor<
      oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>
      _desc_sc;
  std::shared_ptr<oneapi::mkl::dft::descriptor<
      oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>>
      _desc_dc;
};

using fft_engine_ptr = fft_engine *;
} // namespace fft
} // namespace dpct

#endif // __DPCT_FFT_UTILS_HPP__
