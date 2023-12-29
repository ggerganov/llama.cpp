//==---- dnnl_utils.hpp ---------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_DNNL_UTILS_HPP__
#define __DPCT_DNNL_UTILS_HPP__

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <unordered_map>
#include <algorithm>
#include <list>

#include "memory.hpp"
#include "device.hpp"
#include "lib_common_utils.hpp"

namespace dpct {
namespace dnnl {
/// Get concatenated library version as an integer.
static inline size_t get_version() {
  const ::dnnl::version_t *ver = ::dnnl::version();
  return ver->major * 1000 + ver->minor * 100 + ver->patch;
}
class engine_ext;
typedef oneapi::mkl::rng::philox4x32x10 rng_engine_t;
/// An enum class representing memory layout. Used by
/// memory_desc_ext to create a memory with pre-defined layout.
enum class memory_format_tag { nchw, nhwc, nchw_blocked };

/// An enum class representing RNN data memory layout. Used by
/// memory_desc_ext to create a memory with pre-defined layout.
enum class rnn_memory_format_tag { tnc, ntc };

/// A class holding the description of an N-dimensions memory.
class memory_desc_ext {
  ::dnnl::memory::desc _desc;
public:
  /// Convert dpct::library_data_t to dnnl::memory::data_type.
  static ::dnnl::memory::data_type to_dnnl_data_type(dpct::library_data_t dt);
  /// Convert dnnl::memory::data_type to dpct::library_data_t.
  static dpct::library_data_t
  to_dpct_library_data_t(::dnnl::memory::data_type dt, unsigned block_size);
  /// Convert dpct::dnnl::memory_format_tag to dnnl::memory::format_tag.
  static ::dnnl::memory::format_tag to_dnnl_format_tag(dpct::library_data_t dt,
                                                       memory_format_tag tag);
  memory_desc_ext() = default;
  memory_desc_ext(::dnnl::memory::desc &desc) : _desc(desc) {}
  memory_desc_ext(::dnnl::memory::desc &&desc) : _desc(std::move(desc)) {}
  /// Setting a 4D memory with given parameters.
  /// \param [in] tag Format tag.
  /// \param [in] dt Data type.
  /// \param [in] n Number of images.
  /// \param [in] c Number of channels.
  /// \param [in] h Height of images.
  /// \param [in] w Width of images.
  void set(memory_format_tag tag, dpct::library_data_t dt, int n, int c, int h,
           int w);
  /// Setting a 3D RNN data memory with given parameters.
  /// \param [in] tag RNN data format tag.
  /// \param [in] dt Data type.
  /// \param [in] t Number of sequence length.
  /// \param [in] n Number of batch.
  /// \param [in] c Height of input channel.
  void set(rnn_memory_format_tag tag, dpct::library_data_t dt, int t, int n, int c);
  /// Setting a 4D memory with given parameters.
  /// \param [in] dt Data type.
  /// \param [in] n Number of images.
  /// \param [in] c Number of channels.
  /// \param [in] h Height of images.
  /// \param [in] w Width of images.
  /// \param [in] n_stride Stride between two continuous images.
  /// \param [in] c_stride Stride between two continuous channels.
  /// \param [in] h_stride Stride between two continuous rows.
  /// \param [in] w_stride Stride between two continuous columns.
  void set(dpct::library_data_t dt, int n, int c, int h, int w, int n_stride,
           int c_stride, int h_stride, int w_stride);
  /// Setting a ND memory with given parameters.
  /// \param [in] dt Data type.
  /// \param [in] ndims Dimension of the memory.
  /// \param [in] dims Array of dimension ndims that contain the size of each
  /// memory dimension. \param [in] strides Array of dimension ndims that
  /// contain the stride of each memory dimension.
  void set(dpct::library_data_t dt, int ndims, const int dims[],
           const int strides[]);
  /// Setting a ND memory with given parameters.
  /// \param [in] tag Format tag.
  /// \param [in] dt Data type.
  /// \param [in] ndims Dimension of the memory.
  /// \param [in] dims Array of dimension ndims that contain the size of each
  /// memory dimension.
  void set(memory_format_tag tag, dpct::library_data_t dt, int ndims,
           const int dims[]);
  /// Getting a ::dnnl::memory::desc from a memory_desc_ext.
  /// \returns The ::dnnl::memory::desc.
  const ::dnnl::memory::desc &get_desc() const { return _desc; }
  /// Setting holding desc with given dnnl memory descriptor.
  void set_desc(::dnnl::memory::desc desc) { _desc = desc; }
  /// Getting a size of a memory_desc_ext in bytes.
  /// \returns The size.
  size_t get_size() const { return _desc.get_size(); }
  /// Getting parameters from a 4D memory.
  /// \param [out] dt Data type.
  /// \param [out] n Number of images.
  /// \param [out] c Number of channels.
  /// \param [out] h Height of images.
  /// \param [out] w Width of images.
  /// \param [out] n_stride Stride between two continuous images.
  /// \param [out] c_stride Stride between two continuous channels.
  /// \param [out] h_stride Stride between two continuous rows.
  /// \param [out] w_stride Stride between two continuous columns.
  void get(dpct::library_data_t *dt, int *n, int *c, int *h, int *w,
           int *n_stride, int *c_stride, int *h_stride, int *w_stride) const;
  /// Getting parameters from a 4D memory.
  /// \param [out] dt Data type.
  /// \param [out] tag Format tag.
  /// \param [out] n Number of images.
  /// \param [out] c Number of channels.
  /// \param [out] h Height of images.
  /// \param [out] w Width of images.
  void get(dpct::library_data_t *dt, memory_format_tag *tag, int *n, int *c,
           int *h, int *w) const;
  /// Getting parameters from a 3D RNN data memory.
  /// \param [out] dt Data type.
  /// \param [out] tag RNN data format tag.
  /// \param [out] t Number of sequence length.
  /// \param [out] n Number of batch.
  /// \param [out] c Height of input channel.
  void get(dpct::library_data_t *dt, rnn_memory_format_tag *tag, int *t, int *n,
           int *c) const;
  /// Getting parameters from a ND memory.
  /// \param [in] requested_ndims Requested number of dimensions to get from a
  /// given memory descriptor.
  /// \param [out] dt Data type.
  /// \param [out] ndims Dimension of the memory.
  /// \param [out] dims Array of dimension requested_ndims that contain the 
  /// size of each memory dimension.
  /// \param [out] strides Array of dimension requested_ndims that contain the
  /// stride of each memory dimension.
  void get(int requested_ndims, dpct::library_data_t *dt, int *ndims,
           int dims[], int strides[]) const;
  /// Getting parameters from a ND memory.
  /// \param [in] requested_ndims Requested number of dimensions to get from a
  /// given memory descriptor.
  /// \param [out] dt Data type.
  /// \param [out] tag Format tag.
  /// \param [out] ndims Dimension of the memory.
  /// \param [out] dims Array of dimension requested_ndims that contain the 
  /// size of each memory dimension.
  void get(int requested_ndims, dpct::library_data_t *dt,
           memory_format_tag *tag, int *ndims, int dims[]) const;
  /// Getting dims from a ND memory.
  /// \return The dims.
  std::vector<int64_t> get_dims() const { return _desc.get_dims(); }
  /// Getting strides from a ND memory.
  /// \return The strides.
  std::vector<int64_t> get_strides() const {
    return _desc.get_strides();
  }
  /// Getting element num from a ND memory.
  /// \return The element number.
  size_t get_element_num() const {
    auto dims = _desc.get_dims();
    if (dims.empty()) {
      return 0;
    }
    size_t result = 1;
    for (auto &dim : dims) {
      result *= dim;
    }
    return result;
  }

  operator bool() const {
    return bool(_desc);
  }

  memory_desc_ext &operator=(std::nullptr_t) {
    _desc.reset(nullptr);
    return *this;
  }
};

/// A class holding description for an activation operation.
class activation_desc {
  ::dnnl::algorithm _alg;
  float _alpha;
  float _beta;

public:
  /// Setting an activation descriptor with given parameters.
  /// \param [in] alg Activation algorithm.
  /// \param [in] alpha Value of alpha parameter.
  void set(::dnnl::algorithm alg, float alpha) {
    _alg = alg;
    if(alg == ::dnnl::algorithm::eltwise_clip) {
      _alpha = 0;
      _beta = alpha;
    } else {
      _alpha = alpha;
    }
  }
  /// Getting parameters form an activation descriptor.
  /// \param [out] alg Activation algorithm.
  /// \param [out] alpha Value of alpha parameter.
  void get(::dnnl::algorithm *alg, float *alpha) const {
    *alg = _alg;
    if(_alg == ::dnnl::algorithm::eltwise_clip) {
      *alpha = _beta;
    } else {
      *alpha = _alpha;
    }
  }
  /// Setting the alpha parameter of an activation descriptor.
  /// \param [in] alpha Value of alpha parameter.
  void set_alpha(float alpha) { _alpha = alpha; }
  /// Setting the beta parameter of an activation descriptor.
  /// \param [in] beta Value of beta parameter.
  void set_beta(float beta) { _beta = beta; }
  /// Setting the algorithm parameter of an activation descriptor.
  /// \param [in] alg Activation algorithm.
  void set_algorithm(::dnnl::algorithm alg) { _alg = alg; }
  /// Getting the alpha parameter from an activation descriptor.
  /// \param [out] alpha Value of alpha parameter.
  float get_alpha() const { return _alpha; }
  /// Getting the beta parameter from an activation descriptor.
  /// \param [out] beta Value of beta parameter.
  float get_beta() const { return _beta; }
  /// Getting the algorithm parameter from an activation descriptor.
  /// \param [out] alg Activation algorithm.
  ::dnnl::algorithm get_algorithm() const { return _alg; }
};

/// A class holding description for a local response normalization operation.
class lrn_desc {
  unsigned int _local_size;
  float _alpha;
  float _beta;
  float _k;

public:
  /// Setting a local response normalization descriptor with given parameters.
  /// \param [in] local_size Value of local_size parameter.
  /// \param [in] alpha Value of alpha parameter.
  /// \param [in] beta Value of beta parameter.
  /// \param [in] k Value of k parameter.
  void set(unsigned int local_size, float alpha, float beta, float k) {
    _local_size = local_size;
    _alpha = alpha;
    _beta = beta;
    _k = k;
  }
  /// Getting parameters form a local response normalization descriptor.
  /// \param [out] local_size Value of local_size parameter.
  /// \param [out] alpha Value of alpha parameter.
  /// \param [out] beta Value of beta parameter.
  /// \param [out] k Value of k parameter.
  void get(unsigned int *local_size, float *alpha, float *beta,
           float *k) const {
    *local_size = _local_size;
    *alpha = _alpha;
    *beta = _beta;
    *k = _k;
  }
  /// Setting the local size parameter of a local response normalization
  /// descriptor.
  /// \param [in] local_size Value of local_size parameter.
  void set_local_size(unsigned int local_size) { _local_size = local_size; }
  /// Setting the alpha parameter of a local response normalization descriptor.
  /// \param [in] alpha Value of alpha parameter.
  void set_alpha(float alpha) { _alpha = alpha; }
  /// Setting the beta parameter of a local response normalization descriptor.
  /// \param [in] beta Value of beta parameter.
  void set_beta(float beta) { _beta = beta; }
  /// Setting the k parameter of a local response normalization descriptor.
  /// \param [in] k Value of k parameter.
  void set_k(float k) { _k = k; }
  /// Getting the local size parameter from a local response normalization
  /// descriptor.
  /// \param [out] local_size Value of local_size parameter.
  unsigned int get_local_size() const { return _local_size; }
  /// Getting the alpha parameter from a local response normalization
  /// descriptor.
  /// \param [out] alpha Value of alpha parameter.
  float get_alpha() const { return _alpha; }
  /// Getting the beta parameter from a local response normalization descriptor.
  /// \param [out] beta Value of beta parameter.
  float get_beta() const { return _beta; }
  /// Getting the k parameter from a local response normalization descriptor.
  /// \param [out] k Value of k parameter.
  float get_k() const { return _k; }
};

/// An enum class representing softmax algorithm.
enum class softmax_algorithm { normal, log };
/// An enum class representing softmax mode.
enum class softmax_mode { instance, channel };

/// A class holding description for a pooling operation.
class pooling_desc {
  ::dnnl::algorithm _alg;
  std::vector<int64_t> _stride;
  std::vector<int64_t> _kernel;
  std::vector<int64_t> _padding;

public:
  /// Setting a 2D pooling descriptor with given parameters.
  /// \param [in] alg Pooling algorithm.
  /// \param [in] kernel_h Value of height of kernel.
  /// \param [in] kernel_w Value of width of kernel.
  /// \param [in] padding_h Value of height of padding.
  /// \param [in] padding_w Value of width of padding.
  /// \param [in] stride_h Value of height of stride.
  /// \param [in] stride_w Value of width of stride.
  void set(::dnnl::algorithm alg, int kernel_h, int kernel_w, int padding_h,
           int padding_w, int stride_h, int stride_w) {
    _alg = alg;
    _stride = {stride_h, stride_w};
    _kernel = {kernel_h, kernel_w};
    _padding = {padding_h, padding_w};
  }
  /// Setting a ND pooling descriptor with given parameters.
  /// \param [in] alg Pooling algorithm.
  /// \param [in] ndims Dimension of the pooling operation.
  /// \param [in] kernel Array of dimension ndims containing the kernel size of
  /// each dimension.
  /// \param [in] padding Array of dimension ndims containing the padding size of
  /// each dimension.
  /// \param [in] stride Array of dimension ndims containing the stride size of
  /// each dimension.
  void set(::dnnl::algorithm alg, int ndims, int kernel[], int padding[],
           int stride[]) {
    _alg = alg;
    _stride = std::vector<int64_t>(stride, stride + ndims);
    _kernel = std::vector<int64_t>(kernel, kernel + ndims);
    _padding = std::vector<int64_t>(padding, padding + ndims);
  }
  /// Getting parameters from a 2D pooling descriptor.
  /// \param [out] alg Pooling algorithm.
  /// \param [out] kernel_h Value of height of kernel.
  /// \param [out] kernel_w Value of width of kernel.
  /// \param [out] padding_h Value of height of padding.
  /// \param [out] padding_w Value of width of padding.
  /// \param [out] stride_h Value of height of stride.
  /// \param [out] stride_w Value of width of stride.
  void get(::dnnl::algorithm *alg, int *kernel_h, int *kernel_w, int *padding_h,
           int *padding_w, int *stride_h, int *stride_w) const {
    *alg = _alg;
    *kernel_h = _kernel[0];
    *kernel_w = _kernel[1];
    *padding_h = _padding[0];
    *padding_w = _padding[1];
    *stride_h = _stride[0];
    *stride_w = _stride[1];
  }
  /// Getting parameters from a ND pooling descriptor.
  /// \param [in] requested_ndims Requested number of dimensions to get from a
  /// given pooling descriptor.
  /// \param [out] alg Pooling algorithm.
  /// \param [out] ndims Dimension of the pooling operation.
  /// \param [out] kernel Array of dimension ndims containing the kernel size of
  /// each dimension.
  /// \param [out] padding Array of dimension ndims containing the padding size
  /// of each dimension.
  /// \param [out] stride Array of dimension ndims containing the stride size of
  /// each dimension.
  void get(int requested_ndims, ::dnnl::algorithm *alg, int *ndims,
           int kernel[], int padding[], int stride[]) const {
    *alg = _alg;
    *ndims = _stride.size();
    for (int i = 0; i < requested_ndims; i++) {
      kernel[i] = _kernel[i];
      padding[i] = _padding[i];
      stride[i] = _stride[i];
    }
  }
  /// Setting the algorithm parameter of a pooling descriptor.
  /// \param [in] alg Pooling algorithm.
  void set_algorithm(::dnnl::algorithm alg) { _alg = alg; }
  /// Setting the stride parameter of a pooling descriptor.
  /// \param [in] stride Array of dimension ndims containing the stride size of
  /// each dimension.
  void set_stride(const std::vector<int64_t> &stride) { _stride = stride; }
  /// Setting the kernel parameter of a pooling descriptor.
  /// \param [in] kernel Array of dimension ndims containing the kernel size of
  /// each dimension.
  void set_kernel(const std::vector<int64_t> &kernel) { _kernel = kernel; }
  /// Setting the padding parameter of a pooling descriptor.
  /// \param [in] padding Array of dimension ndims containing the padding size
  /// of each dimension.
  void set_padding(const std::vector<int64_t> &padding) { _padding = padding; }

  /// Getting the algorithm parameter from a pooling descriptor.
  /// \param [out] alg Pooling algorithm.
  ::dnnl::algorithm get_algorithm() const { return _alg; }
  /// Getting the stride parameter from a pooling descriptor.
  /// \returns Array of dimension ndims containing the stride size of each
  /// dimension.
  const std::vector<int64_t> &get_stride() const { return _stride; }
  /// Getting the kernel parameter from a pooling descriptor.
  /// \returns Array of dimension ndims containing the kernel size of each
  /// dimension.
  const std::vector<int64_t> &get_kernel() const { return _kernel; }
  /// Getting the padding parameter from a pooling descriptor.
  /// \returns Array of dimension ndims containing the padding size of each
  /// dimension.
  const std::vector<int64_t> &get_padding() const { return _padding; }
  /// Getting the output dimensions of a memory after 2D pooling has been
  /// applied.
  /// \param [in] desc Input memory descriptor.
  /// \param [out] out_n Number of images.
  /// \param [out] out_c Number of channels.
  /// \param [out] out_h Height of images.
  /// \param [out] out_w Width of images.
  void get_forward_output_dim(const memory_desc_ext &desc, int *out_n,
                              int *out_c, int *out_h, int *out_w) const {
    auto dims = desc.get_dims();
    *out_n = dims[0];
    *out_c = dims[1];
    *out_h = 1 + (dims[2] + 2 * _padding[0] - _kernel[0]) / _stride[0];
    *out_w = 1 + (dims[3] + 2 * _padding[1] - _kernel[1]) / _stride[1];
  }
  /// Getting the output dimensions of a memory after ND pooling has been
  /// applied.
  /// \param [in] desc Input memory descriptor.
  /// \param [out] ndims Dimension of the memory.
  /// \param [out] out_dims Array of dimension requested_ndims that contain
  /// the size of each memory dimension.
  void get_forward_output_dim(const memory_desc_ext &desc, int ndims,
                              int out_dims[]) const {
    assert(ndims >= 4 && "ndims is at least 4.");
    auto dims = desc.get_dims();
    out_dims[0] = dims[0];
    out_dims[1] = dims[1];
    for (int i = 2; i < ndims; i++) {
      out_dims[i] =
          1 + (dims[i] + 2 * _padding[i - 2] - _kernel[i - 2]) / _stride[i - 2];
    }
  }
};

/// An enum class representing reduction operations.
enum class reduction_op {
  max,
  min,
  sum,
  mul,
  mean,
  amax,
  mul_no_zeros,
  norm1,
  norm2
};

/// An enum class representing batch normalization mode.
enum class batch_normalization_mode { per_activation, spatial };

/// An enum class representing batch normalization operations.
enum class batch_normalization_ops { none, activation, add_activation };

/// An enum class representing binary operations.
enum class binary_op { add, sub, mul, div, min, max, sqrt, neg };

/// An struct representing convolution algorithm infomation.
struct convolution_algorithm_info {
  ::dnnl::algorithm algo = ::dnnl::algorithm::convolution_auto;
  int status = 0;
};

/// A class holding description for a convolution operation.
class convolution_desc {
  std::vector<int64_t> _strides;
  std::vector<int64_t> _dilates;
  std::vector<int64_t> _paddings;
  int _group_count = 1;
  ::dnnl::fpmath_mode _math_mode = ::dnnl::fpmath_mode::strict;
public:
  /// Setting a group count to be used in the convolution.
  /// \param [in] group_count Value of group count.
  void set_group_count(int group_count) { _group_count = group_count; }
  /// Getting a group count specified in the given convolution descriptor.
  /// \returns Value of group count.
  int get_group_count() { return _group_count; }
  /// Setting floating point math mode to be used in the convolution.
  /// \param [in] math_mode Value of math_mode.
  void set_math_mode(::dnnl::fpmath_mode math_mode) { _math_mode = math_mode; }
  /// Getting floating point math mode specified in the given convolution descriptor.
  /// \returns Value of math mode.
  ::dnnl::fpmath_mode get_math_mode() { return _math_mode; }
  /// Setting a 2D convolution descriptor with given parameters.
  /// \param [in] padding_h Value of height of padding.
  /// \param [in] padding_w Value of width of padding.
  /// \param [in] stride_h Value of height of stride.
  /// \param [in] stride_w Value of width of stride.
  /// \param [in] dilate_h Value of height of dilate.
  /// \param [in] dilate_w Value of width of dilate.
  void set(int padding_h, int padding_w, int stride_h, int stride_w,
           int dilate_h, int dilate_w) {
    _strides = {stride_h, stride_w};
    _dilates = {dilate_h - 1, dilate_w - 1};
    _paddings = {padding_h, padding_w};
  }
  /// Setting a ND convolution descriptor with given parameters.
  /// \param [in] ndims Dimension of the convolution operation.
  /// \param [in] paddings Array of dimension ndims containing the padding size of
  /// each dimension.
  /// \param [in] strides Array of dimension ndims containing the stride size of
  /// each dimension.
  /// \param [in] dilates Array of dimension ndims containing the kernel size of
  /// each dimension.
  void set(int ndims, int paddings[], int strides[], int dilates[]) {
    _strides = std::vector<int64_t>(strides, strides + ndims);
    _paddings = std::vector<int64_t>(paddings, paddings + ndims);
    _dilates = std::vector<int64_t>(dilates, dilates + ndims);
    for (auto &dilate : _dilates) {
      dilate--;
    }
  }
  /// Getting parameters from a 2D convolution descriptor.
  /// \param [out] padding_h Value of height of padding.
  /// \param [out] padding_w Value of width of padding.
  /// \param [out] stride_h Value of height of stride.
  /// \param [out] stride_w Value of width of stride.
  /// \param [out] dilate_h Value of height of dilate.
  /// \param [out] dilate_w Value of width of dilate.
  void get(int *padding_h, int *padding_w, int *stride_h, int *stride_w,
           int *dilate_h, int *dilate_w) const {
    *dilate_h = _dilates[0];
    *dilate_w = _dilates[1];
    *padding_h = _paddings[0];
    *padding_w = _paddings[1];
    *stride_h = _strides[0];
    *stride_w = _strides[1];
  }
  /// Getting parameters from a ND convolution descriptor.
  /// \param [in] requested_ndims Requested number of dimensions to get from a
  /// given convolution descriptor.
  /// \param [out] ndims Dimension of the pooling operation.
  /// \param [out] paddings Array of dimension ndims containing the padding size
  /// of each dimension.
  /// \param [out] strides Array of dimension ndims containing the stride size of
  /// each dimension.
  /// \param [out] dilates Array of dimension ndims containing the dilate size of
  /// each dimension.
  void get(int requested_ndims, int *ndims, int paddings[], int strides[],
           int dilates[]) const {
    *ndims = _strides.size();
    for (int i = 0; i < requested_ndims; i++) {
      dilates[i] = _dilates[i];
      paddings[i] = _paddings[i];
      strides[i] = _strides[i];
    }
  }
  /// Getting the stride parameter from a convolution descriptor.
  /// \returns Array of dimension ndims containing the stride size of each
  /// dimension.
  const std::vector<int64_t> &get_stride() const { return _strides; }
  /// Getting the kernel parameter from a convolution descriptor.
  /// \returns Array of dimension ndims containing the dilate size of each
  /// dimension.
  const std::vector<int64_t> &get_dilate() const { return _dilates; }
  /// Getting the padding parameter from a convolution descriptor.
  /// \returns Array of dimension ndims containing the padding size of each
  /// dimension.
  const std::vector<int64_t> &get_padding() const { return _paddings; }
  /// Getting the output dimensions of a memory after 2D convolution has been
  /// applied.
  /// \param [in] desc Input memory descriptor.
  /// \param [in] weight_desc Input weight memory descriptor.
  /// \param [out] out_n Number of images.
  /// \param [out] out_c Number of channels.
  /// \param [out] out_h Height of images.
  /// \param [out] out_w Width of images.
  void get_forward_output_dim(const memory_desc_ext &desc,
                              const memory_desc_ext &weight_desc, int *out_n,
                              int *out_c, int *out_h, int *out_w) const {
    auto dims = desc.get_dims();
    auto weight_dims = weight_desc.get_dims();
    *out_n = dims[0];
    *out_c = weight_dims[0];
    *out_h = 1 + (dims[2] + 2 * _paddings[0] -
                  (1 + (_dilates[0] * (weight_dims[2] - 1)))) /
                     _strides[0];
    *out_w = 1 + (dims[3] + 2 * _paddings[1] -
                  (1 + (_dilates[1] * (weight_dims[3] - 1)))) /
                     _strides[1];
  }
  /// Getting the output dimensions of a memory after ND convolution has been
  /// applied.
  /// \param [in] desc Input memory descriptor.
  /// \param [in] weight_desc Input weight memory descriptor.
  /// \param [out] ndims Dimension of the memory.
  /// \param [out] out_dims Array of dimension requested_ndims that contain
  /// the size of each memory dimension.
  void get_forward_output_dim(const memory_desc_ext &desc,
                              const memory_desc_ext &weight_desc, int ndims,
                              int out_dims[]) const {
    assert(ndims >= 4 && "ndims is at least 4.");
    auto dims = desc.get_dims();
    auto weight_dims = weight_desc.get_dims();
    out_dims[0] = dims[0];
    out_dims[1] = weight_dims[1];
    for (int i = 2; i < ndims; i++) {
      out_dims[i] = 1 + (dims[i] + 2 * _paddings[i - 2] -
                         (1 + (_dilates[i - 2] * (weight_dims[i] - 1)))) /
                            _strides[i - 2];
    }
  }

  convolution_desc &operator=(std::nullptr_t) {
    return *this = convolution_desc();
  }

  operator bool() const {
    return !(_strides.size() == 0
             && _dilates.size() == 0
             && _paddings.size() == 0);
  }
};

/// An enum class representing rnn mode.
enum class rnn_mode { vanilla_relu, vanilla_tanh, lstm, gru };

/// An enum class representing rnn bias mode.
enum class rnn_bias_mode { none, single };

/// An enum class representing rnn direction.
enum class rnn_direction {unidirectional, bidirectional};

/// A class holding description for a RNN operation.
class rnn_desc {
  rnn_mode _mode;
  rnn_bias_mode _bias_mode;
  rnn_direction _direction;
  dpct::library_data_t _dt;
  int _input_size;
  int _hidden_size;
  int _projection_size;
  int _layer_size;

public:
  void set(rnn_mode mode, rnn_bias_mode bias_mode, rnn_direction direction,
           dpct::library_data_t dt, int input_size, int hidden_size,
           int projection_size, int layer_size) {
    _mode = mode;
    _bias_mode = bias_mode;
    _direction = direction;
    _input_size = input_size;
    _hidden_size = hidden_size;
    _projection_size = projection_size;
    _layer_size = layer_size;
    _dt = dt;
  }
  void get(rnn_mode *mode, rnn_bias_mode *bias_mode, rnn_direction *direction,
           dpct::library_data_t *dt, int *input_size, int *hidden_size,
           int *projection_size, int *layer_size) const {
    *mode = _mode;
    *bias_mode = _bias_mode;
    *direction = _direction;
    *input_size = _input_size;
    *hidden_size = _hidden_size;
    *projection_size = _projection_size;
    *layer_size = _layer_size;
    *dt = _dt;
  }
};

/// A class holding description for a Dropout operation.
class dropout_desc {
  struct dropout_desc_imp {
    float _p = 0.5f;
    unsigned long long _seed = 1;
    void *_state = nullptr;
    std::vector<std::uint8_t> _host_state;
    rng_engine_t _rng_engine;
    dropout_desc_imp() : _rng_engine(dpct::get_default_queue(), 1) {}
  };
  std::shared_ptr<dropout_desc_imp> _imp;

  void generate(sycl::queue *q, std::int64_t required_state_size,
                std::int64_t num, void *buffer) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
    sycl::event e_gen = oneapi::mkl::rng::generate(
        oneapi::mkl::rng::bernoulli<std::int32_t>(1.f - _imp->_p),
        _imp->_rng_engine, num, (std::int32_t *)buffer);
    sycl::event e_save = q->submit([&](sycl::handler &cgh) {
      cgh.depends_on(e_gen);
      cgh.host_task([=] {
        oneapi::mkl::rng::save_state(_imp->_rng_engine,
                                     _imp->_host_state.data());
      });
    });
    q->memcpy(_imp->_state, _imp->_host_state.data(), required_state_size,
              e_save);
#endif
  }
public:
  operator bool() const {
    return bool(_imp);
  }
  dropout_desc &operator=(std::nullptr_t) {
    _imp.reset();
    return *this;
  }
  /// Initializing a dropout descriptor.
  void init(){
    _imp = std::make_shared<dropout_desc_imp>();
  }
  /// Setting a dropout descriptor with given parameters.
  /// \param [in] engine Engine of the dropout operation.
  /// \param [in] p Probability of value set to zero.
  /// \param [in] state Memory that store random generator state.
  /// \param [in] state_size Required size to store random generator state.
  /// \param [in] seed Seed to initialize conditions of the generator state.
  void set(engine_ext &engine, float p, void *state, size_t state_size,
           unsigned long long seed);
  /// Getting parameters from a dropout descriptor.
  /// \param [in] engine Engine of the dropout operation.
  /// \param [in] p Probability of value set to zero.
  /// \param [in] state Memory that store random generator state.
  /// \param [in] seed Seed to initialize conditions of the generator state.
  void get(float *p, void **states, unsigned long long *seed) const noexcept {
    *seed = _imp->_seed;
    *states = _imp->_state;
    *p = _imp->_p;
  }
  /// Getting the probability of value set to zero.
  /// \returns Probability.
  float get_probability() const noexcept { return _imp->_p; }
  /// Restoreing a dropout descriptor from stored state.
  /// \param [in] engine Engine of the dropout operation.
  /// \param [in] p Probability of value set to zero.
  /// \param [in] state Memory that store random generator state.
  /// \param [in] state_size Required size to store random generator state.
  /// \param [in] seed Seed to initialize conditions of the generator state.
  void restore(engine_ext &engine, float p, void *state, size_t state_size,
               unsigned long long seed);
  friend class engine_ext;
};

namespace detail {
typedef std::string primitive_cache_key_type;
typedef std::list<primitive_cache_key_type> usage_list_type;
struct primitive_cache_value_type {
  ::dnnl::primitive *_primitive;
  std::unordered_map<int, ::dnnl::memory> *_args;
  usage_list_type::iterator _usage_it;
  std::function<void(::dnnl::primitive *)> _destructor;
  sycl::event _e;
  sycl::queue _q;
  primitive_cache_value_type(
      ::dnnl::primitive *primitive,
      std::unordered_map<int, ::dnnl::memory> *args,
      usage_list_type::iterator usage_it,
      std::function<void(::dnnl::primitive *)> destructor, sycl::event e,
      sycl::queue q)
      : _primitive(primitive), _args(args), _usage_it(usage_it),
        _destructor(destructor), _e(e), _q(q) {}
};
struct primitive_and_args {
  ::dnnl::primitive *primitive;
  std::unordered_map<int, ::dnnl::memory> *args;
};
typedef std::unordered_map<primitive_cache_key_type,
                           std::shared_ptr<primitive_cache_value_type>>
    cache_map_type;

// The primitive cache uses LRU replacement policy, and the default cache
// capacity is 1024.
class primitive_cache {
  int _capacity = 1024;
  usage_list_type usage;
  cache_map_type cache_map;
  void touch(cache_map_type::iterator it, sycl::event e = {},
             bool update_event = false) {
    if (it->second->_usage_it != usage.begin()) {
      const primitive_cache_key_type &key = it->first;
      usage.erase(it->second->_usage_it);
      usage.push_front(key);
      it->second->_usage_it = usage.begin();
    }
    if (update_event) {
      it->second->_e = e;
    }
  }

public:
  std::shared_ptr<primitive_cache_value_type>
  get(const primitive_cache_key_type &key) {
    auto it = cache_map.find(key);
    if (it == cache_map.end()) {
      return nullptr;
    }
    touch(it);
    return it->second;
  }
  void put(const primitive_cache_key_type &key, ::dnnl::primitive *value,
           std::unordered_map<int, ::dnnl::memory> *args,
           std::function<void(::dnnl::primitive *)> destructor, sycl::event e,
           sycl::queue *q) {
    auto it = cache_map.find(key);
    if (it != cache_map.end()) {
      touch(it, e, true);
    } else {
      if (cache_map.size() == _capacity) {
        auto v = *(cache_map.find(usage.back())->second);
        v._q.submit([=](sycl::handler &cgh) {
          cgh.depends_on(v._e);
          cgh.host_task([=] {
            delete v._args;
            v._destructor(v._primitive);
          });
        });
        cache_map.erase(usage.back());
        usage.pop_back();
      }
      usage.push_front(key);
      cache_map[key] = std::make_shared<primitive_cache_value_type>(
          value, args, usage.begin(), destructor, e, *q);
    }
  }
};
} // namespace detail

/// A class holding the oneDNN engine.
class engine_ext {
  struct output_argument_info {
    float _alpha;
    float _beta;
    int _name;
    memory_desc_ext _desc;
    void *_data;
    output_argument_info(float alpha, float beta, int name,
                         memory_desc_ext desc, void *data)
        : _alpha(alpha), _beta(beta), _name(name), _desc(desc), _data(data) {}
    output_argument_info(float alpha, float beta, memory_desc_ext desc,
                         void *data)
        : _alpha(alpha), _beta(beta), _name(0), _desc(desc), _data(data) {}
  };
  struct buffer_info {
    size_t capacity = 0;
    uint8_t *buffer = nullptr;
    size_t usage = 0;
    sycl::queue q;
    sycl::event deps;
    size_t primitive_depth = 0;
  };
  struct internal_resource {
    std::int64_t random_engine_state_size = -1;
    buffer_info binfo;
  };
  std::shared_ptr<::dnnl::engine> _eng = nullptr;
  std::shared_ptr<::dnnl::stream> _s = nullptr;
  sycl::queue *_q = nullptr;
  unsigned int _engine_id = 0;
  static thread_local unsigned int _engine_count;
  static thread_local std::map<void *, ::dnnl::memory> _workspace_map;
  static thread_local std::map<sycl::queue *,
                               std::shared_ptr<internal_resource>>
      _internal_resource_cache;
  static thread_local detail::primitive_cache _primitive_cache;
  ::dnnl::memory &get_workspace(void *key) { return _workspace_map[key]; }
  void insert_workspace(void *key, ::dnnl::memory workspace) {
    _workspace_map[key] = workspace;
  }
  const ::dnnl::stream &get_stream() const { return *_s; }
  const ::dnnl::engine &get_engine() const { return *_eng; }

  void *allocate(const memory_desc_ext &desc, int count = 1);
  void *allocate(size_t size);
  std::shared_ptr<internal_resource> get_internal_resource(sycl::queue *q){
    auto it = _internal_resource_cache.find(_q);
    if (it == _internal_resource_cache.end()) {
      return _internal_resource_cache[_q] = std::make_shared<internal_resource>();
    }
    return it->second;
  }
  void enter_primitive(size_t request_buffer_size = 0) {
    auto &info = get_internal_resource(_q)->binfo;
    if (info.primitive_depth == 0) {
      info.usage = 0;
      if (request_buffer_size > info.capacity) {
        if (info.buffer && (info.capacity != 0)) {
          auto ainfo = info;
          ainfo.q.submit([=](sycl::handler &cgh) {
            cgh.depends_on(ainfo.deps);
            cgh.host_task([=] { sycl::free(ainfo.buffer, ainfo.q); });
          });
        }
        size_t new_buffer_capacity =
            std::max(request_buffer_size, info.capacity * 2);
        info.capacity = new_buffer_capacity;
        info.buffer = (uint8_t *)sycl::malloc_device(new_buffer_capacity, *_q);
        info.q = *_q;
        info.deps = sycl::event();
      }
    }
    info.primitive_depth++;
  }
  sycl::event exit_primitive(const sycl::event &e) {
    auto &info = get_internal_resource(_q)->binfo;
    info.primitive_depth--;
    if ((info.primitive_depth == 0) && info.usage) {
      info.deps = e;
    }
    return e;
  }
  ::dnnl::memory::desc
  compress_spatial_dimensions_to_channel(const ::dnnl::memory::desc &desc);
  ::dnnl::memory::desc
  get_bn_scale_bias_mean_var_desc(const ::dnnl::memory::desc &desc,
                                  batch_normalization_mode mode);
  sycl::event batch_normalization_backward_internal(
      batch_normalization_mode mode, float epsilon, float alpha_data,
      const memory_desc_ext &src_desc, void *src,
      const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta_data,
      const memory_desc_ext &diff_src_desc, void *diff_src, float alpha_param,
      const memory_desc_ext &diff_scale_bias_desc, void *scale, void *bias,
      float beta_param, void *diff_scale, void *diff_bias,
      const memory_desc_ext &mean_var_desc, void *saved_mean, void *saved_var);
  sycl::event batch_normalization_forward_internal(
      bool is_infer, batch_normalization_mode mode, float epsilon, float factor,
      float alpha, const memory_desc_ext &src_desc, void *src, float beta,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &scale_bias_desc, void *scale, void *bias,
      const memory_desc_ext &mean_var_desc, void *saved_mean, void *saved_var,
      void *running_mean, void *running_var);
  ::dnnl::memory::desc
  transfer_memory_desc_to_channel_major_format(const ::dnnl::memory::desc &desc);
  ::dnnl::memory::desc
  bn_reorder_memory_to_channel_major_format(
      bool is_input, ::dnnl::memory::desc &desc, void *src, void **cache);
  ::dnnl::memory::desc
  transfer_memory_desc_to_format_tag_any(const ::dnnl::memory::desc &desc){
    return ::dnnl::memory::desc(desc.get_dims(), desc.get_data_type(),
                                ::dnnl::memory::format_tag::any);
  }
  void allocate_and_reorder_memory_to_optimal(::dnnl::memory::desc &from_desc,
                                              void *&from,
                                              ::dnnl::memory::desc &to_desc,
                                              void *&to) {
    if (from_desc != to_desc) {
      to = allocate(to_desc);
      async_reorder(1.f, from_desc, from, 0.f, to_desc, to);
    }
  }
  template <typename primitive_type, typename... args_type>
  std::pair<detail::primitive_cache_key_type, detail::primitive_and_args>
  create_primitive_args_or_get(args_type &&...args);
  template <typename primitive_type>
  typename primitive_type::primitive_desc
  get_primitive_desc(::dnnl::primitive *p);
  template <typename primitive_type, typename... args_type>
  typename primitive_type::primitive_desc
  create_primitive_desc(args_type &&...args);
  template <typename T>
  void generate_cache_key(std::string &key_buffer, const T &arg);
  template <typename T, typename... args_type>
  void generate_cache_key(std::string &key_buffer, const T &first_arg,
                          const args_type &...args);
  void insert_arg(std::unordered_map<int, ::dnnl::memory> *args, int name,
                  const ::dnnl::memory::desc &desc, void *data) {
    auto it = args->find(name);
    if (it != args->end()) {
      it->second.set_data_handle(data);
    } else {
      args->insert({name, ::dnnl::memory(desc, *_eng, data)});
    }
  }
  void insert_arg(std::unordered_map<int, ::dnnl::memory> *args, int name,
                  const ::dnnl::memory &mem) {
    (*args)[name] = mem;
  }
  sycl::event execute_rnn_forward_primitive(
      rnn_mode mode, ::dnnl::prop_kind kind, ::dnnl::rnn_direction direction,
      rnn_bias_mode bias_mode, ::dnnl::memory::data_type dt,
      ::dnnl::memory::format_tag tag, int seq_length, int batch_size, int src_c,
      int dst_c, int layer_size, int direction_num, int hidden_size,
      int gate_num, int projection_size, std::vector<void *> &data,
      std::vector<int> &offset, int iter_num, size_t *weight_size = nullptr,
      size_t *workspace_size = nullptr, size_t *scratchpad_size = nullptr);

  sycl::event rnn_forward_internal(
      const rnn_desc &desc, ::dnnl::prop_kind kind,
      const memory_desc_ext &src_desc, void *src,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &iter_desc, void *src_iter, void *dst_iter,
      const memory_desc_ext &iter_c_desc, void *src_iter_c, void *dst_iter_c,
      size_t weight_size, void *weight, size_t workspace_size, void *workspace,
      size_t scratchpad_size, void *scratchpad, bool is_get_execution_args,
      size_t *weight_size_query, size_t *workspace_size_query,
      size_t *scratchpad_size_query);

  sycl::event execute_rnn_backward_primitive(
      rnn_mode mode, ::dnnl::rnn_direction direction, rnn_bias_mode bias_mode,
      ::dnnl::memory::data_type dt, ::dnnl::memory::format_tag tag,
      int seq_length, int batch_size, int src_c, int dst_c, int layer_size,
      int direction_num, int hidden_size, int gate_num, int projection_size,
      std::vector<void *> &data, std::vector<int> &offset, int iter_num);
  bool
  scale_parameter_preprocess(const std::vector<output_argument_info> &args);
  template <typename primitive_type>
  sycl::event
  execute_primitive(const std::pair<detail::primitive_cache_key_type,
                                    detail::primitive_and_args> &primitive,
                    const std::vector<output_argument_info> &extra_args = {});
  template <typename T>
  sycl::event fill_with_type(sycl::queue *q, void *src, const void *value,
                             size_t size_with_byte) {
    return q->fill<T>(static_cast<T *>(src), *static_cast<const T *>(value),
                      size_with_byte / sizeof(T));
  }
  template <typename T> struct no_zero_op {
    T operator()(T e) {
      if (!e) {
        return 1;
      }
      return e;
    }
  };
  template <typename T>
  void transform_no_zero_with_type(sycl::queue *q, void *src, void *dst,
                                   size_t num) {
    std::transform(oneapi::dpl::execution::make_device_policy(*q),
                   static_cast<T *>(src), static_cast<T *>(src) + num,
                   static_cast<T *>(dst), no_zero_op<T>());
  }
  void transform_no_zero(const memory_desc_ext &desc, void *src, void *dst);
  ::dnnl::memory::desc get_group_weight_desc(int group_count,
                                             const memory_desc_ext &weight_desc);
  void get_rnn_configuration(const ::dnnl::memory::desc &desc,
                             rnn_direction direction, rnn_mode mode,
                             dpct::library_data_t dt, int hidden_size,
                             ::dnnl::memory::data_type *dnnl_dt,
                             ::dnnl::memory::format_tag *tag,
                             int *projection_size, int *output_size,
                             int *seq_length, int *batch_size,
                             int *direction_num, int *gate_num);
public:
  engine_ext() {}
  operator bool() const {
    return bool(_eng) && bool(_s) && bool(_q);
  }
  engine_ext &operator=(std::nullptr_t) {
    _eng = nullptr;
    _s = nullptr;
    _q = nullptr;
    return *this;
  }
  /// Creating oneDNN engine.
  void create_engine() {
    _q = &dpct::get_current_device().default_queue();
    _eng = std::make_shared<::dnnl::engine>(::dnnl::sycl_interop::make_engine(
        dpct::get_current_device(), dpct::get_current_device().get_context()));
    _s = std::make_shared<::dnnl::stream>(
        ::dnnl::sycl_interop::make_stream(*_eng, *_q));
    _engine_id = _engine_count++;
  }
  /// Setting the user's SYCL queue for an oneDNN engine.
  /// \param [in] q Pointer to the SYCL queue.
  void set_queue(sycl::queue *q) {
    if (!q) {
      throw std::runtime_error("set_queue: pointer must not be nullptr.");
    }
    if (!_eng) {
      throw std::runtime_error("set_queue: current engine is invalid.");
    }
    if (q->get_context() != ::dnnl::sycl_interop::get_context(*_eng)) {
      throw std::runtime_error(
          "set_queue: queue is mismatch with current engine context.");
    }
    _q = q;
    _s = std::make_shared<::dnnl::stream>(
        ::dnnl::sycl_interop::make_stream(*_eng, *_q));
  }
  /// Retrieving the user's SYCL queue set in the oneDNN engine.
  /// \returns Pointer to the SYCL queue.
  sycl::queue *get_queue() const { return _q; }
  /// Setting all elements of a memory to a given value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] valuePtr Pointer to a single value.
  void fill(const memory_desc_ext &src_desc, void *src,
                   const void *valuePtr);
  /// Coping the scaled data from a memory to another memory with a different
  /// description.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  void reorder(float alpha, const memory_desc_ext &src_desc, void *src,
                      float beta, const memory_desc_ext &dst_desc, void *dst);
  /// Scaling all the elements of a memory by a given factor.
  /// \param [in] alpha Value to scaling factors.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [out] src Pointer to source data.
  void scale(float alpha, const memory_desc_ext &src_desc, void *src);
  /// Adding the scaled values of a memory to another memory.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  void sum(float alpha, const memory_desc_ext &src_desc, void *src,
                  float beta, const memory_desc_ext &dst_desc, void *dst);
  /// Computing a specified activation function value.
  /// \param [in] desc Activation descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  void activation_forward(activation_desc &desc, float alpha,
                                 const memory_desc_ext &src_desc, void *src,
                                 float beta, const memory_desc_ext &dst_desc,
                                 void *dst);
  /// Computing the gradient of a specified activation function.
  /// \param [in] desc Activation descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  void
  activation_backward(activation_desc &desc, float alpha,
                      const memory_desc_ext &dst_desc, void *dst,
                      const memory_desc_ext &diff_dst_desc, void *diff_dst,
                      const memory_desc_ext &src_desc, void *src, float beta,
                      const memory_desc_ext &diff_src_desc, void *diff_src);
  /// Computing a specified pooling function value.
  /// \param [in] desc Pooling descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [out] workspace Pointer to workspace generated from forward propagation.
  void pooling_forward(pooling_desc &desc, float alpha,
                              const memory_desc_ext &src_desc, void *src,
                              float beta, const memory_desc_ext &dst_desc,
                              void *dst, ::dnnl::memory *workspace = nullptr);
  /// Computing the gradient of a specified pooling function.
  /// \param [in] desc Activation descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential
  /// source data. 
  /// \param [in] workspace Pointer to workspace used for backward
  /// propagation.
  void pooling_backward(pooling_desc &desc, float alpha,
                               const memory_desc_ext &dst_desc, void *dst,
                               const memory_desc_ext &diff_dst_desc,
                               void *diff_dst, const memory_desc_ext &src_desc,
                               void *src, float beta,
                               const memory_desc_ext &diff_src_desc,
                               void *diff_src,
                               ::dnnl::memory *workspace = nullptr);
  /// Computing a specified softmax function value.
  /// \param [in] alg Softmax algorithm.
  /// \param [in] mode Softmax mode.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value. 
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data. 
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  void softmax_forward(softmax_algorithm alg, softmax_mode mode,
                              float alpha, const memory_desc_ext &src_desc,
                              void *src, float beta,
                              const memory_desc_ext &dst_desc, void *dst);
  /// Computing the gradient of a specified softmax function.
  /// \param [in] alg Softmax algorithm.
  /// \param [in] mode Softmax mode.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value. 
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  void softmax_backward(softmax_algorithm alg, softmax_mode mode,
                               float alpha, const memory_desc_ext &dst_desc,
                               void *dst, const memory_desc_ext &diff_dst_desc,
                               void *diff_dst, float beta,
                               const memory_desc_ext &diff_src_desc,
                               void *diff_src);
  /// Computing a specified local response normalization function value.
  /// \param [in] desc Local response normalization descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [out] workspace Pointer to workspace generated from forward
  /// propagation.
  void lrn_forward(lrn_desc &desc, float alpha,
                          const memory_desc_ext &src_desc, void *src,
                          float beta, const memory_desc_ext &dst_desc,
                          void *dst, ::dnnl::memory *workspace = nullptr);
  /// Computing the gradient of a specified local response normalization
  /// function.
  /// \param [in] desc Local response normalization descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed value.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] workspace Pointer to workspace used for backward propagation.
  void lrn_backward(lrn_desc &desc, float alpha,
                           const memory_desc_ext &dst_desc, void *dst,
                           const memory_desc_ext &diff_dst_desc, void *diff_dst,
                           const memory_desc_ext &src_desc, void *src,
                           float beta, const memory_desc_ext &diff_src_desc,
                           void *diff_src, ::dnnl::memory *workspace = nullptr);
  /// Setting all elements of a memory to a given value asynchronously.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] valuePtr Pointer to a single value.
  /// \returns An event representing the fill operations.
  sycl::event async_fill(const memory_desc_ext &src_desc, void *src,
                   const void *valuePtr);
  /// Coping the scaled data from a memory to another memory with a different
  /// description asynchronously.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the reorder operations.
  sycl::event async_reorder(float alpha, const memory_desc_ext &src_desc, void *src,
                      float beta, const memory_desc_ext &dst_desc, void *dst);
  /// Scaling all the elements of a memory by a given factor asynchronously.
  /// \param [in] alpha Value to scaling factors.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [out] src Pointer to source data.
  /// \returns An event representing the scale operations.
  sycl::event async_scale(float alpha, const memory_desc_ext &src_desc, void *src);
  /// Adding the scaled values of a memory to another memory asynchronously.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the sum operations.
  sycl::event async_sum(float alpha, const memory_desc_ext &src_desc, void *src,
                  float beta, const memory_desc_ext &dst_desc, void *dst);

  /// Perform specified binary operation asynchronously.
  /// \param [in] op Specified binary operation.
  /// \param [in] alpha_0 Value to scaling factors used to scale the src_0
  /// value.
  /// \param [in] src_desc_0 Source 0 memory descriptor.
  /// \param [in] src_0 Pointer to source 0 data.
  /// \param [in] alpha_1 Value to scaling factors used to scale the src_1
  /// value.
  /// \param [in] src_desc_1 Source 1 memory descriptor.
  /// \param [in] src_1 Pointer to source 1 data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the binary operations.
  sycl::event async_binary(binary_op op, float alpha_0,
                     const memory_desc_ext &src_desc_0, void *src_0,
                     float alpha_1, const memory_desc_ext &src_desc_1,
                     void *src_1, float beta, const memory_desc_ext &dst_desc,
                     void *dst);

  /// Perform specified binary operation asynchronously.
  /// \param [in] op Specified reduction operation.
  /// \param [in] alpha Value to scaling factors used to scale the data
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the reduction operations.
  sycl::event async_reduction(reduction_op op, float alpha,
                        const memory_desc_ext &src_desc, void *src, float beta,
                        const memory_desc_ext &dst_desc, void *dst);
  /// Computing a specified activation function value asynchronously.
  /// \param [in] desc Activation descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the activation forward operations.
  sycl::event async_activation_forward(activation_desc &desc, float alpha,
                                 const memory_desc_ext &src_desc, void *src,
                                 float beta, const memory_desc_ext &dst_desc,
                                 void *dst);
  /// Computing the gradient of a specified activation function asynchronously.
  /// \param [in] desc Activation descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \returns An event representing the activation backward operations.
  sycl::event
  async_activation_backward(activation_desc &desc, float alpha,
                      const memory_desc_ext &dst_desc, void *dst,
                      const memory_desc_ext &diff_dst_desc, void *diff_dst,
                      const memory_desc_ext &src_desc, void *src, float beta,
                      const memory_desc_ext &diff_src_desc, void *diff_src);
  /// Computing a specified pooling function value asynchronously.
  /// \param [in] desc Pooling descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [out] workspace Pointer to workspace generated from forward propagation.
  /// \returns An event representing the pooling forward operations.
  sycl::event async_pooling_forward(pooling_desc &desc, float alpha,
                              const memory_desc_ext &src_desc, void *src,
                              float beta, const memory_desc_ext &dst_desc,
                              void *dst, ::dnnl::memory *workspace = nullptr);
  /// Computing the gradient of a specified pooling function asynchronously.
  /// \param [in] desc Activation descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential
  /// source data. 
  /// \param [in] workspace Pointer to workspace used for backward
  /// propagation.
  /// \returns An event representing the pooling backward operations.
  sycl::event async_pooling_backward(pooling_desc &desc, float alpha,
                               const memory_desc_ext &dst_desc, void *dst,
                               const memory_desc_ext &diff_dst_desc,
                               void *diff_dst, const memory_desc_ext &src_desc,
                               void *src, float beta,
                               const memory_desc_ext &diff_src_desc,
                               void *diff_src,
                               ::dnnl::memory *workspace = nullptr);
  /// Computing a specified softmax function value asynchronously.
  /// \param [in] alg Softmax algorithm.
  /// \param [in] mode Softmax mode.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value. 
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data. 
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the softmax forward operations.
  sycl::event async_softmax_forward(softmax_algorithm alg, softmax_mode mode,
                              float alpha, const memory_desc_ext &src_desc,
                              void *src, float beta,
                              const memory_desc_ext &dst_desc, void *dst);
  /// Computing the gradient of a specified softmax function asynchronously.
  /// \param [in] alg Softmax algorithm.
  /// \param [in] mode Softmax mode.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value. 
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \returns An event representing the softmax backward operations.
  sycl::event async_softmax_backward(softmax_algorithm alg, softmax_mode mode,
                               float alpha, const memory_desc_ext &dst_desc,
                               void *dst, const memory_desc_ext &diff_dst_desc,
                               void *diff_dst, float beta,
                               const memory_desc_ext &diff_src_desc,
                               void *diff_src);
  /// Computing a specified local response normalization function value
  /// asynchronously.
  /// \param [in] desc Local response normalization descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [out] workspace Pointer to workspace generated from forward
  /// propagation.
  /// \returns An event representing the lrn forward operations.
  sycl::event async_lrn_forward(lrn_desc &desc, float alpha,
                          const memory_desc_ext &src_desc, void *src,
                          float beta, const memory_desc_ext &dst_desc,
                          void *dst, ::dnnl::memory *workspace = nullptr);
  /// Computing the gradient of a specified local response normalization
  /// function asynchronously.
  /// \param [in] desc Local response normalization descriptor.
  /// \param [in] alpha Value to scaling factors used to scale the computed value.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the differential destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] workspace Pointer to workspace used for backward propagation.
  /// \returns An event representing the lrn backward operations.
  sycl::event async_lrn_backward(lrn_desc &desc, float alpha,
                           const memory_desc_ext &dst_desc, void *dst,
                           const memory_desc_ext &diff_dst_desc, void *diff_dst,
                           const memory_desc_ext &src_desc, void *src,
                           float beta, const memory_desc_ext &diff_src_desc,
                           void *diff_src, ::dnnl::memory *workspace = nullptr);

  /// Derives a memory descriptor for the batch normalization scale, bias, mean,
  /// variance from the source memory descriptor and batch normalization mode.
  /// \param [out] desc Derived memory descriptor.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] mode Batch normalization mode.
  static void derive_batch_normalization_memory_desc(memory_desc_ext &desc,
                                              const memory_desc_ext &src_desc,
                                              batch_normalization_mode mode);

  /// Derives a memory descriptor for the batch normalization scale, bias, mean,
  /// variance from the source memory descriptor and batch normalization mode.
  /// \param [out] scale_bias_desc Derived scale and bias memory descriptor.
  /// \param [out] mean_var_desc Derived mean and var memory descriptor.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] mode Batch normalization mode.
  static void derive_batch_normalization_memory_desc(memory_desc_ext &scale_bias_desc,
                                             memory_desc_ext &mean_var_desc,
                                             const memory_desc_ext &src_desc,
                                             batch_normalization_mode mode);

  /// Get the size of workspace that needed by batch normalization. The data stored
  /// in workspace must be preserved between forward and backward.
  /// \param [in] ops Batch normalization operation mode. This mode can set to
  /// perform only batch normalization, or batch normalization followed by
  /// activation, or batch normalization followed by element-wise addition and
  /// activation.
  /// \param [in] src_desc Source memory descriptor.
  /// \returns Size of workspace.
  size_t get_batch_normalization_workspace_size(
    batch_normalization_ops ops, const memory_desc_ext &src_desc);

  /// Computing a specified batch normalization inference stage function value
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] scale_bias_mean_var_desc Scale, bias, mean, variance memory
  /// descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [in] mean Pointer to mean data.
  /// \param [in] var Pointer to variance data.
  /// \returns An event representing the batch normalization forward operations.
  sycl::event async_batch_normalization_forward_inference(
      batch_normalization_mode mode, float epsilon, float alpha,
      const memory_desc_ext &src_desc, void *src, float beta,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &scale_bias_mean_var_desc, void *scale, void *bias,
      void *mean, void *var);

  /// Computing a specified batch normalization inference stage function value
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] ops Batch normalization operation mode. This mode can set to
  /// perform only batch normalization, or batch normalization followed by
  /// activation, or batch normalization followed by element-wise addition and
  /// activation.
  /// \param [in] adesc Activation operation descriptor.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] summand_desc Summand memory descriptor.
  /// \param [in] summand Pointer to summand data.
  /// \param [in] scale_bias_desc Scale, bias memory descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [in] mean_var_desc Mean, variance memory descriptor.
  /// \param [in] mean Pointer to mean data.
  /// \param [in] var Pointer to variance data.
  /// \returns An event representing the batch normalization forward operations.
  sycl::event async_batch_normalization_forward_inference(
      batch_normalization_mode mode, batch_normalization_ops ops,
      activation_desc &adesc, float epsilon, float alpha,
      const memory_desc_ext &src_desc, void *src, float beta,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &summand_desc, void *summand,
      const memory_desc_ext &scale_bias_desc, void *scale, void *bias,
      const memory_desc_ext &mean_var_desc, void *mean, void *var);

  /// Computing a specified batch normalization training stage function value
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] factor Factor value used in running mean and variance
  /// computation.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] scale_bias_mean_var_desc Scale, bias, mean, variance memory
  /// descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [out] running_mean Pointer to running mean data.
  /// \param [out] running_var Pointer to running variance data.
  /// \param [out] saved_mean Pointer to optional cache to save mean data.
  /// \param [out] saved_var Pointer to optional cache to save variance data.
  /// \returns An event representing the batch normalization forward operations.
  sycl::event async_batch_normalization_forward_training(
      batch_normalization_mode mode, float epsilon, float factor, float alpha,
      const memory_desc_ext &src_desc, void *src, float beta,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &scale_bias_mean_var_desc, void *scale, void *bias,
      void *running_mean, void *running_var, void *saved_mean, void *saved_var);

  /// Computing a specified batch normalization training stage function value
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] ops Batch normalization operation mode. This mode can set to
  /// perform only batch normalization, or batch normalization followed by
  /// activation, or batch normalization followed by element-wise addition and
  /// activation.
  /// \param [in] adesc Activation operation descriptor.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] factor Factor value used in running mean and variance
  /// computation.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] summand_desc Summand memory descriptor.
  /// \param [in] summand Pointer to summand data.
  /// \param [in] scale_bias_mean_var_desc Scale, bias, mean, variance memory
  /// descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [out] running_mean Pointer to running mean data.
  /// \param [out] running_var Pointer to running variance data.
  /// \param [out] saved_mean Pointer to optional cache to save mean data.
  /// \param [out] saved_var Pointer to optional cache to save variance data.
  /// \param [in] workspace_size Size of workspace.
  /// \param [out] workspace Pointer to workspace generated from forward
  /// propagation.
  /// \returns An event representing the batch normalization forward operations.
  sycl::event async_batch_normalization_forward_training(
      batch_normalization_mode mode, batch_normalization_ops ops,
      activation_desc &adesc, float epsilon, float factor, float alpha,
      const memory_desc_ext &src_desc, void *src, float beta,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &summand_desc, void *summand,
      const memory_desc_ext &scale_bias_mean_var_desc, void *scale, void *bias,
      void *running_mean, void *running_var, void *saved_mean, void *saved_var,
      size_t workspace_size, void *workspace);

  /// Computing a specified batch normalization training stage function value
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] ops Batch normalization operation mode. This mode can set to
  /// perform only batch normalization, or batch normalization followed by
  /// activation, or batch normalization followed by element-wise addition and
  /// activation.
  /// \param [in] adesc Activation operation descriptor.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] factor Factor value used in running mean and variance
  /// computation.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] summand_desc Summand memory descriptor.
  /// \param [in] summand Pointer to summand data.
  /// \param [in] scale_bias_desc Scale, bias memory descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [in] mean_var_desc Mean, variance memory descriptor.
  /// \param [out] running_mean Pointer to running mean data.
  /// \param [out] running_var Pointer to running variance data.
  /// \param [out] saved_mean Pointer to optional cache to save mean data.
  /// \param [out] saved_var Pointer to optional cache to save variance data.
  /// \param [in] workspace_size Size of workspace.
  /// \param [out] workspace Pointer to workspace generated from forward
  /// propagation.
  /// \returns An event representing the batch normalization forward operations.
  sycl::event async_batch_normalization_forward_training(
      batch_normalization_mode mode, batch_normalization_ops ops,
      activation_desc &adesc, float epsilon, float factor, float alpha,
      const memory_desc_ext &src_desc, void *src, float beta,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &summand_desc, void *summand,
      const memory_desc_ext &scale_bias_desc, void *scale, void *bias,
      const memory_desc_ext &mean_var_desc, void *running_mean, void *running_var,
      void *saved_mean, void *saved_var, size_t workspace_size, void *workspace);

  /// Computing the gradient of a specified batch normalization function asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] alpha_data Value to scaling factors used to scale the computed
  /// data value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta_data Value to scaling factors used to scale the prior value
  /// in the data memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] alpha_param Value to scaling factors used to scale the computed
  /// parameter value.
  /// \param [in] diff_scale_bias_mean_var_desc Differential scale, bias, mean,
  /// variance memory descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] beta_param Value to scaling factors used to scale the prior value
  /// in the parameter memory.
  /// \param [in] diff_scale Pointer to differential scale data.
  /// \param [in] diff_bias Pointer to differential bias data.
  /// \param [in] saved_mean Pointer to optional cache saved mean data in forward.
  /// \param [in] saved_var Pointer to optional cache saved variance data in forward.
  /// \returns An event representing the batch normalization backward operations.
  sycl::event async_batch_normalization_backward(
      batch_normalization_mode mode, float epsilon, float alpha_data,
      const memory_desc_ext &src_desc, void *src,
      const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta_data,
      const memory_desc_ext &diff_src_desc, void *diff_src, float alpha_param,
      const memory_desc_ext &diff_scale_bias_mean_var_desc, void *scale,
      float beta_param, void *diff_scale, void *diff_bias, void *saved_mean,
      void *saved_var);

  /// Computing the gradient of a specified batch normalization function
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] ops Batch normalization operation mode. This mode can set to
  /// perform only batch normalization, or batch normalization followed by
  /// activation, or batch normalization followed by element-wise addition and
  /// activation.
  /// \param [in] adesc Activation operation descriptor.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] alpha_data Value to scaling factors used to scale the computed
  /// data value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta_data Value to scaling factors used to scale the prior value
  /// in the data memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] diff_summand_desc Differential summand memory descriptor.
  /// \param [out] diff_summand Pointer to differential summand data.
  /// \param [in] alpha_param Value to scaling factors used to scale the computed
  /// parameter value.
  /// \param [in] diff_scale_bias_mean_var_desc Differential scale, bias, mean,
  /// variance memory descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [in] beta_param Value to scaling factors used to scale the prior value
  /// in the parameter memory.
  /// \param [out] diff_scale Pointer to differential scale data.
  /// \param [out] diff_bias Pointer to differential bias data.
  /// \param [in] saved_mean Pointer to optional cache saved mean data in forward.
  /// \param [in] saved_var Pointer to optional cache saved variance data in forward.
  /// \param [in] workspace_size Size of workspace.
  /// \param [in] workspace Pointer to workspace used for backward propagation.
  /// \returns An event representing the batch normalization backward operations.
  sycl::event async_batch_normalization_backward(
      batch_normalization_mode mode, batch_normalization_ops ops,
      activation_desc &adesc, float epsilon, float alpha_data,
      const memory_desc_ext &src_desc, void *src,
      const memory_desc_ext &dst_desc, void *dst,
      const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta_data,
      const memory_desc_ext &diff_src_desc, void *diff_src,
      const memory_desc_ext &diff_summand_desc, void *diff_summand,
      float alpha_param, const memory_desc_ext &diff_scale_bias_mean_var_desc,
      void *scale, void *bias, float beta_param, void *diff_scale,
      void *diff_bias, void *saved_mean, void *saved_var,
      size_t workspace_size, void *workspace);

  /// Computing the gradient of a specified batch normalization function
  /// asynchronously.
  /// \param [in] mode Batch normalization mode.
  /// \param [in] ops Batch normalization operation mode. This mode can set to
  /// perform only batch normalization, or batch normalization followed by
  /// activation, or batch normalization followed by element-wise addition and
  /// activation.
  /// \param [in] adesc Activation operation descriptor.
  /// \param [in] epsilon Epsilon value used in computation.
  /// \param [in] alpha_data Value to scaling factors used to scale the computed
  /// data value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta_data Value to scaling factors used to scale the prior value
  /// in the data memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] diff_summand_desc Differential summand memory descriptor.
  /// \param [out] diff_summand Pointer to differential summand data.
  /// \param [in] alpha_param Value to scaling factors used to scale the computed
  /// parameter value.
  /// \param [in] diff_scale_bias_desc Differential scale, bias memory descriptor.
  /// \param [in] scale Pointer to scale data.
  /// \param [in] bias Pointer to bias data.
  /// \param [in] beta_param Value to scaling factors used to scale the prior value
  /// in the parameter memory.
  /// \param [out] diff_scale Pointer to differential scale data.
  /// \param [out] diff_bias Pointer to differential bias data.
  /// \param [in] mean_var_desc Differential mean, variance memory descriptor.
  /// \param [in] saved_mean Pointer to optional cache saved mean data in forward.
  /// \param [in] saved_var Pointer to optional cache saved variance data in forward.
  /// \param [in] workspace_size Size of workspace.
  /// \param [in] workspace Pointer to workspace used for backward propagation.
  /// \returns An event representing the batch normalization backward operations.
  sycl::event async_batch_normalization_backward(
      batch_normalization_mode mode, batch_normalization_ops ops,
      activation_desc &adesc, float epsilon, float alpha_data,
      const memory_desc_ext &src_desc, void *src, const memory_desc_ext &dst_desc,
      void *dst, const memory_desc_ext &diff_dst_desc, void *diff_dst,
      float beta_data, const memory_desc_ext &diff_src_desc, void *diff_src,
      const memory_desc_ext &diff_summand_desc, void *diff_summand,
      float alpha_param, const memory_desc_ext &diff_scale_bias_desc, void *scale,
      void *bias, float beta_param, void *diff_scale, void *diff_bias,
      const memory_desc_ext &mean_var_desc, void *saved_mean, void *saved_var,
      size_t workspace_size, void *workspace);

  /// Computing a specified convolution function value asynchronously.
  /// \param [in] desc Convolution descriptor.
  /// \param [in] alg Convolution algorithm.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] weight_desc Weight memory descriptor.
  /// \param [in] weight Pointer to weight data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the convolution forward operations.
  sycl::event async_convolution_forward(convolution_desc &desc, ::dnnl::algorithm alg,
                                  float alpha, const memory_desc_ext &src_desc,
                                  void *src, const memory_desc_ext &weight_desc,
                                  void *weight, float beta,
                                  const memory_desc_ext &dst_desc, void *dst);

  /// Computing a specified convolution function value asynchronously.
  /// \param [in] desc Convolution descriptor.
  /// \param [in] alg Convolution algorithm.
  /// \param [in] adesc Activation operation descriptor.
  /// \param [in] alpha_0 Value to scaling factors used to scale the data
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] weight_desc Weight memory descriptor.
  /// \param [in] weight Pointer to weight data.
  /// \param [in] alpha_1 Value to scaling factors used to scale the summand
  /// value.
  /// \param [in] summand_desc Summand memory descriptor.
  /// \param [in] summand Pointer to summand data.
  /// \param [in] bias_desc Bias memory descriptor.
  /// \param [in] bias Pointer to bias data.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \returns An event representing the convolution forward operations.
  sycl::event async_convolution_forward(
      convolution_desc &desc, ::dnnl::algorithm alg, activation_desc &adesc,
      float alpha_0, const memory_desc_ext &src_desc, void *src,
      const memory_desc_ext &weight_desc, void *weight, float alpha_1,
      const memory_desc_ext &summand_desc, void *summand,
      const memory_desc_ext &bias_desc, void *bias,
      const memory_desc_ext &dst_desc, void *dst);

  /// Computing the data gradient of a specified convolution function asynchronously.
  /// \param [in] desc Convolution descriptor.
  /// \param [in] alg Convolution algorithm.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] weight_desc Weight memory descriptor.
  /// \param [in] weight Pointer to weight data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \returns An event representing the convolution backward data operations.
  sycl::event async_convolution_backward_data(
      convolution_desc &desc, ::dnnl::algorithm alg, float alpha,
      const memory_desc_ext &weight_desc, void *weight,
      const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta,
      const memory_desc_ext &diff_src_desc, void *diff_src);

  /// Computing the weight gradient of a specified convolution function
  /// asynchronously.
  /// \param [in] desc Convolution descriptor.
  /// \param [in] alg Convolution algorithm.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] diff_weight_desc Differential weight memory descriptor.
  /// \param [out] diff_weight Pointer to differential weight data.
  /// \returns An event representing the convolution backward weight operations.
  sycl::event async_convolution_backward_weight(
      convolution_desc &desc, ::dnnl::algorithm alg, float alpha,
      const memory_desc_ext &src_desc, void *src,
      const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta,
      const memory_desc_ext &diff_weight_desc, void *diff_weight);

  /// Computing the bias gradient of a specified convolution function
  /// asynchronously.
  /// \param [in] alpha Value to scaling factors used to scale the computed
  /// value.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] beta Value to scaling factors used to scale the prior value
  /// in the destination memory.
  /// \param [in] diff_bias_desc Differential bias memory descriptor.
  /// \param [out] diff_bias Pointer to differential bias data.
  /// \returns An event representing the convolution backward bias operations.
  sycl::event async_convolution_backward_bias(float alpha,
                                        const memory_desc_ext &diff_dst_desc,
                                        void *diff_dst, float beta,
                                        const memory_desc_ext &diff_bias_desc,
                                        void *diff_bias);

  /// Getting the required weight space size for specified rnn operation.  
  /// \param [in] desc RNN descriptor.
  /// \param [out] weight_space_size Size of required weight space.
  void rnn_get_weight_space_size(const rnn_desc &desc,
                                 size_t *weight_space_size);

  /// Getting the required scratchpad size and workspace size for specified rnn operation.  
  /// \param [in] desc RNN descriptor.
  /// \param [in] kind Propagation kind.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [out] scratchpad_size Size of required scratchpad.
  /// \param [out] workspace_size Size of required workspace.
  void rnn_get_scratchpad_workspace_size(const rnn_desc &desc, ::dnnl::prop_kind kind,
                              const memory_desc_ext &src_desc,
                              size_t *scratchpad_size, size_t *workspace_size);

  /// Computing a specified rnn function value asynchronously.
  /// \param [in] desc RNN descriptor.
  /// \param [in] kind Propagation kind.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] iter_desc Recurrent hidden state data memory descriptor.
  /// \param [in] src_iter Pointer to input recurrent hidden state data.
  /// \param [in] dst_iter Pointer to output recurrent hidden state data.
  /// \param [in] iter_c_desc Recurrent cell state data memory descriptor.
  /// \param [in] src_c_iter Pointer to input recurrent cell state data.
  /// \param [in] dst_c_iter Pointer to output recurrent cell state data.
  /// \param [in] weight_size Size of weight memory.
  /// \param [in] weight Pointer to weight data.
  /// \param [in] scratchpad_size Size of scratchpad memory.
  /// \param [in] scratchpad Pointer to scratchpad data.
  /// \param [in] workspace_size Size of workspace memory.
  /// \param [in] workspace Pointer to workspace data.
  /// \returns An event representing the status of rnn forward operations.
  sycl::event async_rnn_forward(const rnn_desc &desc, ::dnnl::prop_kind kind,
                               const memory_desc_ext &src_desc, void *src,
                               const memory_desc_ext &dst_desc, void *dst,
                               const memory_desc_ext &iter_desc, void *src_iter,
                               void *dst_iter,
                               const memory_desc_ext &iter_c_desc,
                               void *src_iter_c, void *dst_iter_c,
                               size_t weight_size, void *weight,
                               size_t scratchpad_size, void *scratchpad,
                               size_t workspace_size, void *workspace);

  /// Computing the data and weight gradient of a specified rnn function
  /// asynchronously.
  /// \param [in] desc RNN descriptor.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [in] dst Pointer to destination data.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] iter_desc Recurrent hidden state data memory descriptor.
  /// \param [in] src_iter Pointer to input recurrent hidden state data.
  /// \param [in] diff_dst_iter Pointer to differential output recurrent hidden state data.
  /// \param [out] diff_src_iter Pointer to differential input recurrent hidden state data.
  /// \param [in] iter_c_desc Recurrent cell state data memory descriptor.
  /// \param [in] src_c_iter Pointer to input recurrent cell state data.
  /// \param [in] diff_dst_c_iter Pointer to differential output recurrent cell state data.
  /// \param [out] diff_src_c_iter Pointer to differential input recurrent cell state data.
  /// \param [in] weight_size Size of weight memory.
  /// \param [in] weight Pointer to weight data.
  /// \param [out] diff_weight Pointer to differential weight data.
  /// \param [in] scratchpad_size Size of scratchpad memory.
  /// \param [in] scratchpad Pointer to scratchpad data.
  /// \param [in] workspace_size Size of workspace memory.
  /// \param [in] workspace Pointer to workspace data.
  /// \returns An event representing the status of rnn backward operations.
  sycl::event async_rnn_backward(
      const rnn_desc &desc, const memory_desc_ext &dst_desc, void *dst,
      void *diff_dst, const memory_desc_ext &src_desc, void *src,
      void *diff_src, const memory_desc_ext &iter_desc, void *src_iter,
      void *diff_dst_iter, void *diff_src_iter,
      const memory_desc_ext &iter_c_desc, void *src_iter_c,
      void *diff_dst_iter_c, void *diff_src_iter_c, size_t weight_size,
      void *weight, void *diff_weight, size_t scratchpad_size, void *scratchpad,
      size_t workspace_size, void *workspace);

  /// Getting the required state size for specified dropout operation.
  /// \param [in] src_desc Source memory descriptor.
  /// \returns Required size of state.
  size_t get_dropout_state_size();

  /// Getting the required workspace size for dropout operation.
  /// \param [in] src_desc Source memory descriptor.
  /// \returns Required size of workspace.
  static size_t get_dropout_workspace_size(const memory_desc_ext &src_desc);

  /// Computing a specified dropout function value asynchronously.
  /// \param [in] desc Dropout descriptor.
  /// \param [in] src_desc Source memory descriptor.
  /// \param [in] src Pointer to source data.
  /// \param [in] dst_desc Destination memory descriptor.
  /// \param [out] dst Pointer to destination data.
  /// \param [in] workspace Pointer to workspace data.
  /// \param [in] workspace_size Size of workspace memory.
  /// \returns An event representing the dropout forward operations.
  sycl::event async_dropout_forward(dropout_desc &desc,
                                    const memory_desc_ext &src_desc, void *src,
                                    const memory_desc_ext &dst_desc, void *dst,
                                    void *workspace, size_t workspace_size);

  /// Computing the gradient of a specified dropout function asynchronously.
  /// \param [in] desc Dropout descriptor.
  /// \param [in] diff_dst_desc Differential destination memory descriptor.
  /// \param [in] diff_dst Pointer to differential destination data.
  /// \param [in] diff_src_desc Differential source memory descriptor.
  /// \param [out] diff_src Pointer to differential source data.
  /// \param [in] workspace Pointer to workspace data.
  /// \param [in] workspace_size Size of workspace memory.
  /// \returns An event representing the dropout backward operations.
  sycl::event async_dropout_backward(dropout_desc &desc,
                                     const memory_desc_ext &diff_dst_desc,
                                     void *diff_dst,
                                     const memory_desc_ext &diff_src_desc,
                                     void *diff_src, void *workspace,
                                     size_t workspace_size);
};

inline thread_local unsigned int engine_ext::_engine_count;
inline thread_local detail::primitive_cache engine_ext::_primitive_cache;
inline thread_local std::map<void *, ::dnnl::memory> engine_ext::_workspace_map;
inline thread_local std::map<sycl::queue *,
                             std::shared_ptr<engine_ext::internal_resource>>
    engine_ext::_internal_resource_cache;

inline
void dropout_desc::restore(engine_ext &engine, float p, void *state,
                                  size_t state_size, unsigned long long seed) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
  if (state) {
    std::int64_t required_state_size = engine.get_dropout_state_size();
    if (state_size < required_state_size) {
      throw std::runtime_error("restore: state_size less than required state size.");
    }
    sycl::queue *q = engine.get_queue();
    _imp->_p = p;
    _imp->_seed = seed;
    _imp->_state = state;
    _imp->_host_state = std::vector<std::uint8_t>(required_state_size);
    q->memcpy(_imp->_host_state.data(), _imp->_state, required_state_size).wait();
    _imp->_rng_engine =
        oneapi::mkl::rng::load_state<rng_engine_t>(
            *q, _imp->_host_state.data());
  }
#endif
}

inline
void dropout_desc::set(engine_ext &engine, float p, void *state,
                              size_t state_size, unsigned long long seed) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
  _imp->_p = p;
  if (state) {
    std::int64_t required_state_size = engine.get_dropout_state_size();
    if (state_size < required_state_size) {
      throw std::runtime_error("set: no sufficient memory to save states.");
    }
    sycl::queue *q = engine.get_queue();
    _imp->_seed = seed;
    _imp->_state = state;
    _imp->_host_state = std::vector<std::uint8_t>(required_state_size);
    _imp->_rng_engine = rng_engine_t(*q, seed);
    oneapi::mkl::rng::save_state(_imp->_rng_engine, _imp->_host_state.data());
    q->memcpy(_imp->_state, _imp->_host_state.data(), required_state_size).wait();
  }
#endif
}

inline
::dnnl::memory::data_type
memory_desc_ext::to_dnnl_data_type(dpct::library_data_t dt) {
  using dnnl_dt = ::dnnl::memory::data_type;
  switch (dt) {
  case dpct::library_data_t::real_half:
    return dnnl_dt::f16;
  case dpct::library_data_t::real_bfloat16:
    return dnnl_dt::bf16;
  case dpct::library_data_t::real_float:
    return dnnl_dt::f32;
  case dpct::library_data_t::real_double:
    return dnnl_dt::f64;
  case dpct::library_data_t::real_int32:
    return dnnl_dt::s32;
  case dpct::library_data_t::real_int8:
    return dnnl_dt::s8;
  case dpct::library_data_t::real_uint8:
    return dnnl_dt::u8;
  case dpct::library_data_t::real_int8_4:
    return dnnl_dt::s8;
  case dpct::library_data_t::real_int8_32:
    return dnnl_dt::s8;
  case dpct::library_data_t::real_uint8_4:
    return dnnl_dt::u8;
  default:
    throw std::runtime_error("to_dnnl_data_type: unsupported data type.");
  }
}

inline
dpct::library_data_t
memory_desc_ext::to_dpct_library_data_t(::dnnl::memory::data_type dt,
                                        unsigned block_size) {
  using dpct_dt = dpct::library_data_t;
  using dnnl_dt = ::dnnl::memory::data_type;
  switch (dt) {
  case dnnl_dt::f16:
    return dpct_dt::real_half;
  case dnnl_dt::bf16:
    return dpct_dt::real_bfloat16;
  case dnnl_dt::f32:
    return dpct_dt::real_float;
  case dnnl_dt::f64:
    return dpct_dt::real_double;
  case dnnl_dt::s32:
    return dpct_dt::real_int32;
  case dnnl_dt::s8:
    if (block_size == 4) {
      return dpct_dt::real_int8_4;
    } else if (block_size == 32) {
      return dpct_dt::real_int8_32;
    } else {
      return dpct_dt::real_int8;
    }
  case dnnl_dt::u8:
    if (block_size == 4) {
      return dpct_dt::real_uint8_4;
    } else {
      return dpct_dt::real_uint8;
    }
  default:
    throw std::runtime_error("to_dpct_library_data_t: unsupported data type "
                             "dnnl::memory::data_type::undef.");
  }
}

inline
::dnnl::memory::format_tag
memory_desc_ext::to_dnnl_format_tag(dpct::library_data_t dt,
                                    memory_format_tag tag) {
  using dpct_dt = dpct::library_data_t;
  using dpct_tag = memory_format_tag;
  using dnnl_tag = ::dnnl::memory::format_tag;
  switch (tag) {
  case dpct_tag::nchw:
    return dnnl_tag::nchw;
  case dpct_tag::nhwc:
    return dnnl_tag::nhwc;
  default:
    if (dt == dpct_dt::real_int8_32) {
      return dnnl_tag::nChw32c;
    } else {
      return dnnl_tag::nChw4c;
    }
  }
}

inline
void memory_desc_ext::set(memory_format_tag tag, dpct::library_data_t dt, int n,
                          int c, int h, int w) {
  _desc = ::dnnl::memory::desc({n, c, h, w}, to_dnnl_data_type(dt),
                               to_dnnl_format_tag(dt, tag));
}

inline
void memory_desc_ext::set(dpct::library_data_t dt, int n, int c, int h, int w,
                          int n_stride, int c_stride, int h_stride,
                          int w_stride) {
  _desc = ::dnnl::memory::desc({n, c, h, w}, to_dnnl_data_type(dt),
                               {n_stride, c_stride, h_stride, w_stride});
}

inline
void memory_desc_ext::set(dpct::library_data_t dt, int ndims, const int dims[],
                          const int strides[]) {
  _desc = ::dnnl::memory::desc({dims, dims + ndims}, to_dnnl_data_type(dt),
                               {strides, strides + ndims});
}

inline
void memory_desc_ext::set(memory_format_tag tag, dpct::library_data_t dt,
                          int ndims, const int dims[]) {
  _desc = ::dnnl::memory::desc({dims, dims + ndims}, to_dnnl_data_type(dt),
                               to_dnnl_format_tag(dt, tag));
}

inline
void memory_desc_ext::set(rnn_memory_format_tag tag, dpct::library_data_t dt,
                          int t, int n, int c) {
  if (tag == rnn_memory_format_tag::tnc) {
    _desc = ::dnnl::memory::desc({t, n, c}, to_dnnl_data_type(dt),
                                 ::dnnl::memory::format_tag::tnc);
  } else if(tag == rnn_memory_format_tag::ntc) {
    _desc = ::dnnl::memory::desc({t, n, c}, to_dnnl_data_type(dt),
                                 ::dnnl::memory::format_tag::ntc);
  } else {
    throw std::runtime_error("set: unsupported memory format tag.");
  }
}

inline
void memory_desc_ext::get(dpct::library_data_t *dt, int *n, int *c, int *h,
                          int *w, int *n_stride, int *c_stride, int *h_stride,
                          int *w_stride) const {
  unsigned block_size = 1;
  auto dims = _desc.get_dims();
  auto inner_blks = _desc.get_inner_blks();
  auto strides = _desc.get_strides();
  if (!inner_blks.empty()) {
    block_size = inner_blks[0];
  }

  *dt = to_dpct_library_data_t(_desc.get_data_type(), block_size);
  *n = dims[0];
  *c = dims[1];
  *h = dims[2];
  *w = dims[3];
  *n_stride = strides[0] / block_size;
  *c_stride = strides[1] / block_size;
  *h_stride = strides[2] / block_size;
  *w_stride = strides[3] / block_size;
}

inline
void memory_desc_ext::get(dpct::library_data_t *dt, memory_format_tag *tag,
                          int *n, int *c, int *h, int *w) const {
  unsigned block_size = 1;
  *tag = memory_format_tag::nchw;
  auto dims = _desc.get_dims();
  auto strides = _desc.get_strides();
  auto inner_blks = _desc.get_inner_blks();
  if (!inner_blks.empty()) {
    block_size = inner_blks[0];
    *tag = memory_format_tag::nchw_blocked;
  }
  if (strides[1] == 1 && dims[1] != 1) {
    *tag = memory_format_tag::nhwc;
  }
  *dt = to_dpct_library_data_t(_desc.get_data_type(), block_size);
  *n = dims[0];
  *c = dims[1];
  *h = dims[2];
  *w = dims[3];
}

inline
void memory_desc_ext::get(dpct::library_data_t *dt, rnn_memory_format_tag *tag,
                          int *t, int *n, int *c) const {
  auto dims = _desc.get_dims();
  auto strides = _desc.get_strides();

  if (strides[0] >= strides[1]) {
    *tag = rnn_memory_format_tag::tnc;
  } else {
    *tag = rnn_memory_format_tag::ntc;
  }

  *dt = to_dpct_library_data_t(_desc.get_data_type(), 1);
  *t = dims[0];
  *n = dims[1];
  *c = dims[2];
}

inline
void memory_desc_ext::get(int requested_ndims, dpct::library_data_t *dt,
                          int *ndims, int dims[], int strides[]) const {
  unsigned block_size = 1;
  auto inner_blks = _desc.get_inner_blks();
  auto adims = _desc.get_dims();
  auto astrides = _desc.get_strides();
  if (!inner_blks.empty()) {
    block_size = inner_blks[0];
  }
  *dt = to_dpct_library_data_t(_desc.get_data_type(), block_size);
  *ndims = _desc.get_ndims();
  for (int index = 0; index < requested_ndims; index++) {
    dims[index] = adims[index];
    strides[index] =
        astrides[index] / block_size;
  }
}

inline
void memory_desc_ext::get(int requested_ndims, dpct::library_data_t *dt,
                          memory_format_tag *tag, int *ndims,
                          int dims[]) const {
  unsigned block_size = 1;
  *tag = memory_format_tag::nchw;
  auto inner_blks = _desc.get_inner_blks();
  auto adims = _desc.get_dims();
  auto astrides = _desc.get_strides();
  if (!inner_blks.empty()) {
    block_size = inner_blks[0];
    *tag = memory_format_tag::nchw_blocked;
  }
  if (astrides[1] == 1 &&
      adims[1] != 1) {
    *tag = memory_format_tag::nhwc;
  }
  *dt = to_dpct_library_data_t(_desc.get_data_type(), block_size);
  *ndims = _desc.get_ndims();
  for (int index = 0; index < requested_ndims; index++) {
    dims[index] = adims[index];
  }
}

inline
void engine_ext::get_rnn_configuration(const ::dnnl::memory::desc &desc,
                                       rnn_direction direction, rnn_mode mode,
                                       dpct::library_data_t dt, int hidden_size,
                                       ::dnnl::memory::data_type *dnnl_dt,
                                       ::dnnl::memory::format_tag *tag,
                                       int *projection_size, int *output_size,
                                       int *seq_length, int *batch_size,
                                       int *direction_num, int *gate_num) {
  if (!desc.is_zero()) {
    auto dims = desc.get_dims();
    auto strides = desc.get_strides();
    if (strides[0] >= strides[1]) {
      *tag = ::dnnl::memory::format_tag::tnc;
      *seq_length = dims[0];
      *batch_size = dims[1];
    } else {
      *tag = ::dnnl::memory::format_tag::ntc;
      *seq_length = dims[1];
      *batch_size = dims[0];
    }
  }
  if (direction == rnn_direction::bidirectional) {
    *direction_num = 2;
  } else {
    *direction_num = 1;
  }
  if (mode == rnn_mode::lstm) {
    *gate_num = 4;
  } else if (mode == rnn_mode::gru) {
    *gate_num = 3;
  } else {
    *gate_num = 1;
  }
  if (*projection_size != hidden_size) {
    *output_size = *projection_size;
  } else {
    *projection_size = 0;
    *output_size = hidden_size;
  }
  *dnnl_dt = memory_desc_ext::to_dnnl_data_type(dt);
}

inline
void *engine_ext::allocate(const memory_desc_ext &data_desc, int count) {
  return allocate(data_desc.get_size() * count);
}

inline
void *engine_ext::allocate(size_t size) {
  auto &Info = get_internal_resource(_q)->binfo;
  uint8_t *result = Info.buffer + Info.usage;
  Info.usage += size;
  return result;
}

inline
void engine_ext::transform_no_zero(const memory_desc_ext &desc, void *src, void *dst) {
  ::dnnl::memory::data_type dt = desc.get_desc().get_data_type();
  size_t element_num = desc.get_element_num();
  switch (dt) {
  case ::dnnl::memory::data_type::f32:
    transform_no_zero_with_type<float>(_q, src, dst, element_num);
    break;
  case ::dnnl::memory::data_type::f16:
    transform_no_zero_with_type<sycl::half>(_q, src, dst, element_num);
    break;
  case ::dnnl::memory::data_type::s32:
    transform_no_zero_with_type<int32_t>(_q, src, dst, element_num);
    break;
  case ::dnnl::memory::data_type::s8:
    transform_no_zero_with_type<int8_t>(_q, src, dst, element_num);
    break;
  case ::dnnl::memory::data_type::u8:
    transform_no_zero_with_type<uint8_t>(_q, src, dst, element_num);
    break;
  default:
    throw std::runtime_error("transform_no_zero: unsupported data type.");
  }
}

inline
::dnnl::memory::desc
engine_ext::get_group_weight_desc(int group_count,
                                  const memory_desc_ext &weight_desc) {
  if (group_count == 1) {
    return weight_desc.get_desc();
  }
  auto help_weight_desc = weight_desc.get_desc();
  int ndims = help_weight_desc.get_ndims();
  if (!help_weight_desc.get_inner_blks().empty()) {
    throw std::runtime_error("get_group_weight_desc: group convolution with "
                             "blocked weight memory unimplemented.");
  }
  std::vector<int64_t> new_size;
  auto old_size = weight_desc.get_dims();
  new_size.push_back(group_count);
  new_size.push_back(old_size[0] / group_count);
  for (int index = 1; index < old_size.size(); index++) {
    new_size.push_back(old_size[index]);
  }
  std::vector<int64_t> strides = help_weight_desc.get_strides();
  ::dnnl::memory::format_tag tag;
  bool is_nhwc = (strides[1] == 1 && old_size[1] != 1);

  if (ndims == 4) {
    if (is_nhwc) {
      tag = ::dnnl::memory::format_tag::gohwi;
    } else {
      tag = ::dnnl::memory::format_tag::goihw;
    }
  } else if (ndims == 5) {
    if (is_nhwc) {
      tag = ::dnnl::memory::format_tag::godhwi;
    } else {
      tag = ::dnnl::memory::format_tag::goidhw;
    }
  }

  help_weight_desc =
      ::dnnl::memory::desc(new_size, weight_desc.get_desc().get_data_type(), tag);
  return help_weight_desc;
}

inline
::dnnl::memory::desc engine_ext::compress_spatial_dimensions_to_channel(
    const ::dnnl::memory::desc &desc) {
  int ndims = desc.get_ndims();
  auto dims = desc.get_dims();
  auto inner_blks = desc.get_inner_blks();
  assert(ndims >= 4 && "ndims is at least 4.");
  std::vector<int64_t> compressed_dims(ndims);
  compressed_dims[0] = dims[0];
  compressed_dims[1] = dims[1];
  for (int index = 2; index < ndims; index++) {
    compressed_dims[1] = compressed_dims[1] * dims[index];
    compressed_dims[index] = 1;
  }
  if (!inner_blks.empty() && inner_blks[0] == 4) {
    return ::dnnl::memory::desc(compressed_dims, desc.get_data_type(),
                                ::dnnl::memory::format_tag::nChw4c);
  } else if (!inner_blks.empty() && inner_blks[0] == 32) {
    return ::dnnl::memory::desc(compressed_dims, desc.get_data_type(),
                                ::dnnl::memory::format_tag::nChw32c);
  }
  std::vector<int64_t> strides(ndims, 1);
  strides[0] = compressed_dims[1];

  return ::dnnl::memory::desc(compressed_dims, desc.get_data_type(), strides);
}

inline
::dnnl::memory::desc
engine_ext::get_bn_scale_bias_mean_var_desc(const ::dnnl::memory::desc &desc,
                                            batch_normalization_mode mode) {
  int ndims = desc.get_ndims();
  auto dims = desc.get_dims();
  assert(ndims >= 4 && "ndims is at least 4.");
  int channel_num = 1;
  if (mode == batch_normalization_mode::spatial) {
    channel_num = dims[1];
  } else {
    for (int index = 1; index < ndims; index++) {
      channel_num = channel_num * dims[index];
    }
  }
  return ::dnnl::memory::desc({channel_num}, desc.get_data_type(),
                              ::dnnl::memory::format_tag::a);
}

inline
::dnnl::memory::desc engine_ext::transfer_memory_desc_to_channel_major_format(
    const ::dnnl::memory::desc &desc) {
  if (!desc.get_inner_blks().empty()) {
    return desc;
  }
  int ndims = desc.get_ndims();
  auto dims = desc.get_dims();
  if (ndims == 4) {
    return ::dnnl::memory::desc(dims, desc.get_data_type(),
                                ::dnnl::memory::format_tag::nchw);
  }
  return ::dnnl::memory::desc(dims, desc.get_data_type(),
                              ::dnnl::memory::format_tag::ncdhw);
}

/// If the alpha = 0 and beta = 1, then the destination (dst = alpha * out +
/// beta * prior_dst) have no change. In this case this function returns true
/// means the operation can exit directly.
inline
bool engine_ext::scale_parameter_preprocess(
    const std::vector<output_argument_info> &args) {
  bool direct_exit = true;
  for (auto &arg : args) {
    if (arg._alpha == 0.f) {
      if (arg._beta != 1.f) {
        async_scale(arg._beta, arg._desc, arg._data);
      }
    } else {
      direct_exit = false;
    }
  }
  return direct_exit;
}

inline
void engine_ext::derive_batch_normalization_memory_desc(
    memory_desc_ext &scale_bias_desc, memory_desc_ext &mean_var_desc,
    const memory_desc_ext &src_desc, batch_normalization_mode mode) {
    derive_batch_normalization_memory_desc(scale_bias_desc, src_desc, mode);
    derive_batch_normalization_memory_desc(mean_var_desc, src_desc, mode);
}

inline
void engine_ext::derive_batch_normalization_memory_desc(
    memory_desc_ext &desc, const memory_desc_ext &src_desc,
    batch_normalization_mode mode) {
  int src_ndims = src_desc.get_desc().get_ndims();
  auto inner_blks = src_desc.get_desc().get_inner_blks();
  if (src_desc.get_desc().get_ndims() != 4 ||
      src_desc.get_desc().get_ndims() != 5) {
    throw std::runtime_error("derive_batch_normalization_memory_desc: only 4d "
                             "and 5d memory descriptor supported.");
  }
  std::vector<int64_t> dims = src_desc.get_dims();
  dims[0] = 1;
  if (mode == batch_normalization_mode::spatial) {
    dims[2] = 1;
    dims[3] = 1;
    if (src_ndims == 5) {
      dims[4] = 1;
    }
  }
  auto data_type = src_desc.get_desc().get_data_type();
  if (data_type == ::dnnl::memory::data_type::f16) {
    data_type = ::dnnl::memory::data_type::f32;
  }
  if (!inner_blks.empty() && inner_blks[0] == 4) {
    desc.set_desc(::dnnl::memory::desc(dims, data_type,
                                       ::dnnl::memory::format_tag::nChw4c));
  } else if (!inner_blks.empty() && inner_blks[0] == 32) {
    desc.set_desc(::dnnl::memory::desc(dims, data_type,
                                       ::dnnl::memory::format_tag::nChw32c));
  } else {
    if (src_ndims == 4) {
      desc.set_desc(::dnnl::memory::desc(dims, data_type,
                                         ::dnnl::memory::format_tag::nchw));
    } else {
      desc.set_desc(::dnnl::memory::desc(dims, data_type,
                                         ::dnnl::memory::format_tag::ncdhw));
    }
  }
}

template <typename primitive_type>
sycl::event engine_ext::execute_primitive(
    const std::pair<detail::primitive_cache_key_type, detail::primitive_and_args>
        &primitive,
    const std::vector<output_argument_info> &output_args) {
  std::vector<void *> caches;
  int output_arg_num = output_args.size();
  for (int i = 0; i < output_arg_num; i++) {
    if (output_args[i]._beta != 0.f) {
      auto cache = allocate(output_args[i]._desc);
      caches.push_back(cache);
      (*primitive.second.args)[output_args[i]._name].set_data_handle(cache);
    }
  }

  auto e = ::dnnl::sycl_interop::execute(
      *(static_cast<primitive_type *>(primitive.second.primitive)), *_s,
      *primitive.second.args);
  _primitive_cache.put(
      primitive.first, primitive.second.primitive, primitive.second.args,
      [](::dnnl::primitive *p) { delete static_cast<primitive_type *>(p); }, e,
      _q);
  int cache_index = 0;
  for (int i = 0; i < output_arg_num; i++) {
    if (output_args[i]._beta != 0.f) {
      e = async_sum(output_args[i]._alpha, output_args[i]._desc,
                    caches[cache_index++], output_args[i]._beta,
                    output_args[i]._desc, output_args[i]._data);
    } else {
      if (output_args[i]._alpha != 1.f) {
        e = async_scale(output_args[i]._alpha, output_args[i]._desc,
                        output_args[i]._data);
      }
    }
  }
  return e;
}

inline
::dnnl::memory::desc engine_ext::bn_reorder_memory_to_channel_major_format(
    bool is_input, ::dnnl::memory::desc &desc, void *src, void **cache) {
  ::dnnl::memory::desc result;
  result = transfer_memory_desc_to_channel_major_format(desc);
  if ((result != desc) || !src) {
    *cache = allocate(desc);
    if (is_input && src) {
      async_reorder(1.f, desc, src, 0.f, result, *cache);
    }
  }
  return result;
}

inline
sycl::event engine_ext::batch_normalization_backward_internal(
    batch_normalization_mode mode, float epsilon, float alpha_data,
    const memory_desc_ext &src_desc, void *src,
    const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta_data,
    const memory_desc_ext &diff_src_desc, void *diff_src, float alpha_param,
    const memory_desc_ext &diff_scale_bias_desc, void *scale, void *bias,
    float beta_param, void *diff_scale, void *diff_bias,
    const memory_desc_ext &mean_var_desc, void *saved_mean, void *saved_var) {
  if (scale_parameter_preprocess(
          {{alpha_data, beta_data, diff_src_desc, diff_src},
           {alpha_param, beta_param, diff_scale_bias_desc, diff_scale},
           {alpha_param, beta_param, diff_scale_bias_desc, diff_bias}})) {
    return sycl::event();
  }

  void *reordered_src = nullptr, *reordered_diff_dst = nullptr,
       *reordered_diff_src = nullptr, *reordered_scale = nullptr,
       *reordered_bias = nullptr, *reordered_diff_scale = nullptr,
       *reordered_diff_bias = nullptr, *reordered_saved_mean = nullptr,
       *reordered_saved_var = nullptr;

  ::dnnl::memory::desc help_src_desc = src_desc.get_desc();
  ::dnnl::memory::desc help_diff_dst_desc = diff_dst_desc.get_desc();
  ::dnnl::memory::desc help_diff_src_desc = diff_src_desc.get_desc();
  ::dnnl::memory::desc help_diff_scale_bias_desc =
      diff_scale_bias_desc.get_desc();
  ::dnnl::memory::desc help_mean_var_desc = mean_var_desc.get_desc();
  ::dnnl::memory::desc actual_diff_src_desc = help_diff_src_desc;
  ::dnnl::memory::desc actual_diff_scale_bias_desc = help_diff_scale_bias_desc;
  enter_primitive(
      help_diff_scale_bias_desc.get_size() * 14 + help_src_desc.get_size() * 2 +
      help_diff_dst_desc.get_size() * 7 + help_diff_src_desc.get_size() * 5 +
      help_mean_var_desc.get_size() * 13);
  if (mode == batch_normalization_mode::per_activation) {
    help_src_desc = bn_reorder_memory_to_channel_major_format(true, help_src_desc, src,
                                                       &reordered_src);
    help_diff_dst_desc = bn_reorder_memory_to_channel_major_format(
        true, help_diff_dst_desc, diff_dst, &reordered_diff_dst);
    help_diff_src_desc = bn_reorder_memory_to_channel_major_format(
        false, help_diff_src_desc, diff_src, &reordered_diff_src);
    actual_diff_src_desc = help_diff_src_desc;
    help_diff_scale_bias_desc = bn_reorder_memory_to_channel_major_format(
        true, help_diff_scale_bias_desc, scale, &reordered_scale);
    actual_diff_scale_bias_desc = help_diff_scale_bias_desc;
    if (bias) {
      bn_reorder_memory_to_channel_major_format(true, help_diff_scale_bias_desc, bias,
                                         &reordered_bias);
    }
    bn_reorder_memory_to_channel_major_format(false, help_diff_scale_bias_desc,
                                       diff_scale, &reordered_diff_scale);
    bn_reorder_memory_to_channel_major_format(false, help_diff_scale_bias_desc,
                                       diff_bias, &reordered_diff_bias);

    help_mean_var_desc = bn_reorder_memory_to_channel_major_format(
        true, help_mean_var_desc, saved_mean, &reordered_saved_mean);
    bn_reorder_memory_to_channel_major_format(true, help_mean_var_desc, saved_var,
                                       &reordered_saved_var);
    help_src_desc = compress_spatial_dimensions_to_channel(help_src_desc);
    help_diff_src_desc =
        compress_spatial_dimensions_to_channel(help_diff_src_desc);
    help_diff_dst_desc =
        compress_spatial_dimensions_to_channel(help_diff_dst_desc);
  } else {
    if ((help_src_desc != help_diff_dst_desc) ||
        (help_src_desc != help_diff_src_desc) ||
        (help_diff_dst_desc != help_diff_src_desc)) {
      help_src_desc = bn_reorder_memory_to_channel_major_format(
          true, help_src_desc, src, &reordered_src);
      help_diff_dst_desc = bn_reorder_memory_to_channel_major_format(
          true, help_diff_dst_desc, diff_dst, &reordered_diff_dst);
      help_diff_src_desc = bn_reorder_memory_to_channel_major_format(
          false, help_diff_src_desc, diff_src, &reordered_diff_src);
      actual_diff_src_desc = help_diff_src_desc;
    }
  }

  help_diff_scale_bias_desc =
      get_bn_scale_bias_mean_var_desc(help_diff_scale_bias_desc, mode);
  help_mean_var_desc =
      get_bn_scale_bias_mean_var_desc(help_mean_var_desc, mode);

  auto forward_primitive =
      create_primitive_desc<::dnnl::batch_normalization_forward>(
          ::dnnl::prop_kind::forward_training, help_src_desc,
          help_diff_dst_desc, epsilon,
          ::dnnl::normalization_flags::use_scale |
              ::dnnl::normalization_flags::use_shift);
  auto primitive_args =
      create_primitive_args_or_get<::dnnl::batch_normalization_backward>(
          ::dnnl::prop_kind::backward, help_diff_src_desc, help_diff_dst_desc,
          help_src_desc, epsilon,
          ::dnnl::normalization_flags::use_scale |
              ::dnnl::normalization_flags::use_shift, forward_primitive);

  void *dst_cache = nullptr;
  if (!saved_mean && !saved_var) {
    dst_cache = allocate(diff_dst_desc);
    if (!reordered_saved_mean) {
      reordered_saved_mean = allocate(mean_var_desc);
    }
    if (!reordered_saved_var) {
      reordered_saved_var = allocate(mean_var_desc);
    }
    if (!bias) {
      _q->fill(reordered_bias, 0, diff_scale_bias_desc.get_size());
    }

    batch_normalization_forward_internal(
        true, mode, epsilon, 0.f, 1.f, src_desc, src, 0.f, diff_dst_desc,
        dst_cache, diff_scale_bias_desc, scale, bias ? bias : reordered_bias,
        mean_var_desc, reordered_saved_mean, reordered_saved_var, nullptr,
        nullptr);
  }

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, help_src_desc,
             reordered_src ? reordered_src : src);
  insert_arg(primitive_args.second.args, DNNL_ARG_SCALE,
             help_diff_scale_bias_desc,
             reordered_scale ? reordered_scale : scale);
  insert_arg(primitive_args.second.args, DNNL_ARG_MEAN, help_mean_var_desc,
             reordered_saved_mean ? reordered_saved_mean : saved_mean);
  insert_arg(primitive_args.second.args, DNNL_ARG_VARIANCE, help_mean_var_desc,
             reordered_saved_var ? reordered_saved_var : saved_var);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST, help_diff_src_desc,
             reordered_diff_dst ? reordered_diff_dst : diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SRC, help_diff_src_desc,
             reordered_diff_src ? reordered_diff_src : diff_src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SCALE,
             help_diff_scale_bias_desc,
             reordered_diff_scale ? reordered_diff_scale : diff_scale);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SHIFT,
             help_diff_scale_bias_desc,
             reordered_diff_bias ? reordered_diff_bias : diff_bias);

  sycl::event e = execute_primitive<::dnnl::batch_normalization_backward>(
      primitive_args,
      {{alpha_data, beta_data, DNNL_ARG_DIFF_SRC, help_diff_src_desc,
        reordered_diff_src ? reordered_diff_src : diff_src},
       {alpha_param, beta_param, DNNL_ARG_DIFF_SCALE, help_diff_scale_bias_desc,
        reordered_diff_scale ? reordered_diff_scale : diff_scale},
       {alpha_param, beta_param, DNNL_ARG_DIFF_SHIFT, help_diff_scale_bias_desc,
        reordered_diff_bias ? reordered_diff_bias : diff_bias}});
  if (actual_diff_src_desc != diff_src_desc.get_desc() && reordered_diff_src) {
    e = async_reorder(1.f, actual_diff_src_desc, reordered_diff_src, 0.f,
                diff_src_desc, diff_src);
  }
  if (actual_diff_scale_bias_desc != diff_scale_bias_desc.get_desc() &&
      reordered_diff_scale && reordered_diff_bias) {
    async_reorder(1.f, actual_diff_scale_bias_desc, reordered_diff_scale, 0.f,
            diff_scale_bias_desc, diff_scale);
    e = async_reorder(1.f, actual_diff_scale_bias_desc, reordered_diff_bias, 0.f,
                diff_scale_bias_desc, diff_bias);
  }
  return exit_primitive(e);
}

inline
sycl::event engine_ext::batch_normalization_forward_internal(
    bool is_infer, batch_normalization_mode mode, float epsilon, float factor,
    float alpha, const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &scale_bias_desc, void *scale, void *bias,
    const memory_desc_ext &mean_var_desc, void *saved_mean, void *saved_var,
    void *running_mean, void *running_var) {
  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }
  enter_primitive(src_desc.get_size() + 5 * dst_desc.get_size() +
                  scale_bias_desc.get_size() * 2 +
                  mean_var_desc.get_size() * 9);
  void *reordered_src = nullptr, *reordered_dst = nullptr,
       *reordered_scale = nullptr, *reordered_bias = nullptr,
       *reordered_saved_mean = nullptr, *reordered_saved_var = nullptr;
  ::dnnl::memory::desc help_src_desc = src_desc.get_desc();
  ::dnnl::memory::desc help_dst_desc = dst_desc.get_desc();
  ::dnnl::memory::desc help_scale_bias_desc = scale_bias_desc.get_desc();
  ::dnnl::memory::desc help_mean_var_desc = mean_var_desc.get_desc();
  ::dnnl::memory::desc actual_dst_desc = help_dst_desc;
  ::dnnl::memory::desc actual_mean_var_desc = help_mean_var_desc;

  if (mode == batch_normalization_mode::per_activation) {
    help_src_desc = bn_reorder_memory_to_channel_major_format(true, help_src_desc, src,
                                                       &reordered_src);
    help_dst_desc = bn_reorder_memory_to_channel_major_format(
        false, help_dst_desc, dst, &reordered_dst);
    actual_dst_desc = help_dst_desc;
    help_scale_bias_desc = bn_reorder_memory_to_channel_major_format(
        true, help_scale_bias_desc, scale, &reordered_scale);
    bn_reorder_memory_to_channel_major_format(true, help_scale_bias_desc, bias,
                                       &reordered_bias);
    help_mean_var_desc = bn_reorder_memory_to_channel_major_format(
        is_infer, help_mean_var_desc, saved_mean,
        &reordered_saved_mean);
    actual_mean_var_desc = help_mean_var_desc;
    bn_reorder_memory_to_channel_major_format(is_infer,
                                       help_mean_var_desc, saved_var,
                                       &reordered_saved_var);
    help_src_desc = compress_spatial_dimensions_to_channel(help_src_desc);
    help_dst_desc = compress_spatial_dimensions_to_channel(help_dst_desc);
  } else {
    if (help_src_desc != help_dst_desc) {
      help_src_desc = bn_reorder_memory_to_channel_major_format(
          true, help_src_desc, src, &reordered_src);
      help_dst_desc = bn_reorder_memory_to_channel_major_format(
          false, help_dst_desc, dst, &reordered_dst);
      actual_dst_desc = help_dst_desc;
    }
  }
  help_scale_bias_desc =
      get_bn_scale_bias_mean_var_desc(help_scale_bias_desc, mode);
  help_mean_var_desc =
      get_bn_scale_bias_mean_var_desc(help_mean_var_desc, mode);

  ::dnnl::prop_kind kind;
  ::dnnl::normalization_flags flag = ::dnnl::normalization_flags::use_scale |
                                     ::dnnl::normalization_flags::use_shift;
  if (is_infer) {
    kind = ::dnnl::prop_kind::forward_inference;
    flag = ::dnnl::normalization_flags::use_global_stats | flag;
  } else {
    kind = ::dnnl::prop_kind::forward_training;
  }
  auto primitive_args =
      create_primitive_args_or_get<::dnnl::batch_normalization_forward>(
          kind, help_src_desc, help_dst_desc, epsilon, flag);

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, help_src_desc,
             reordered_src ? reordered_src : src);
  insert_arg(primitive_args.second.args, DNNL_ARG_SCALE, help_scale_bias_desc,
             reordered_scale ? reordered_scale : scale);
  insert_arg(primitive_args.second.args, DNNL_ARG_SHIFT, help_scale_bias_desc,
             reordered_bias ? reordered_bias : bias);
  insert_arg(primitive_args.second.args, DNNL_ARG_MEAN, help_mean_var_desc,
             reordered_saved_mean ? reordered_saved_mean
                                            : saved_mean);
  insert_arg(primitive_args.second.args, DNNL_ARG_VARIANCE, help_mean_var_desc,
             reordered_saved_var ? reordered_saved_var
                                           : saved_var);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, help_dst_desc,
             reordered_dst ? reordered_dst : dst);
  sycl::event e = execute_primitive<::dnnl::batch_normalization_forward>(primitive_args,
                                    {{alpha, beta, DNNL_ARG_DST, help_dst_desc,
                                      reordered_dst ? reordered_dst : dst}});

  if (!is_infer && running_var) {
    auto src_ndim = src_desc.get_desc().get_ndims();
    auto src_dims = src_desc.get_dims();
    int element_num = src_dims[0];
    if (mode == batch_normalization_mode::spatial) {
      for (int index = 2; index < src_ndim; index++) {
        element_num *= src_dims[index];
      }
    }
    float unbias_factor = element_num / (element_num - 1.f);
    async_scale(1.f - factor, mean_var_desc, running_var);
    e = async_sum(factor * unbias_factor, mean_var_desc,
            reordered_saved_var ? reordered_saved_var : saved_var,
            1.f, mean_var_desc, running_var);
  }
  if (!is_infer && running_mean) {
    e = async_sum(factor, mean_var_desc,
            reordered_saved_mean ? reordered_saved_mean : saved_mean,
            (1.f - factor), mean_var_desc, running_mean);
  }
  if (reordered_dst && (actual_dst_desc != dst_desc.get_desc())) {
    e = async_reorder(1.f, actual_dst_desc, reordered_dst, 0.f, dst_desc, dst);
  }
  if (!is_infer && reordered_saved_mean && reordered_saved_var && saved_mean &&
      saved_var && (actual_mean_var_desc != mean_var_desc.get_desc())) {
    e = async_reorder(1.f, actual_mean_var_desc, reordered_saved_mean, 0.f,
                mean_var_desc, saved_mean);
    e = async_reorder(1.f, actual_mean_var_desc, reordered_saved_var, 0.f,
                mean_var_desc, saved_var);
  }
  return exit_primitive(e);
}

inline
sycl::event engine_ext::rnn_forward_internal(
    const rnn_desc &desc, ::dnnl::prop_kind kind,
    const memory_desc_ext &src_desc, void *src, const memory_desc_ext &dst_desc,
    void *dst, const memory_desc_ext &iter_desc, void *src_iter, void *dst_iter,
    const memory_desc_ext &iter_c_desc, void *src_iter_c, void *dst_iter_c,
    size_t weight_size, void *weight, size_t workspace_size, void *workspace,
    size_t scratchpad_size, void *scratchpad, bool is_get_execution_args,
    size_t *weight_size_query, size_t *workspace_size_query,
    size_t *scratchpad_size_query) {
  ::dnnl::memory::data_type src_dt;
  ::dnnl::memory::format_tag src_format_tag;
  rnn_mode mode;
  rnn_bias_mode bias_mode;
  rnn_direction direction;
  dpct::library_data_t dt;
  int direction_num = 1, input_size = 0, hidden_size = 0, projection_size = 0,
      layer_size = 0, gate_num = 1, output_size = 0, data_type_size = 0,
      seq_length = 1, batch_size = 1;
  std::vector<void *> data = {src,        dst,        src_iter, dst_iter,
                              src_iter_c, dst_iter_c, weight,   workspace,
                              scratchpad};
  std::vector<int> offset(6, 0);
  void *input_layer_cache = nullptr, *hidden_layer_cache = nullptr;
  sycl::event e;
  enter_primitive(src_desc.get_size() * 2);
  desc.get(&mode, &bias_mode, &direction, &dt, &input_size, &hidden_size,
           &projection_size, &layer_size);

  get_rnn_configuration(src_desc.get_desc(), direction, mode, dt, hidden_size,
                        &src_dt, &src_format_tag, &projection_size,
                        &output_size, &seq_length, &batch_size, &direction_num,
                        &gate_num);

  if (direction == rnn_direction::bidirectional) {
    // Here to combine the oneDNN bidirectional_sum and 
    // bidirectional_concat config, so call execute_rnn_forward_primitive
    // twice.
    if (layer_size > 1) {
      if (!is_get_execution_args) {
        input_layer_cache = allocate(src_desc);
        hidden_layer_cache = allocate(src_desc);
        _q->memcpy(input_layer_cache, src, src_desc.get_size());
      }
      data[0] = input_layer_cache;
      data[1] = hidden_layer_cache;
      e = execute_rnn_forward_primitive(
          mode, kind, ::dnnl::rnn_direction::bidirectional_sum, bias_mode,
          src_dt, src_format_tag, seq_length, batch_size, output_size,
          output_size, 1, direction_num, hidden_size, gate_num, projection_size,
          data, offset, layer_size - 1, weight_size_query, workspace_size_query,
          scratchpad_size_query);
      data[0] =
          ((layer_size - 1) % 2 == 0) ? input_layer_cache : hidden_layer_cache;
      data[1] = dst;
    }
    e = execute_rnn_forward_primitive(
        mode, kind, ::dnnl::rnn_direction::bidirectional_concat, bias_mode,
        src_dt, src_format_tag, seq_length, batch_size, output_size,
        2 * output_size, 1, direction_num, hidden_size, gate_num,
        projection_size, data, offset, 1, weight_size_query,
        workspace_size_query, scratchpad_size_query);
  } else {
    e = execute_rnn_forward_primitive(
        mode, kind, ::dnnl::rnn_direction::unidirectional_left2right, bias_mode,
        src_dt, src_format_tag, seq_length, batch_size, output_size,
        output_size, layer_size, direction_num, hidden_size, gate_num,
        projection_size, data, offset, 1, weight_size_query,
        workspace_size_query, scratchpad_size_query);
  }

  return exit_primitive(e);
}

inline
sycl::event engine_ext::execute_rnn_forward_primitive(
    rnn_mode mode, ::dnnl::prop_kind kind, ::dnnl::rnn_direction direction,
    rnn_bias_mode bias_mode, ::dnnl::memory::data_type dt,
    ::dnnl::memory::format_tag tag, int seq_length, int batch_size, int src_c,
    int dst_c, int layer_size, int direction_num, int hidden_size, int gate_num,
    int projection_size, std::vector<void *> &data, std::vector<int> &offset,
    int iter_num, size_t *weight_size, size_t *workspace_size,
    size_t *scratchpad_size) {

  sycl::event e;
  ::dnnl::primitive *p = nullptr;
  std::unordered_map<int, ::dnnl::memory> *args = nullptr;
  detail::primitive_cache_key_type key;
  std::unordered_map<int, ::dnnl::memory> *execution_args;
  ::dnnl::memory::desc bias_desc(
      {layer_size, direction_num, gate_num, hidden_size}, dt,
      ::dnnl::memory::format_tag::ldgo);
  ::dnnl::memory::desc weight_layer_desc(
      {layer_size, direction_num,
       projection_size ? projection_size : hidden_size, gate_num, hidden_size},
      dt, ::dnnl::memory::format_tag::ldigo);
  ::dnnl::memory::desc weight_iter_desc(
      {layer_size, direction_num,
       projection_size ? projection_size : hidden_size, gate_num, hidden_size},
      dt, ::dnnl::memory::format_tag::ldigo);
  ::dnnl::memory::desc projection_desc;
  if (projection_size) {
    projection_desc = ::dnnl::memory::desc(
        {layer_size, direction_num, hidden_size, projection_size}, dt,
        ::dnnl::memory::format_tag::ldio);
  }

  if (weight_size) {
    *weight_size +=
        (weight_layer_desc.get_size() + weight_iter_desc.get_size() +
         projection_desc.get_size() + bias_desc.get_size()) *
        iter_num;
    return e;
  }

  ::dnnl::memory::desc src_desc({seq_length, batch_size, src_c}, dt, tag);
  ::dnnl::memory::desc dst_desc({seq_length, batch_size, dst_c}, dt, tag);
  ::dnnl::memory::desc iter_desc(
      {layer_size, direction_num, batch_size,
       projection_size ? projection_size : hidden_size},
      dt, ::dnnl::memory::format_tag::ldnc);
  ::dnnl::memory::desc iter_c_desc(
      {layer_size, direction_num, batch_size, hidden_size}, dt,
      ::dnnl::memory::format_tag::ldnc);

  ::dnnl::memory::desc workspace_desc;
  ::dnnl::memory::desc scratchpad_desc;
  ::dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(::dnnl::scratchpad_mode::user);

  if (mode == rnn_mode::vanilla_relu || mode == rnn_mode::vanilla_tanh) {
    auto primitive = create_primitive_args_or_get<::dnnl::vanilla_rnn_forward>(
        kind,
        mode == rnn_mode::vanilla_relu ? ::dnnl::algorithm::eltwise_relu
                                       : ::dnnl::algorithm::eltwise_tanh,
        direction, src_desc, iter_desc, weight_layer_desc, weight_iter_desc,
        bias_desc, dst_desc, iter_desc, attr);

    auto pd = get_primitive_desc<::dnnl::vanilla_rnn_forward>(
        primitive.second.primitive);

    workspace_desc = pd.workspace_desc();
    scratchpad_desc = pd.scratchpad_desc();
    if (workspace_size && scratchpad_size) {
      *workspace_size += workspace_desc.get_size() * iter_num;
      *scratchpad_size = scratchpad_desc.get_size() > *scratchpad_size
                             ? scratchpad_desc.get_size()
                             : *scratchpad_size;
    } else {
      key = primitive.first;
      p = primitive.second.primitive;
      args = primitive.second.args;
    }
  } else if (mode == rnn_mode::gru) {
    auto primitive = create_primitive_args_or_get<::dnnl::gru_forward>(
        kind, direction, src_desc, iter_desc, weight_layer_desc,
        weight_iter_desc, bias_desc, dst_desc, iter_desc, attr);

    auto pd =
        get_primitive_desc<::dnnl::gru_forward>(primitive.second.primitive);

    workspace_desc = pd.workspace_desc();
    scratchpad_desc = pd.scratchpad_desc();
    if (workspace_size && scratchpad_size) {
      *workspace_size += workspace_desc.get_size() * iter_num;
      *scratchpad_size = scratchpad_desc.get_size() > *scratchpad_size
                             ? scratchpad_desc.get_size()
                             : *scratchpad_size;
    } else {
      key = primitive.first;
      p = primitive.second.primitive;
      args = primitive.second.args;
    }
  } else if (mode == rnn_mode::lstm) {
    auto primitive = create_primitive_args_or_get<::dnnl::lstm_forward>(
        kind, direction, src_desc, iter_desc, iter_c_desc, weight_layer_desc,
        weight_iter_desc, ::dnnl::memory::desc(), projection_desc, bias_desc,
        dst_desc, iter_desc, iter_c_desc, attr);

    auto pd =
        get_primitive_desc<::dnnl::lstm_forward>(primitive.second.primitive);

    workspace_desc = pd.workspace_desc();
    scratchpad_desc = pd.scratchpad_desc();
    if (workspace_size && scratchpad_size) {
      *workspace_size += workspace_desc.get_size() * iter_num;
      *scratchpad_size = scratchpad_desc.get_size() > *scratchpad_size
                             ? scratchpad_desc.get_size()
                             : *scratchpad_size;
    } else {
      key = primitive.first;
      p = primitive.second.primitive;
      args = primitive.second.args;
    }
  }

  for (int i = 0; i < iter_num; i++) {
    void *in_cache = data[0], *out_cache = data[1], *dst_iter_c_cache = nullptr,
         *dst_iter_cache = ((uint8_t *)(data[3]) + offset[1]);
    if (mode == rnn_mode::lstm) {
      dst_iter_c_cache = (uint8_t *)(data[4]) + offset[2];
    }
    if (!workspace_size) {
      insert_arg(args, DNNL_ARG_SRC_LAYER, src_desc, data[0]);
      insert_arg(args, DNNL_ARG_DST_LAYER, dst_desc, data[1]);
      insert_arg(args, DNNL_ARG_SCRATCHPAD, scratchpad_desc, data[8]);
      auto insert_rnn_arg = [&](int arg_name, ::dnnl::memory::desc &d, void *data,
                             int &offset) {
        insert_arg(args, arg_name, d, (uint8_t *)data + offset);
        offset += d.get_size();
      };
      insert_rnn_arg(DNNL_ARG_SRC_ITER, iter_desc, data[2], offset[0]);
      insert_rnn_arg(DNNL_ARG_DST_ITER, iter_desc, data[3], offset[1]);

      if (mode == rnn_mode::lstm) {
        insert_rnn_arg(DNNL_ARG_SRC_ITER_C, iter_c_desc, data[4], offset[2]);
        insert_rnn_arg(DNNL_ARG_DST_ITER_C, iter_c_desc, data[5], offset[3]);
      }
      insert_rnn_arg(DNNL_ARG_WEIGHTS_LAYER, weight_layer_desc, data[6],
                  offset[4]);
      insert_rnn_arg(DNNL_ARG_WEIGHTS_ITER, weight_iter_desc, data[6], offset[4]);
      if (projection_size) {
        insert_rnn_arg(DNNL_ARG_WEIGHTS_PROJECTION, projection_desc, data[6],
                    offset[4]);
      }
      if (bias_mode == rnn_bias_mode::none) {
        _q->memset((uint8_t *)(data[6]) + offset[4], 0, bias_desc.get_size());
      }
      insert_rnn_arg(DNNL_ARG_BIAS, bias_desc, data[6], offset[4]);
      if (kind == ::dnnl::prop_kind::forward_training) {
        insert_rnn_arg(DNNL_ARG_WORKSPACE, workspace_desc, data[7], offset[5]);
      }
      if (mode == rnn_mode::vanilla_relu || mode == rnn_mode::vanilla_tanh) {
        execute_primitive<::dnnl::vanilla_rnn_forward>(
            {key, {static_cast<::dnnl::vanilla_rnn_forward *>(p), args}});
      } else if (mode == rnn_mode::gru) {
        execute_primitive<::dnnl::gru_forward>(
            {key, {static_cast<::dnnl::gru_forward *>(p), args}});
      } else if (mode == rnn_mode::lstm) {
        execute_primitive<::dnnl::lstm_forward>(
            {key, {static_cast<::dnnl::lstm_forward *>(p), args}});
      }
      if (i != iter_num - 1) {
        std::swap(data[0], data[1]);
      }
    }
    if (kind == ::dnnl::prop_kind::forward_training) {
      if (workspace_size) {
        *workspace_size +=
            (src_desc.get_size() + dst_desc.get_size() + iter_desc.get_size());
        if (mode == rnn_mode::lstm) {
          *workspace_size += iter_c_desc.get_size();
        }
      } else {
        _q->memcpy((uint8_t *)(data[7]) + offset[5], in_cache,
                   src_desc.get_size());
        offset[5] += src_desc.get_size();
        _q->memcpy((uint8_t *)(data[7]) + offset[5], out_cache,
                   dst_desc.get_size());
        offset[5] += dst_desc.get_size();
        _q->memcpy((uint8_t *)(data[7]) + offset[5], dst_iter_cache,
                   iter_desc.get_size());
        offset[5] += iter_desc.get_size();
        if (mode == rnn_mode::lstm) {
          _q->memcpy((uint8_t *)(data[7]) + offset[5], dst_iter_c_cache,
                     iter_c_desc.get_size());
          offset[5] += iter_c_desc.get_size();
        }
      }
    }
  }
  return e;
}

inline
sycl::event engine_ext::execute_rnn_backward_primitive(
    rnn_mode mode, ::dnnl::rnn_direction direction, rnn_bias_mode bias_mode,
    ::dnnl::memory::data_type dt, ::dnnl::memory::format_tag tag,
    int seq_length, int batch_size, int src_c, int dst_c, int layer_size,
    int direction_num, int hidden_size, int gate_num, int projection_size,
    std::vector<void *> &data, std::vector<int> &offset, int iter_num) {

  sycl::event e;
  ::dnnl::primitive *p = nullptr;
  std::unordered_map<int, ::dnnl::memory> *args = nullptr;
  detail::primitive_cache_key_type key;
  ::dnnl::prop_kind fkind = ::dnnl::prop_kind::forward_training;
  ::dnnl::prop_kind bkind = ::dnnl::prop_kind::backward;
  ::dnnl::memory::desc bias_desc(
      {layer_size, direction_num, gate_num, hidden_size}, dt,
      ::dnnl::memory::format_tag::ldgo);
  ::dnnl::memory::desc weight_layer_desc(
      {layer_size, direction_num,
       projection_size ? projection_size : hidden_size, gate_num, hidden_size},
      dt, ::dnnl::memory::format_tag::ldigo);
  ::dnnl::memory::desc weight_iter_desc(
      {layer_size, direction_num,
       projection_size ? projection_size : hidden_size, gate_num, hidden_size},
      dt, ::dnnl::memory::format_tag::ldigo);
  ::dnnl::memory::desc diff_weight_layer_desc(
      {layer_size, direction_num,
       projection_size ? projection_size : hidden_size, gate_num, hidden_size},
      dt, ::dnnl::memory::format_tag::ldgoi);
  ::dnnl::memory::desc diff_weight_iter_desc(
      {layer_size, direction_num,
       projection_size ? projection_size : hidden_size, gate_num, hidden_size},
      dt, ::dnnl::memory::format_tag::ldgoi);
  ::dnnl::memory::desc projection_desc, diff_projection_desc;
  if (projection_size) {
    projection_desc = ::dnnl::memory::desc(
        {layer_size, direction_num, hidden_size, projection_size}, dt,
        ::dnnl::memory::format_tag::ldio);
    diff_projection_desc = ::dnnl::memory::desc(
        {layer_size, direction_num, hidden_size, projection_size}, dt,
        ::dnnl::memory::format_tag::ldoi);
  }

  ::dnnl::memory::desc src_desc({seq_length, batch_size, src_c}, dt, tag);
  ::dnnl::memory::desc dst_desc({seq_length, batch_size, dst_c}, dt, tag);
  ::dnnl::memory::desc iter_desc(
      {layer_size, direction_num, batch_size,
       projection_size ? projection_size : hidden_size},
      dt, ::dnnl::memory::format_tag::ldnc);
  ::dnnl::memory::desc iter_c_desc(
      {layer_size, direction_num, batch_size, hidden_size}, dt,
      ::dnnl::memory::format_tag::ldnc);

  ::dnnl::memory::desc workspace_desc;
  ::dnnl::memory::desc scratchpad_desc;
  ::dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(::dnnl::scratchpad_mode::user);

  if (mode == rnn_mode::vanilla_relu || mode == rnn_mode::vanilla_tanh) {
    auto fpd = create_primitive_desc<::dnnl::vanilla_rnn_forward>(
        fkind,
        mode == rnn_mode::vanilla_relu ? ::dnnl::algorithm::eltwise_relu
                                       : ::dnnl::algorithm::eltwise_tanh,
        direction, src_desc, iter_desc, weight_layer_desc, weight_iter_desc,
        bias_desc, dst_desc, iter_desc, attr);
    auto primitive = create_primitive_args_or_get<::dnnl::vanilla_rnn_backward>(
        bkind,
        mode == rnn_mode::vanilla_relu ? ::dnnl::algorithm::eltwise_relu
                                       : ::dnnl::algorithm::eltwise_tanh,
        direction, src_desc, iter_desc, diff_weight_layer_desc,
        diff_weight_iter_desc, bias_desc, dst_desc, iter_desc, src_desc,
        iter_desc, weight_layer_desc, weight_iter_desc, bias_desc, dst_desc,
        iter_desc, fpd, attr);
    auto pd = get_primitive_desc<::dnnl::vanilla_rnn_backward>(
        primitive.second.primitive);
    workspace_desc = pd.workspace_desc();
    scratchpad_desc = pd.scratchpad_desc();
    key = primitive.first;
    p = primitive.second.primitive;
    args = primitive.second.args;
  } else if (mode == rnn_mode::gru) {
    auto fpd = create_primitive_desc<::dnnl::gru_forward>(
        fkind, direction, src_desc, iter_desc, weight_layer_desc,
        weight_iter_desc, bias_desc, dst_desc, iter_desc, attr);
    auto primitive = create_primitive_args_or_get<::dnnl::gru_backward>(
        bkind, direction, src_desc, iter_desc, diff_weight_layer_desc,
        diff_weight_iter_desc, bias_desc, dst_desc, iter_desc, src_desc,
        iter_desc, weight_layer_desc, weight_iter_desc, bias_desc, dst_desc,
        iter_desc, fpd, attr);
    auto pd =
        get_primitive_desc<::dnnl::gru_backward>(primitive.second.primitive);
    workspace_desc = pd.workspace_desc();
    scratchpad_desc = pd.scratchpad_desc();
    key = primitive.first;
    p = primitive.second.primitive;
    args = primitive.second.args;
  } else if (mode == rnn_mode::lstm) {
    auto fpd = create_primitive_desc<::dnnl::lstm_forward>(
        fkind, direction, src_desc, iter_desc, iter_c_desc, weight_layer_desc,
        weight_iter_desc, ::dnnl::memory::desc(), projection_desc, bias_desc,
        dst_desc, iter_desc, iter_c_desc, attr);
    auto primitive = create_primitive_args_or_get<::dnnl::lstm_backward>(
        bkind, direction, src_desc, iter_desc, iter_c_desc,
        diff_weight_layer_desc, diff_weight_iter_desc, ::dnnl::memory::desc(),
        diff_projection_desc, bias_desc, dst_desc, iter_desc, iter_c_desc,
        src_desc, iter_desc, iter_c_desc, weight_layer_desc, weight_iter_desc,
        ::dnnl::memory::desc(), projection_desc, bias_desc, dst_desc, iter_desc,
        iter_c_desc, fpd, attr);
    auto pd =
        get_primitive_desc<::dnnl::lstm_backward>(primitive.second.primitive);
    workspace_desc = pd.workspace_desc();
    scratchpad_desc = pd.scratchpad_desc();
    key = primitive.first;
    p = primitive.second.primitive;
    args = primitive.second.args;
  }

  for (int i = 0; i < iter_num; i++) {
    insert_arg(args, DNNL_ARG_DIFF_SRC_LAYER, src_desc, data[8]);
    insert_arg(args, DNNL_ARG_DIFF_DST_LAYER, dst_desc, data[9]);
    insert_arg(args, DNNL_ARG_SCRATCHPAD, scratchpad_desc, data[15]);
    auto insert_rnn_arg = [&](int arg_name, ::dnnl::memory::desc &d, void *data,
                           int &offset) {
      offset += d.get_size();
      insert_arg(args, arg_name, d, (uint8_t *)data - offset);
    };
    if (mode == rnn_mode::lstm) {
      insert_rnn_arg(DNNL_ARG_DST_ITER_C, iter_c_desc, data[7], offset[0]);
      insert_rnn_arg(DNNL_ARG_SRC_ITER_C, iter_c_desc, data[4], offset[2]);
    }
    insert_rnn_arg(DNNL_ARG_DST_ITER, iter_desc, data[7], offset[0]);
    insert_rnn_arg(DNNL_ARG_DST_LAYER, dst_desc, data[7], offset[0]);
    insert_rnn_arg(DNNL_ARG_SRC_LAYER, src_desc, data[7], offset[0]);
    insert_rnn_arg(DNNL_ARG_WORKSPACE, workspace_desc, data[7], offset[0]);
    insert_rnn_arg(DNNL_ARG_SRC_ITER, iter_desc, data[2], offset[1]);
    insert_rnn_arg(DNNL_ARG_BIAS, bias_desc, data[6], offset[3]);
    if (projection_size) {
      insert_rnn_arg(DNNL_ARG_WEIGHTS_PROJECTION, diff_projection_desc, data[6],
                  offset[3]);
    }
    insert_rnn_arg(DNNL_ARG_WEIGHTS_ITER, diff_weight_iter_desc, data[6],
                offset[3]);
    insert_rnn_arg(DNNL_ARG_WEIGHTS_LAYER, diff_weight_layer_desc, data[6],
                offset[3]);
    insert_rnn_arg(DNNL_ARG_DIFF_SRC_ITER, iter_desc, data[10], offset[4]);
    insert_rnn_arg(DNNL_ARG_DIFF_DST_ITER, iter_desc, data[11], offset[5]);
    if (mode == rnn_mode::lstm) {
      insert_rnn_arg(DNNL_ARG_DIFF_SRC_ITER_C, iter_c_desc, data[12], offset[6]);
      insert_rnn_arg(DNNL_ARG_DIFF_DST_ITER_C, iter_c_desc, data[13], offset[7]);
    }
    insert_rnn_arg(DNNL_ARG_DIFF_BIAS, bias_desc, data[14], offset[8]);
    if (bias_mode == rnn_bias_mode::none) {
      _q->memset((uint8_t *)(data[14]) - offset[8], 0, bias_desc.get_size());
    }
    if (projection_size) {
      insert_rnn_arg(DNNL_ARG_DIFF_WEIGHTS_PROJECTION, projection_desc, data[14],
                  offset[8]);
    }
    insert_rnn_arg(DNNL_ARG_DIFF_WEIGHTS_ITER, weight_iter_desc, data[14],
                offset[8]);
    insert_rnn_arg(DNNL_ARG_DIFF_WEIGHTS_LAYER, weight_layer_desc, data[14],
                offset[8]);
    if (mode == rnn_mode::vanilla_relu || mode == rnn_mode::vanilla_tanh) {
      e = execute_primitive<::dnnl::vanilla_rnn_backward>(
          {key, {static_cast<::dnnl::vanilla_rnn_backward *>(p), args}});
    } else if (mode == rnn_mode::gru) {
      e = execute_primitive<::dnnl::gru_backward>(
          {key, {static_cast<::dnnl::gru_backward *>(p), args}});
    } else if (mode == rnn_mode::lstm) {
      e = execute_primitive<::dnnl::lstm_backward>(
          {key, {static_cast<::dnnl::lstm_backward *>(p), args}});
    }
    if (i != iter_num - 1) {
      std::swap(data[8], data[9]);
    }
  }
  return e;
}

#define EMPTY_CACHE_KEY(type)                                                  \
  template <>                                                                  \
  inline void engine_ext::generate_cache_key<type>(std::string & key_buffer,   \
                                                   const type &arg) {}

EMPTY_CACHE_KEY(::dnnl::engine)
EMPTY_CACHE_KEY(::dnnl::convolution_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::eltwise_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::softmax_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::pooling_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::lrn_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::batch_normalization_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::vanilla_rnn_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::lstm_forward::primitive_desc)
EMPTY_CACHE_KEY(::dnnl::gru_forward::primitive_desc)
#undef EMPTY_CACHE_KEY

template <>
inline void engine_ext::generate_cache_key<std::vector<float>>(
    std::string &key_buffer, const std::vector<float> &vec) {
  key_buffer.append((char *)vec.data(), vec.size() * sizeof(float));
}

template <>
inline void engine_ext::generate_cache_key<::dnnl::primitive_attr>(
    std::string &key_buffer, const ::dnnl::primitive_attr &attr) {
  if (!attr) {
    return;
  }
  auto math_mode = (uint8_t)attr.get_fpmath_mode();
  key_buffer.append((char *)&math_mode, sizeof(uint8_t));
}

template <>
inline void engine_ext::generate_cache_key<::dnnl::memory::dims>(
    std::string &key_buffer, const ::dnnl::memory::dims &dims) {
  key_buffer.append((char *)dims.data(), dims.size() * sizeof(int64_t));
}

template <>
inline void engine_ext::generate_cache_key<::dnnl::memory::desc>(
    std::string &key_buffer, const ::dnnl::memory::desc &desc) {
  uint8_t params[3] = {(uint8_t)desc.get_format_kind(),
                       (uint8_t)desc.get_ndims(),
                       (uint8_t)desc.get_data_type()};
  generate_cache_key(key_buffer, desc.get_inner_blks());
  generate_cache_key(key_buffer, desc.get_dims());
  generate_cache_key(key_buffer, desc.get_strides());
}

template <typename T>
void engine_ext::generate_cache_key(std::string &key_buffer, const T &arg) {
  key_buffer.append((char *)&arg, sizeof(T));
}

template <typename T, typename... args_type>
void engine_ext::generate_cache_key(std::string &key_buffer, const T &first_arg,
                                    const args_type &...args) {
  generate_cache_key(key_buffer, first_arg);
  generate_cache_key(key_buffer, args...);
}

template <typename primitive_type, typename... args_type>
std::pair<detail::primitive_cache_key_type, detail::primitive_and_args>
engine_ext::create_primitive_args_or_get(args_type &&...args) {
  std::string buffer;
  buffer.reserve(512);
  generate_cache_key(buffer, std::forward<args_type>(args)...);
  buffer.append(std::to_string(_engine_id));
  auto value = _primitive_cache.get(buffer);
  primitive_type *p = nullptr;
  std::unordered_map<int, ::dnnl::memory> *a = nullptr;
  if (value) {
    p = (primitive_type *)value->_primitive;
    a = value->_args;
  } else {
    p = new primitive_type(create_primitive_desc<primitive_type>(
        std::forward<args_type>(args)...));
    a = new std::unordered_map<int, ::dnnl::memory>();
  }
  return {buffer, {p, a}};
}

template <typename primitive_type>
typename primitive_type::primitive_desc
engine_ext::get_primitive_desc(::dnnl::primitive *p) {
  return typename primitive_type::primitive_desc(
      const_cast<dnnl_primitive_desc_t>(p->get_primitive_desc()));
}

template <typename primitive_type, typename... args_type>
typename primitive_type::primitive_desc
engine_ext::create_primitive_desc(args_type &&...args) {
  return typename primitive_type::primitive_desc(
      *_eng, std::forward<args_type>(args)...);
}

inline
void engine_ext::fill(const memory_desc_ext &src_desc, void *src,
                      const void *valuePtr) {
  async_fill(src_desc, src, valuePtr).wait();
}

inline
void engine_ext::reorder(float alpha, const memory_desc_ext &src_desc,
                         void *src, float beta, const memory_desc_ext &dst_desc,
                         void *dst) {
  async_reorder(alpha, src_desc, src, beta, dst_desc, dst).wait();
}

inline
void engine_ext::scale(float alpha, const memory_desc_ext &src_desc,
                       void *src) {
  async_scale(alpha, src_desc, src).wait();
}
inline
void engine_ext::sum(float alpha, const memory_desc_ext &src_desc, void *src,
                     float beta, const memory_desc_ext &dst_desc, void *dst) {
  async_sum(alpha, src_desc, src, beta, dst_desc, dst).wait();
}
inline
void engine_ext::activation_forward(activation_desc &desc, float alpha,
                                    const memory_desc_ext &src_desc, void *src,
                                    float beta, const memory_desc_ext &dst_desc,
                                    void *dst) {
  async_activation_forward(desc, alpha, src_desc, src, beta, dst_desc, dst)
      .wait();
}
inline
void engine_ext::activation_backward(
    activation_desc &desc, float alpha, const memory_desc_ext &dst_desc,
    void *dst, const memory_desc_ext &diff_dst_desc, void *diff_dst,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src) {
  async_activation_backward(desc, alpha, dst_desc, dst, diff_dst_desc, diff_dst,
                            src_desc, src, beta, diff_src_desc, diff_src)
      .wait();
}
inline
void engine_ext::pooling_forward(pooling_desc &desc, float alpha,
                                 const memory_desc_ext &src_desc, void *src,
                                 float beta, const memory_desc_ext &dst_desc,
                                 void *dst,
                                 ::dnnl::memory *workspace) {
  async_pooling_forward(desc, alpha, src_desc, src, beta, dst_desc, dst,
                        workspace).wait();
}

inline
void engine_ext::pooling_backward(
    pooling_desc &desc, float alpha, const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &diff_dst_desc, void *diff_dst,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src,
    ::dnnl::memory *workspace) {
  async_pooling_backward(desc, alpha, dst_desc, dst, diff_dst_desc, diff_dst,
                         src_desc, src, beta, diff_src_desc, diff_src,
                         workspace)
      .wait();
}

inline
void engine_ext::softmax_forward(softmax_algorithm alg, softmax_mode mode,
                                 float alpha, const memory_desc_ext &src_desc,
                                 void *src, float beta,
                                 const memory_desc_ext &dst_desc, void *dst) {
  async_softmax_forward(alg, mode, alpha, src_desc, src, beta, dst_desc, dst)
      .wait();
}

inline
void engine_ext::softmax_backward(softmax_algorithm alg, softmax_mode mode,
                                  float alpha, const memory_desc_ext &dst_desc,
                                  void *dst,
                                  const memory_desc_ext &diff_dst_desc,
                                  void *diff_dst, float beta,
                                  const memory_desc_ext &diff_src_desc,
                                  void *diff_src) {
  async_softmax_backward(alg, mode, alpha, dst_desc, dst, diff_dst_desc,
                         diff_dst, beta, diff_src_desc, diff_src)
      .wait();
}

inline
void engine_ext::lrn_forward(lrn_desc &desc, float alpha,
                             const memory_desc_ext &src_desc, void *src,
                             float beta, const memory_desc_ext &dst_desc,
                             void *dst, ::dnnl::memory *workspace) {
  async_lrn_forward(desc, alpha, src_desc, src, beta, dst_desc, dst, workspace)
      .wait();
}

inline
void engine_ext::lrn_backward(lrn_desc &desc, float alpha,
                              const memory_desc_ext &dst_desc, void *dst,
                              const memory_desc_ext &diff_dst_desc,
                              void *diff_dst, const memory_desc_ext &src_desc,
                              void *src, float beta,
                              const memory_desc_ext &diff_src_desc,
                              void *diff_src,
                              ::dnnl::memory *workspace) {
  async_lrn_backward(desc, alpha, dst_desc, dst, diff_dst_desc, diff_dst,
                     src_desc, src, beta, diff_src_desc, diff_src, workspace)
      .wait();
}

inline
sycl::event engine_ext::async_fill(const memory_desc_ext &src_desc, void *src,
                             const void *valuePtr) {
  ::dnnl::memory::data_type dt = src_desc.get_desc().get_data_type();
  unsigned mem_size = src_desc.get_size();
  switch (dt) {
  case ::dnnl::memory::data_type::f32:
    return fill_with_type<float>(_q, src, valuePtr, mem_size);
  case ::dnnl::memory::data_type::f16:
    return fill_with_type<sycl::half>(_q, src, valuePtr, mem_size);
  case ::dnnl::memory::data_type::s32:
    return fill_with_type<int32_t>(_q, src, valuePtr, mem_size);
  case ::dnnl::memory::data_type::s8:
    return fill_with_type<int8_t>(_q, src, valuePtr, mem_size);
  case ::dnnl::memory::data_type::u8:
    return fill_with_type<uint8_t>(_q, src, valuePtr, mem_size);
  default:
    throw std::runtime_error("async_fill: unsupported data type.");
  }
}

inline
sycl::event engine_ext::async_reorder(float alpha, const memory_desc_ext &src_desc,
                                void *src, float beta,
                                const memory_desc_ext &dst_desc, void *dst) {
  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }
  enter_primitive(2 * dst_desc.get_size());

  auto primitive_args = create_primitive_args_or_get<::dnnl::reorder>(
      src_desc.get_desc(), *_eng, dst_desc.get_desc());

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);

  return exit_primitive(execute_primitive<::dnnl::reorder>(
      primitive_args, {{alpha, beta, DNNL_ARG_DST, dst_desc, dst}}));
}

inline
sycl::event engine_ext::async_scale(float alpha, const memory_desc_ext &src_desc,
                              void *src) {
  if (alpha == 1.f) {
    return sycl::event();
  }
  size_t cache_size = src_desc.get_size();
  enter_primitive(cache_size);
  void *src_cache = allocate(cache_size);
  _q->memcpy(src_cache, src, cache_size);
  auto primitive_args = create_primitive_args_or_get<::dnnl::eltwise_forward>(
      ::dnnl::prop_kind::forward_inference, ::dnnl::algorithm::eltwise_linear,
      src_desc.get_desc(), src_desc.get_desc(), alpha, 0.f);

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src_cache);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, src_desc.get_desc(),
             src);

  return exit_primitive(
      execute_primitive<::dnnl::eltwise_forward>(primitive_args));
}

inline sycl::event
engine_ext::async_sum(float alpha, const memory_desc_ext &src_desc, void *src,
                      float beta, const memory_desc_ext &dst_desc, void *dst) {
  if (alpha == 0.f && beta == 1.f) {
    return sycl::event();
  }
  size_t cache_size = dst_desc.get_size();
  enter_primitive(cache_size);
  void *dst_cache = allocate(dst_desc);
  _q->memcpy(dst_cache, dst, cache_size);

  auto primitive_args = create_primitive_args_or_get<::dnnl::sum>(
      std::vector<float>{alpha, beta},
      std::vector<::dnnl::memory::desc>{src_desc.get_desc(),
                                        dst_desc.get_desc()});
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_MULTIPLE_SRC,
             src_desc.get_desc(), src);
  insert_arg(primitive_args.second.args, DNNL_ARG_MULTIPLE_SRC + 1,
             dst_desc.get_desc(), dst_cache);

  return exit_primitive(execute_primitive<::dnnl::sum>(primitive_args));
}

inline
sycl::event engine_ext::async_binary(binary_op op, float alpha_0,
                               const memory_desc_ext &src_desc_0, void *src_0,
                               float alpha_1, const memory_desc_ext &src_desc_1,
                               void *src_1, float beta,
                               const memory_desc_ext &dst_desc, void *dst) {
  ::dnnl::algorithm onednn_algorithm;
  switch (op) {
  case binary_op::max:
    onednn_algorithm = ::dnnl::algorithm::binary_max;
    break;
  case binary_op::min:
    onednn_algorithm = ::dnnl::algorithm::binary_min;
    break;
  case binary_op::add:
    onednn_algorithm = ::dnnl::algorithm::binary_add;
    break;
  case binary_op::sub:
    onednn_algorithm = ::dnnl::algorithm::binary_sub;
    break;
  case binary_op::mul:
    onednn_algorithm = ::dnnl::algorithm::binary_mul;
    break;
  case binary_op::div:
    onednn_algorithm = ::dnnl::algorithm::binary_div;
    break;
  case binary_op::sqrt:
    onednn_algorithm = ::dnnl::algorithm::eltwise_sqrt;
    break;
  case binary_op::neg:
    onednn_algorithm = ::dnnl::algorithm::eltwise_linear;
    break;
  }
  size_t src0_cache_size = src_desc_0.get_size();
  size_t src1_cache_size = src_desc_1.get_size();
  size_t dst_cache_size = dst_desc.get_size();
  enter_primitive(2 * src0_cache_size + 2 * src1_cache_size +
                  5 * dst_cache_size);
  if (onednn_algorithm == ::dnnl::algorithm::eltwise_sqrt ||
      onednn_algorithm == ::dnnl::algorithm::eltwise_linear) {
    void *src_cache = nullptr, *dst_cache = nullptr;
    src_cache = allocate(src0_cache_size);
    dst_cache = allocate(dst_cache_size);
    _q->memcpy(src_cache, src_0, src0_cache_size);
    _q->memcpy(dst_cache, dst, dst_cache_size);
    async_scale(alpha_0, src_desc_0, src_cache);
    async_scale(beta, dst_desc, dst_cache);

    // Let the output = 1 - input to simulate the behavior of neg.
    auto primitive_args = create_primitive_args_or_get<::dnnl::eltwise_forward>(
        ::dnnl::prop_kind::forward_inference, onednn_algorithm,
        src_desc_0.get_desc(), dst_desc.get_desc(), -1.f, 1.f);

    insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc_0.get_desc(),
               src_cache);
    insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
               dst);

    execute_primitive<::dnnl::eltwise_forward>(
        primitive_args, {{1.f, 0.f, DNNL_ARG_DST, dst_desc, dst}});
    return exit_primitive(
        async_sum(1.f, dst_desc, dst_cache, 1.f, dst_desc, dst));
  }

  void *src_0_cache = nullptr, *src_1_cache = nullptr, *dst_cache = nullptr;

  src_0_cache = allocate(src0_cache_size);
  src_1_cache = allocate(src1_cache_size);
  dst_cache = allocate(dst_cache_size);

  _q->memcpy(src_0_cache, src_0, src0_cache_size);
  _q->memcpy(src_1_cache, src_1, src1_cache_size);
  _q->memcpy(dst_cache, dst, dst_cache_size);

  async_scale(alpha_0, src_desc_0, src_0_cache);
  async_scale(alpha_1, src_desc_1, src_1_cache);
  async_scale(beta, dst_desc, dst_cache);

  auto primitive_args = create_primitive_args_or_get<::dnnl::binary>(
      onednn_algorithm, src_desc_0.get_desc(), src_desc_1.get_desc(),
      dst_desc.get_desc());

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC_0, src_desc_0.get_desc(),
             src_0_cache);
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC_1, src_desc_1.get_desc(),
             src_1_cache);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);

  execute_primitive<::dnnl::binary>(primitive_args,
                                    {{1.f, 0.f, DNNL_ARG_DST, dst_desc, dst}});
  return exit_primitive(
      async_sum(1.f, dst_desc, dst_cache, 1.f, dst_desc, dst));
}

inline
sycl::event engine_ext::async_reduction(reduction_op op, float alpha,
                                  const memory_desc_ext &src_desc, void *src,
                                  float beta, const memory_desc_ext &dst_desc,
                                  void *dst) {
  if (alpha == 0.f && beta == 1.f) {
    return sycl::event();
  }
  size_t src_cache_size = src_desc.get_size();
  size_t dst_cache_size = dst_desc.get_size();
  enter_primitive(3 * src_cache_size + 2 * dst_cache_size);
  float p = 2.f;
  ::dnnl::algorithm onednn_algorithm;
  void *cache = nullptr;
  switch (op) {
  case reduction_op::amax:
    cache = allocate(src_cache_size);
    activation_desc adesc;
    adesc.set_algorithm(::dnnl::algorithm::eltwise_abs);
    async_activation_forward(adesc, 1.f, src_desc, src, 0.f, src_desc, cache);
    onednn_algorithm = ::dnnl::algorithm::reduction_max;
    src = cache;
    break;
  case reduction_op::max:
    onednn_algorithm = ::dnnl::algorithm::reduction_max;
    break;
  case reduction_op::min:
    onednn_algorithm = ::dnnl::algorithm::reduction_min;
    break;
  case reduction_op::sum:
    onednn_algorithm = ::dnnl::algorithm::reduction_sum;
    break;
  case reduction_op::mean:
    onednn_algorithm = ::dnnl::algorithm::reduction_mean;
    break;
  case reduction_op::mul:
    onednn_algorithm = ::dnnl::algorithm::reduction_mul;
    break;
  case reduction_op::mul_no_zeros:
    cache = allocate(src_cache_size);
    transform_no_zero(src_desc, src, cache);
    onednn_algorithm = ::dnnl::algorithm::reduction_mul;
    src = cache;
    break;
  case reduction_op::norm1:
    p = 1.f;
    onednn_algorithm = ::dnnl::algorithm::reduction_norm_lp_power_p_sum;
    break;
  case reduction_op::norm2:
    onednn_algorithm = ::dnnl::algorithm::reduction_norm_lp_sum;
    break;
  }
  auto primitive_args = create_primitive_args_or_get<::dnnl::reduction>(
      onednn_algorithm, src_desc.get_desc(), dst_desc.get_desc(), p, 0.f);

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);

  return exit_primitive(execute_primitive<::dnnl::reduction>(
      primitive_args, {{alpha, beta, DNNL_ARG_DST, dst_desc, dst}}));
}

inline
sycl::event engine_ext::async_activation_forward(activation_desc &desc, float alpha,
                                           const memory_desc_ext &src_desc,
                                           void *src, float beta,
                                           const memory_desc_ext &dst_desc,
                                           void *dst) {
  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }
  enter_primitive(2 * dst_desc.get_size());
  auto primitive_args = create_primitive_args_or_get<::dnnl::eltwise_forward>(
      ::dnnl::prop_kind::forward, desc.get_algorithm(), src_desc.get_desc(),
      dst_desc.get_desc(), desc.get_alpha(), desc.get_beta());

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);

  return exit_primitive(execute_primitive<::dnnl::eltwise_forward>(
      primitive_args, {{alpha, beta, DNNL_ARG_DST, dst_desc, dst}}));
}

inline
sycl::event engine_ext::async_activation_backward(
    activation_desc &desc, float alpha, const memory_desc_ext &dst_desc,
    void *dst, const memory_desc_ext &diff_dst_desc, void *diff_dst,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src) {

  if (scale_parameter_preprocess({{alpha, beta, diff_src_desc, diff_src}})) {
    return sycl::event();
  }
  enter_primitive(2 * diff_src_desc.get_size());
  ::dnnl::memory::desc data_desc = dst_desc.get_desc();
  auto alg = desc.get_algorithm();
  if ((alg == ::dnnl::algorithm::eltwise_clip) ||
      (alg == ::dnnl::algorithm::eltwise_linear) ||
      (alg == ::dnnl::algorithm::eltwise_swish)) {
    data_desc = src_desc.get_desc();
  }
  auto primitive_args = create_primitive_args_or_get<::dnnl::eltwise_backward>(
      alg, diff_src_desc.get_desc(), diff_dst_desc.get_desc(), data_desc,
      desc.get_alpha(), desc.get_beta(),
      create_primitive_desc<::dnnl::eltwise_forward>(
          ::dnnl::prop_kind::forward, alg, src_desc.get_desc(),
          dst_desc.get_desc(), desc.get_alpha(), desc.get_beta()));

  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST,
             diff_dst_desc.get_desc(), diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SRC,
             diff_src_desc.get_desc(), diff_src);

  return exit_primitive(execute_primitive<::dnnl::eltwise_backward>(
      primitive_args,
      {{alpha, beta, DNNL_ARG_DIFF_SRC, diff_src_desc, diff_src}}));
}

inline
sycl::event engine_ext::async_pooling_forward(pooling_desc &desc, float alpha,
                                        const memory_desc_ext &src_desc,
                                        void *src, float beta,
                                        const memory_desc_ext &dst_desc,
                                        void *dst, ::dnnl::memory *workspace) {
  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }
  enter_primitive(2 * dst_desc.get_size());
  int pooling_dim = desc.get_stride().size();
  std::vector<int64_t> dilation(pooling_dim, 0);
  auto primitive_args =
      create_primitive_args_or_get<::dnnl::pooling_forward>(
          ::dnnl::prop_kind::forward_training, desc.get_algorithm(),
          src_desc.get_desc(), dst_desc.get_desc(), desc.get_stride(),
          desc.get_kernel(), dilation, desc.get_padding(), desc.get_padding());
  auto pd = get_primitive_desc<::dnnl::pooling_forward>(
      primitive_args.second.primitive);
  ::dnnl::memory ws_mem(pd.workspace_desc(), *_eng);
  if (workspace) {
    *workspace = ws_mem;
  } else {
    insert_workspace(src, ws_mem);
  }
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_WORKSPACE, ws_mem);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);

  return exit_primitive(execute_primitive<::dnnl::pooling_forward>(
      primitive_args, {{alpha, beta, DNNL_ARG_DST, dst_desc, dst}}));
}

inline
sycl::event engine_ext::async_pooling_backward(
    pooling_desc &desc, float alpha, const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &diff_dst_desc, void *diff_dst,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src,
    ::dnnl::memory *workspace) {
  if (scale_parameter_preprocess({{alpha, beta, diff_src_desc, diff_src}})) {
    return sycl::event();
  }
  enter_primitive(2 * diff_src_desc.get_size());
  int pooling_dim = desc.get_stride().size();
  std::vector<int64_t> dilation(pooling_dim, 0);
  auto primitive_args = create_primitive_args_or_get<::dnnl::pooling_backward>(
      desc.get_algorithm(), diff_src_desc.get_desc(), diff_dst_desc.get_desc(),
      desc.get_stride(), desc.get_kernel(), dilation, desc.get_padding(),
      desc.get_padding(),
      create_primitive_desc<::dnnl::pooling_forward>(
          ::dnnl::prop_kind::forward_training, desc.get_algorithm(),
          src_desc.get_desc(), dst_desc.get_desc(), desc.get_stride(),
          desc.get_kernel(), dilation, desc.get_padding(), desc.get_padding()));

  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST,
             diff_dst_desc.get_desc(), diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SRC,
             diff_src_desc.get_desc(), diff_src);

  if (workspace) {
    insert_arg(primitive_args.second.args, DNNL_ARG_WORKSPACE, *workspace);
  } else {
    insert_arg(primitive_args.second.args, DNNL_ARG_WORKSPACE,
               get_workspace(src));
  }

  return exit_primitive(execute_primitive<::dnnl::pooling_backward>(
      primitive_args,
      {{alpha, beta, DNNL_ARG_DIFF_SRC, diff_src_desc, diff_src}}));
}

inline
sycl::event engine_ext::async_softmax_forward(softmax_algorithm alg,
                                        softmax_mode mode, float alpha,
                                        const memory_desc_ext &src_desc,
                                        void *src, float beta,
                                        const memory_desc_ext &dst_desc,
                                        void *dst) {
  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }

  ::dnnl::memory::desc help_src_desc = src_desc.get_desc();
  ::dnnl::memory::desc help_dst_desc = dst_desc.get_desc();
  if (mode == softmax_mode::instance) {
    help_src_desc = compress_spatial_dimensions_to_channel(help_src_desc);
    help_dst_desc = compress_spatial_dimensions_to_channel(help_dst_desc);
  }
  enter_primitive(2 * help_dst_desc.get_size());

  ::dnnl::algorithm softmax_alg = ::dnnl::algorithm::softmax_accurate;
  if (alg == softmax_algorithm::log) {
    softmax_alg = ::dnnl::algorithm::softmax_log;
  }
  auto primitive_args = create_primitive_args_or_get<::dnnl::softmax_forward>(
      ::dnnl::prop_kind::forward, softmax_alg, help_src_desc, 
      help_dst_desc, 1);

  insert_arg(primitive_args.second.args, DNNL_ARG_DST, help_dst_desc, dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, help_src_desc, src);

  return exit_primitive(execute_primitive<::dnnl::softmax_forward>(
      primitive_args,
      {{alpha, beta, DNNL_ARG_DST, memory_desc_ext(help_dst_desc), dst}}));
}

inline
sycl::event engine_ext::async_softmax_backward(
    softmax_algorithm alg, softmax_mode mode, float alpha,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src) {
  if (scale_parameter_preprocess({{alpha, beta, diff_src_desc, diff_src}})) {
    return sycl::event();
  }
  ::dnnl::memory::desc help_diff_src_desc = diff_src_desc.get_desc();
  ::dnnl::memory::desc help_dst_desc = dst_desc.get_desc();
  ::dnnl::memory::desc help_diff_dst_desc = diff_dst_desc.get_desc();
  if (mode == softmax_mode::instance) {
    help_diff_src_desc =
        compress_spatial_dimensions_to_channel(help_diff_src_desc);
    help_dst_desc = compress_spatial_dimensions_to_channel(help_dst_desc);
    help_diff_dst_desc =
        compress_spatial_dimensions_to_channel(help_diff_dst_desc);
  }
  enter_primitive(2 * help_diff_src_desc.get_size());

  ::dnnl::algorithm softmax_alg = ::dnnl::algorithm::softmax_accurate;
  if (alg == softmax_algorithm::log) {
    softmax_alg = ::dnnl::algorithm::softmax_log;
  }

  auto primitive_args = create_primitive_args_or_get<::dnnl::softmax_backward>(
      softmax_alg, help_diff_src_desc, help_diff_dst_desc, help_dst_desc, 1,
      create_primitive_desc<::dnnl::softmax_forward>(
          ::dnnl::prop_kind::forward, softmax_alg, help_diff_src_desc,
          help_dst_desc, 1));
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, help_dst_desc, dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST, help_diff_dst_desc,
             diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SRC, help_diff_src_desc,
             diff_src);

  return exit_primitive(execute_primitive<::dnnl::softmax_backward>(
      primitive_args, {{alpha, beta, DNNL_ARG_DIFF_SRC,
                        memory_desc_ext(help_diff_src_desc), diff_src}}));
}

inline
sycl::event engine_ext::async_lrn_forward(lrn_desc &desc, float alpha,
                                    const memory_desc_ext &src_desc, void *src,
                                    float beta, const memory_desc_ext &dst_desc,
                                    void *dst, ::dnnl::memory *workspace) {

  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }
  enter_primitive(2 * dst_desc.get_size());
  auto primitive_args = create_primitive_args_or_get<::dnnl::lrn_forward>(
      ::dnnl::prop_kind::forward_training,
      ::dnnl::algorithm::lrn_across_channels, src_desc.get_desc(),
      dst_desc.get_desc(), desc.get_local_size(), desc.get_alpha(),
      desc.get_beta(), desc.get_k());
  auto pd =
      get_primitive_desc<::dnnl::lrn_forward>(primitive_args.second.primitive);
  ::dnnl::memory ws_mem(pd.workspace_desc(), *_eng);
  if (workspace) {
    *workspace = ws_mem;
  } else {
    insert_workspace(src, ws_mem);
  }

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_WORKSPACE, ws_mem);

  return exit_primitive(execute_primitive<::dnnl::lrn_forward>(
      primitive_args, {{alpha, beta, DNNL_ARG_DST, dst_desc, dst}}));
}

inline
sycl::event
engine_ext::async_lrn_backward(lrn_desc &desc, float alpha,
                         const memory_desc_ext &dst_desc, void *dst,
                         const memory_desc_ext &diff_dst_desc, void *diff_dst,
                         const memory_desc_ext &src_desc, void *src, float beta,
                         const memory_desc_ext &diff_src_desc, void *diff_src,
                         ::dnnl::memory *workspace) {

  if (scale_parameter_preprocess({{alpha, beta, diff_src_desc, diff_src}})) {
    return sycl::event();
  }
  enter_primitive(2 * diff_src_desc.get_size());
  auto primitive_args = create_primitive_args_or_get<::dnnl::lrn_backward>(
      ::dnnl::algorithm::lrn_across_channels, diff_src_desc.get_desc(),
      diff_dst_desc.get_desc(), src_desc.get_desc(), desc.get_local_size(),
      desc.get_alpha(), desc.get_beta(), desc.get_k(),
      create_primitive_desc<::dnnl::lrn_forward>(
          ::dnnl::prop_kind::forward_training,
          ::dnnl::algorithm::lrn_across_channels, src_desc.get_desc(),
          dst_desc.get_desc(), desc.get_local_size(), desc.get_alpha(),
          desc.get_beta(), desc.get_k()));

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST,
             diff_dst_desc.get_desc(), diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SRC,
             diff_src_desc.get_desc(), diff_src);

  if (workspace) {
    insert_arg(primitive_args.second.args, DNNL_ARG_WORKSPACE, *workspace);
  } else {
    insert_arg(primitive_args.second.args, DNNL_ARG_WORKSPACE,
               get_workspace(src));
  }

  return exit_primitive(execute_primitive<::dnnl::lrn_backward>(
      primitive_args,
      {{alpha, beta, DNNL_ARG_DIFF_SRC, diff_src_desc, diff_src}}));
}

inline
size_t engine_ext::get_batch_normalization_workspace_size(
    batch_normalization_ops ops, const memory_desc_ext &src_desc) {
  if(ops == batch_normalization_ops::none) {
    return 0;
  }
  return src_desc.get_size();
}

inline
sycl::event engine_ext::async_batch_normalization_forward_inference(
    batch_normalization_mode mode, float epsilon, float alpha,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &scale_bias_mean_var_desc, void *scale, void *bias,
    void *mean, void *var) {

  return batch_normalization_forward_internal(
      true, mode, epsilon, 0.f, alpha, src_desc, src, beta, dst_desc, dst,
      scale_bias_mean_var_desc, scale, bias, scale_bias_mean_var_desc, mean,
      var, nullptr, nullptr);
}

inline
sycl::event engine_ext::async_batch_normalization_forward_inference(
    batch_normalization_mode mode, batch_normalization_ops ops,
    activation_desc &adesc, float epsilon, float alpha,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &summand_desc, void *summand,
    const memory_desc_ext &scale_bias_desc, void *scale, void *bias,
    const memory_desc_ext &mean_var_desc, void *mean, void *var) {

  bool has_post_op = (ops != batch_normalization_ops::none);
  sycl::event e;
  enter_primitive(src_desc.get_size() + dst_desc.get_size() * 4 +
                  scale_bias_desc.get_size() * 2 +
                  mean_var_desc.get_size() * 5);
  if (has_post_op) {
    void *dst_cache = allocate(dst_desc);
    batch_normalization_forward_internal(
        true, mode, epsilon, 0.f, 1.f, src_desc, src, 0.f, dst_desc, dst_cache,
        scale_bias_desc, scale, bias, mean_var_desc, mean, var, nullptr,
        nullptr);

    if (ops == batch_normalization_ops::add_activation) {
      async_sum(1.f, summand_desc, summand, 1.f, dst_desc, dst_cache);
    }
    async_activation_forward(adesc, 1.f, dst_desc, dst_cache, 0.f, dst_desc,
                       dst_cache);
    return exit_primitive(
        async_sum(alpha, dst_desc, dst_cache, beta, dst_desc, dst));
  }
  return exit_primitive(batch_normalization_forward_internal(
      true, mode, epsilon, 0.f, alpha, src_desc, src, beta, dst_desc, dst,
      scale_bias_desc, scale, bias, mean_var_desc, mean, var, nullptr,
      nullptr));
}

inline
sycl::event engine_ext::async_batch_normalization_forward_training(
    batch_normalization_mode mode, float epsilon, float factor, float alpha,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &scale_bias_mean_var_desc, void *scale, void *bias,
    void *running_mean, void *running_var, void *saved_mean, void *saved_var) {
  return batch_normalization_forward_internal(
      false, mode, epsilon, factor, alpha, src_desc, src, beta, dst_desc, dst,
      scale_bias_mean_var_desc, scale, bias, scale_bias_mean_var_desc,
      saved_mean, saved_var, running_mean, running_var);
}

inline
sycl::event engine_ext::async_batch_normalization_forward_training(
    batch_normalization_mode mode, batch_normalization_ops ops,
    activation_desc &adesc, float epsilon, float factor, float alpha,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &summand_desc, void *summand,
    const memory_desc_ext &scale_bias_desc, void *scale, void *bias,
    const memory_desc_ext &mean_var_desc, void *running_mean, void *running_var,
    void *saved_mean, void *saved_var, size_t workspace_size,
    void *workspace) {
  enter_primitive(src_desc.get_size() + dst_desc.get_size() * 3 +
                  mean_var_desc.get_size() * 5 +
                  scale_bias_desc.get_size() * 2);
  bool has_post_op = (ops != batch_normalization_ops::none);
  sycl::event e;
  if (has_post_op) {
    if(workspace_size < dst_desc.get_desc().get_size()) {
      throw std::runtime_error("async_batch_normalization_forward_training_ex: "
        "no sufficient workspace.");
    }
    batch_normalization_forward_internal(
        false, mode, epsilon, factor, 1.f, src_desc, src, 0.f, dst_desc,
        workspace, scale_bias_desc, scale, bias, mean_var_desc,
        saved_mean, saved_var, running_mean, running_var);
    if (ops == batch_normalization_ops::add_activation) {
      async_sum(1.f, summand_desc, summand, 1.f, dst_desc,
          workspace);
    }
    return exit_primitive(async_activation_forward(
        adesc, alpha, dst_desc, workspace, beta, dst_desc, dst));
  }
  return exit_primitive(batch_normalization_forward_internal(
      false, mode, epsilon, factor, alpha, src_desc, src, beta, dst_desc, dst,
      scale_bias_desc, scale, bias, mean_var_desc, saved_mean, saved_var,
      running_mean, running_var));
}

inline
sycl::event engine_ext::async_batch_normalization_forward_training(
    batch_normalization_mode mode, batch_normalization_ops ops,
    activation_desc &adesc, float epsilon, float factor, float alpha,
    const memory_desc_ext &src_desc, void *src, float beta,
    const memory_desc_ext &dst_desc, void *dst,
    const memory_desc_ext &summand_desc, void *summand,
    const memory_desc_ext &scale_bias_mean_var_desc, void *scale, void *bias,
    void *running_mean, void *running_var, void *saved_mean, void *saved_var,
    size_t workspace_size, void *workspace) {
  return async_batch_normalization_forward_training(
      mode, ops, adesc, epsilon, factor, alpha, src_desc, src, beta, dst_desc,
      dst, summand_desc, summand, scale_bias_mean_var_desc, scale, bias,
      scale_bias_mean_var_desc, running_mean, running_var, saved_mean,
      saved_var, workspace_size, workspace);
}

inline
sycl::event engine_ext::async_batch_normalization_backward(
    batch_normalization_mode mode, float epsilon, float alpha_data,
    const memory_desc_ext &src_desc, void *src,
    const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta_data,
    const memory_desc_ext &diff_src_desc, void *diff_src, float alpha_param,
    const memory_desc_ext &diff_scale_bias_mean_var_desc, void *scale,
    float beta_param, void *diff_scale, void *diff_bias, void *saved_mean,
    void *saved_var) {

  return batch_normalization_backward_internal(
      mode, epsilon, alpha_data, src_desc, src, diff_dst_desc, diff_dst,
      beta_data, diff_src_desc, diff_src, alpha_param,
      diff_scale_bias_mean_var_desc, scale, nullptr, beta_param, diff_scale,
      diff_bias, diff_scale_bias_mean_var_desc, saved_mean, saved_var);
}

inline
sycl::event engine_ext::async_batch_normalization_backward(
    batch_normalization_mode mode, batch_normalization_ops ops,
    activation_desc &adesc, float epsilon, float alpha_data,
    const memory_desc_ext &src_desc, void *src, const memory_desc_ext &dst_desc,
    void *dst, const memory_desc_ext &diff_dst_desc, void *diff_dst,
    float beta_data, const memory_desc_ext &diff_src_desc, void *diff_src,
    const memory_desc_ext &diff_summand_desc, void *diff_summand,
    float alpha_param, const memory_desc_ext &diff_scale_bias_desc, void *scale,
    void *bias, float beta_param, void *diff_scale, void *diff_bias,
    const memory_desc_ext &mean_var_desc, void *saved_mean, void *saved_var,
    size_t workspace_size, void *workspace) {
  std::vector<void *> caches;
  ::dnnl::memory::desc real_diff_dst_desc = diff_dst_desc.get_desc();
  void *real_diff_dst = diff_dst;

  if (ops != batch_normalization_ops::none &&
      workspace_size < dst_desc.get_desc().get_size()) {
    throw std::runtime_error("async_batch_normalization_backward_ex: "
                             "no sufficient workspace.");
  }
  enter_primitive(diff_scale_bias_desc.get_size() * 8 +
                  src_desc.get_size() * 3 + diff_dst_desc.get_size() * 5 +
                  diff_src_desc.get_size() + mean_var_desc.get_size() * 9 +
                  diff_summand_desc.get_size());
  if (ops == batch_normalization_ops::add_activation) {
    void *diff_summand_cache = allocate(diff_summand_desc);
    async_activation_backward(adesc, 1.f, dst_desc, dst, diff_dst_desc, diff_dst,
                        dst_desc, workspace, 0.f,
                        diff_summand_desc, diff_summand_cache);
    async_sum(alpha_data, diff_summand_desc, diff_summand_cache, beta_data,
        diff_summand_desc, diff_summand);
    real_diff_dst_desc = diff_summand_desc.get_desc();
    real_diff_dst = diff_summand_cache;
  } else if (ops == batch_normalization_ops::activation) {
    void *diff_dst_cache = allocate(diff_dst_desc);
    async_activation_backward(adesc, 1.f, dst_desc, dst, diff_dst_desc,
                        diff_dst, dst_desc, workspace,
                        0.f, diff_dst_desc, diff_dst_cache);
    real_diff_dst = diff_dst_cache;
  }

  return exit_primitive(batch_normalization_backward_internal(
      mode, epsilon, alpha_data, src_desc, src, real_diff_dst_desc,
      real_diff_dst, beta_data, diff_src_desc, diff_src, alpha_param,
      diff_scale_bias_desc, scale, bias, beta_param, diff_scale, diff_bias,
      mean_var_desc, saved_mean, saved_var));
}

inline
sycl::event engine_ext::async_batch_normalization_backward(
    batch_normalization_mode mode, batch_normalization_ops ops,
    activation_desc &adesc, float epsilon, float alpha_data,
    const memory_desc_ext &src_desc, void *src, const memory_desc_ext &dst_desc,
    void *dst, const memory_desc_ext &diff_dst_desc, void *diff_dst,
    float beta_data, const memory_desc_ext &diff_src_desc, void *diff_src,
    const memory_desc_ext &diff_summand_desc, void *diff_summand,
    float alpha_param, const memory_desc_ext &diff_scale_bias_mean_var_desc,
    void *scale, void *bias, float beta_param, void *diff_scale,
    void *diff_bias, void *saved_mean, void *saved_var,
    size_t workspace_size, void *workspace) {

  return async_batch_normalization_backward(
      mode, ops, adesc, epsilon, alpha_data, src_desc, src, dst_desc, dst,
      diff_dst_desc, diff_dst, beta_data, diff_src_desc, diff_src,
      diff_summand_desc, diff_summand, alpha_param,
      diff_scale_bias_mean_var_desc, scale, bias, beta_param, diff_scale,
      diff_bias, diff_scale_bias_mean_var_desc, saved_mean, saved_var,
      workspace_size, workspace);
}

inline
sycl::event
engine_ext::async_convolution_forward(convolution_desc &desc, ::dnnl::algorithm alg,
                                float alpha, const memory_desc_ext &src_desc,
                                void *src, const memory_desc_ext &weight_desc,
                                void *weight, float beta,
                                const memory_desc_ext &dst_desc, void *dst) {
  if (scale_parameter_preprocess({{alpha, beta, dst_desc, dst}})) {
    return sycl::event();
  }
  auto help_weight_desc =
      get_group_weight_desc(desc.get_group_count(), weight_desc);

  ::dnnl::primitive_attr attr;
  attr.set_fpmath_mode(desc.get_math_mode());

  auto origin_src_md = src_desc.get_desc();
  auto origin_dst_md = dst_desc.get_desc();
  auto origin_weight_md = help_weight_desc;
  auto src_md = transfer_memory_desc_to_format_tag_any(origin_src_md);
  auto dst_md = transfer_memory_desc_to_format_tag_any(origin_dst_md);
  auto weight_md = transfer_memory_desc_to_format_tag_any(origin_weight_md);

  auto primitive_args =
      create_primitive_args_or_get<::dnnl::convolution_forward>(
          ::dnnl::prop_kind::forward_training, alg, src_md, weight_md, dst_md,
          desc.get_stride(), desc.get_dilate(), desc.get_padding(),
          desc.get_padding(), attr);

  auto pd = get_primitive_desc<::dnnl::convolution_forward>(
      primitive_args.second.primitive);
  auto optimal_src_md = pd.src_desc();
  auto optimal_dst_md = pd.dst_desc();
  auto optimal_weight_md = pd.weights_desc();

  enter_primitive(
      optimal_src_md.get_size() * 3 + optimal_dst_md.get_size() * 5 +
      optimal_weight_md.get_size() * 3 + origin_dst_md.get_size() * 2);

  void *optimal_src = src, *optimal_dst = dst, *optimal_weight = weight;
  allocate_and_reorder_memory_to_optimal(origin_src_md, src, optimal_src_md,
                                         optimal_src);
  allocate_and_reorder_memory_to_optimal(origin_weight_md, weight,
                                         optimal_weight_md, optimal_weight);

  if (beta == 0.f) {
    if(origin_dst_md != optimal_dst_md) {
      optimal_dst = allocate(optimal_dst_md);
    }
  } else {
    allocate_and_reorder_memory_to_optimal(origin_dst_md, dst, optimal_dst_md,
                                           optimal_dst);
  }

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, optimal_src_md,
             optimal_src);
  insert_arg(primitive_args.second.args, DNNL_ARG_WEIGHTS, optimal_weight_md,
             optimal_weight);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, optimal_dst_md,
             optimal_dst);

  auto e = execute_primitive<::dnnl::convolution_forward>(
      primitive_args,
      {{alpha, beta, DNNL_ARG_DST, optimal_dst_md, optimal_dst}});

  if (origin_dst_md != optimal_dst_md) {
    e = async_reorder(1.f, optimal_dst_md, optimal_dst, 0.f, origin_dst_md,
                      dst);
  }
  return exit_primitive(e);
}

inline
sycl::event engine_ext::async_convolution_forward(
    convolution_desc &desc, ::dnnl::algorithm alg, activation_desc &adesc,
    float alpha_0, const memory_desc_ext &src_desc, void *src,
    const memory_desc_ext &weight_desc, void *weight, float alpha_1,
    const memory_desc_ext &summand_desc, void *summand,
    const memory_desc_ext &bias_desc, void *bias,
    const memory_desc_ext &dst_desc, void *dst) {

  int channel_num = bias_desc.get_element_num();
  auto help_weight_desc =
      get_group_weight_desc(desc.get_group_count(), weight_desc);
  ::dnnl::memory::desc help_bias_desc = {{channel_num},
                                         bias_desc.get_desc().get_data_type(),
                                         ::dnnl::memory::format_tag::a};
  auto origin_weight_md = help_weight_desc;
  auto origin_bias_md = help_bias_desc;
  auto origin_src_md = src_desc.get_desc();
  auto origin_dst_md = dst_desc.get_desc();
  auto src_md = transfer_memory_desc_to_format_tag_any(origin_src_md);
  auto dst_md = transfer_memory_desc_to_format_tag_any(origin_dst_md);
  auto weight_md = transfer_memory_desc_to_format_tag_any(origin_weight_md);
  auto bias_md = transfer_memory_desc_to_format_tag_any(origin_bias_md);

  ::dnnl::primitive_attr attr;
  attr.set_fpmath_mode(desc.get_math_mode());

  auto primitive_args =
      create_primitive_args_or_get<::dnnl::convolution_forward>(
          ::dnnl::prop_kind::forward_training, alg, src_md, weight_md, bias_md,
          dst_md, desc.get_stride(), desc.get_dilate(), desc.get_padding(),
          desc.get_padding(), attr);

  auto pd = get_primitive_desc<::dnnl::convolution_forward>(
      primitive_args.second.primitive);
  auto optimal_src_md = pd.src_desc();
  auto optimal_dst_md = pd.dst_desc();
  auto optimal_weight_md = pd.weights_desc();
  auto optimal_bias_md = pd.bias_desc();

  enter_primitive(optimal_src_md.get_size() + 3 * optimal_weight_md.get_size() +
                  optimal_bias_md.get_size() + 7 * optimal_dst_md.get_size() +
                  summand_desc.get_size());

  void *optimal_src = src, *optimal_dst = dst, *optimal_weight = weight,
       *optimal_bias = bias;
  allocate_and_reorder_memory_to_optimal(origin_src_md, src, optimal_src_md,
                                         optimal_src);
  allocate_and_reorder_memory_to_optimal(origin_weight_md, weight,
                                         optimal_weight_md, optimal_weight);
  allocate_and_reorder_memory_to_optimal(origin_bias_md, bias, optimal_bias_md,
                                         optimal_bias);
  if (origin_dst_md != optimal_dst_md) {
    optimal_dst = allocate(optimal_dst_md);
  }

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, optimal_src_md,
             optimal_src);
  insert_arg(primitive_args.second.args, DNNL_ARG_BIAS, optimal_bias_md,
             optimal_bias);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, optimal_dst_md,
             optimal_dst);

  void *cache = nullptr;
  if (alpha_0 != 1.f) {
    cache = allocate(optimal_weight_md);
    _q->memcpy(cache, optimal_weight, optimal_weight_md.get_size());
    async_scale(alpha_0, optimal_weight_md, cache);
    insert_arg(primitive_args.second.args, DNNL_ARG_WEIGHTS, optimal_weight_md,
               cache);
    execute_primitive<::dnnl::convolution_forward>(
        primitive_args,
        {{1.f, 0.f, DNNL_ARG_DST, optimal_dst_md, optimal_dst}});
  } else {
    insert_arg(primitive_args.second.args, DNNL_ARG_WEIGHTS, optimal_weight_md,
               optimal_weight);
    execute_primitive<::dnnl::convolution_forward>(
        primitive_args,
        {{1.f, 0.f, DNNL_ARG_DST, optimal_dst_md, optimal_dst}});
  }
  if (origin_dst_md != optimal_dst_md) {
    async_reorder(1.f, optimal_dst_md, optimal_dst, 0.f, origin_dst_md, dst);
  }
  async_sum(alpha_1, summand_desc, summand, 1.f, dst_desc, dst);
  return exit_primitive(
      async_activation_forward(adesc, 1.f, dst_desc, dst, 0.f, dst_desc, dst));
}

inline
sycl::event engine_ext::async_convolution_backward_data(
    convolution_desc &desc, ::dnnl::algorithm alg, float alpha,
    const memory_desc_ext &weight_desc, void *weight,
    const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta,
    const memory_desc_ext &diff_src_desc, void *diff_src) {

  if (scale_parameter_preprocess({{alpha, beta, diff_dst_desc, diff_dst}})) {
    return sycl::event();
  }

  auto help_weight_desc =
      get_group_weight_desc(desc.get_group_count(), weight_desc);

  auto origin_weight_md = help_weight_desc;
  auto origin_diff_src_md = diff_src_desc.get_desc();
  auto origin_diff_dst_md = diff_dst_desc.get_desc();
  auto diff_src_md = transfer_memory_desc_to_format_tag_any(origin_diff_src_md);
  auto diff_dst_md = transfer_memory_desc_to_format_tag_any(origin_diff_dst_md);
  auto weight_md = transfer_memory_desc_to_format_tag_any(origin_weight_md);

  ::dnnl::primitive_attr attr;
  attr.set_fpmath_mode(desc.get_math_mode());

  auto forward_primitive = create_primitive_desc<::dnnl::convolution_forward>(
      ::dnnl::prop_kind::forward_training, ::dnnl::algorithm::convolution_auto,
      diff_src_md, weight_md, diff_dst_md, desc.get_stride(), desc.get_dilate(),
      desc.get_padding(), desc.get_padding(), attr);

  auto primitive_args =
      create_primitive_args_or_get<::dnnl::convolution_backward_data>(
          ::dnnl::algorithm::convolution_auto, diff_src_md, weight_md,
          diff_dst_md, desc.get_stride(), desc.get_dilate(), desc.get_padding(),
          desc.get_padding(), forward_primitive, attr);

  auto pd = get_primitive_desc<::dnnl::convolution_backward_data>(
      primitive_args.second.primitive);
  auto optimal_diff_src_md = pd.diff_src_desc();
  auto optimal_diff_dst_md = pd.diff_dst_desc();
  auto optimal_weight_md = pd.weights_desc();

  enter_primitive(5 * optimal_diff_src_md.get_size() +
                  optimal_diff_dst_md.get_size() +
                  optimal_weight_md.get_size());

  void *optimal_diff_src = diff_src, *optimal_diff_dst = diff_dst,
       *optimal_weight = weight;
  allocate_and_reorder_memory_to_optimal(origin_diff_dst_md, diff_dst,
                                         optimal_diff_dst_md, optimal_diff_dst);
  allocate_and_reorder_memory_to_optimal(origin_weight_md, weight,
                                         optimal_weight_md, optimal_weight);
  if (beta == 0.f) {
    if (origin_diff_src_md != optimal_diff_src_md) {
      optimal_diff_src = allocate(optimal_diff_src_md);
    }
  } else {
    allocate_and_reorder_memory_to_optimal(
        origin_diff_src_md, diff_src, optimal_diff_src_md, optimal_diff_src);
  }

  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST, optimal_diff_dst_md,
             optimal_diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_WEIGHTS, optimal_weight_md,
             optimal_weight);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_SRC, optimal_diff_src_md,
             optimal_diff_src);

  auto e = execute_primitive<::dnnl::convolution_backward_data>(
      primitive_args,
      {{alpha, beta, DNNL_ARG_DIFF_SRC, optimal_diff_src_md, optimal_diff_src}});

  if (origin_diff_src_md != optimal_diff_src_md) {
    e = async_reorder(1.f, optimal_diff_src_md, optimal_diff_src, 0.f,
                      origin_diff_src_md, diff_src);
  }
  return exit_primitive(e);
}

inline
sycl::event engine_ext::async_convolution_backward_weight(
    convolution_desc &desc, ::dnnl::algorithm alg, float alpha,
    const memory_desc_ext &src_desc, void *src,
    const memory_desc_ext &diff_dst_desc, void *diff_dst, float beta,
    const memory_desc_ext &diff_weight_desc, void *diff_weight) {

  if (scale_parameter_preprocess(
          {{alpha, beta, diff_weight_desc, diff_weight}})) {
    return sycl::event();
  }

  auto help_diff_weight_desc =
      get_group_weight_desc(desc.get_group_count(), diff_weight_desc);

  ::dnnl::primitive_attr attr;
  attr.set_fpmath_mode(desc.get_math_mode());

  auto origin_diff_weight_md = help_diff_weight_desc;
  auto origin_src_md = src_desc.get_desc();
  auto origin_diff_dst_md = diff_dst_desc.get_desc();
  auto src_md = transfer_memory_desc_to_format_tag_any(origin_src_md);
  auto diff_dst_md = transfer_memory_desc_to_format_tag_any(origin_diff_dst_md);
  auto diff_weight_md =
      transfer_memory_desc_to_format_tag_any(origin_diff_weight_md);

  auto forward_primitive = create_primitive_desc<::dnnl::convolution_forward>(
      ::dnnl::prop_kind::forward_training, ::dnnl::algorithm::convolution_auto,
      src_md, diff_weight_md, diff_dst_md, desc.get_stride(), desc.get_dilate(),
      desc.get_padding(), desc.get_padding(), attr);

  auto primitive_args =
      create_primitive_args_or_get<::dnnl::convolution_backward_weights>(
          ::dnnl::algorithm::convolution_auto, src_md, diff_weight_md,
          diff_dst_md, desc.get_stride(), desc.get_dilate(), desc.get_padding(),
          desc.get_padding(), forward_primitive, attr);

  auto pd = get_primitive_desc<::dnnl::convolution_backward_weights>(
      primitive_args.second.primitive);
  auto optimal_src_md = pd.src_desc();
  auto optimal_diff_dst_md = pd.diff_dst_desc();
  auto optimal_diff_weight_md = pd.diff_weights_desc();

  enter_primitive(optimal_diff_weight_md.get_size() * 5 +
                  optimal_diff_dst_md.get_size() + optimal_src_md.get_size());

  void *optimal_src = src, *optimal_diff_dst = diff_dst,
       *optimal_diff_weight = diff_weight;
  allocate_and_reorder_memory_to_optimal(origin_diff_dst_md, diff_dst,
                                         optimal_diff_dst_md, optimal_diff_dst);
  allocate_and_reorder_memory_to_optimal(origin_src_md, src, optimal_src_md,
                                         optimal_src);
  if (beta == 0.f) {
    if (origin_diff_weight_md != optimal_diff_weight_md) {
      optimal_diff_weight = allocate(optimal_diff_weight_md);
    }
  } else {
    allocate_and_reorder_memory_to_optimal(origin_diff_weight_md, diff_weight,
                                           optimal_diff_weight_md,
                                           optimal_diff_weight);
  }

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC, optimal_src_md,
             optimal_src);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_DST, optimal_diff_dst_md,
             optimal_diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_DIFF_WEIGHTS,
             optimal_diff_weight_md, optimal_diff_weight);

  auto e = execute_primitive<::dnnl::convolution_backward_weights>(
      primitive_args, {{alpha, beta, DNNL_ARG_DIFF_WEIGHTS,
                        optimal_diff_weight_md, optimal_diff_weight}});

  if (origin_diff_weight_md != optimal_diff_weight_md) {
    e = async_reorder(1.f, optimal_diff_weight_md, optimal_diff_weight, 0.f,
                      origin_diff_weight_md, diff_weight);
  }
  return exit_primitive(e);
}

inline
sycl::event engine_ext::async_convolution_backward_bias(
    float alpha, const memory_desc_ext &diff_dst_desc, void *diff_dst,
    float beta, const memory_desc_ext &diff_bias_desc, void *diff_bias) {
  return async_reduction(reduction_op::sum, alpha, diff_dst_desc, diff_dst, beta,
                   diff_bias_desc, diff_bias);
}

inline
void engine_ext::rnn_get_weight_space_size(const rnn_desc &desc,
                                           size_t *weight_space_size) {
  *weight_space_size = 0;
  rnn_forward_internal(desc, ::dnnl::prop_kind::forward_inference,
                       memory_desc_ext(), nullptr, memory_desc_ext(), nullptr,
                       memory_desc_ext(), nullptr, nullptr, memory_desc_ext(),
                       nullptr, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, true,
                       weight_space_size, nullptr, nullptr);
  return;
}

inline
void engine_ext::rnn_get_scratchpad_workspace_size(
    const rnn_desc &desc, ::dnnl::prop_kind kind,
    const memory_desc_ext &src_desc, size_t *scratchpad_size,
    size_t *workspace_size) {
  *workspace_size = 0;
  *scratchpad_size = 0;
  rnn_forward_internal(desc, kind, src_desc, nullptr, memory_desc_ext(),
                       nullptr, memory_desc_ext(), nullptr, nullptr,
                       memory_desc_ext(), nullptr, nullptr, 0, nullptr, 0,
                       nullptr, 0, nullptr, true, nullptr, workspace_size,
                       scratchpad_size);
  return;
}

inline
sycl::event engine_ext::async_rnn_forward(
    const rnn_desc &desc, ::dnnl::prop_kind kind,
    const memory_desc_ext &src_desc, void *src, const memory_desc_ext &dst_desc,
    void *dst, const memory_desc_ext &iter_desc, void *src_iter, void *dst_iter,
    const memory_desc_ext &iter_c_desc, void *src_iter_c, void *dst_iter_c,
    size_t weight_size, void *weight, size_t scratchpad_size, void *scratchpad,
    size_t workspace_size, void *workspace) {

  return rnn_forward_internal(
      desc, kind, src_desc, src, dst_desc, dst, iter_desc, src_iter, dst_iter,
      iter_c_desc, src_iter_c, dst_iter_c, weight_size, weight, workspace_size,
      workspace, scratchpad_size, scratchpad, false, nullptr, nullptr,
      nullptr);
}

inline
sycl::event engine_ext::async_rnn_backward(
    const rnn_desc &desc, const memory_desc_ext &dst_desc, void *dst,
    void *diff_dst, const memory_desc_ext &src_desc, void *src, void *diff_src,
    const memory_desc_ext &iter_desc, void *src_iter, void *diff_dst_iter,
    void *diff_src_iter, const memory_desc_ext &iter_c_desc, void *src_iter_c,
    void *diff_dst_iter_c, void *diff_src_iter_c, size_t weight_size,
    void *weight, void *diff_weight, size_t scratchpad_size, void *scratchpad,
    size_t workspace_size, void *workspace) {
  ::dnnl::memory::data_type src_dt;
  ::dnnl::memory::format_tag src_format_tag;
  rnn_mode mode;
  rnn_memory_format_tag format_tag;
  rnn_bias_mode bias_mode;
  rnn_direction direction;
  dpct::library_data_t dt;
  int direction_num = 1, input_size = 0, hidden_size = 0, projection_size = 0,
      layer_size = 0, gate_num = 1, output_size = 0, data_type_size = 0,
      seq_length = 1, batch_size = 1;
  void *last_layer_cache = nullptr;
  void *hidden_layer_cache = nullptr;
  sycl::event e;
  enter_primitive(src_desc.get_size() * 2);
  std::vector<int> offset(9, 0);
  std::vector<void *> data = {
      src,
      dst,
      (uint8_t *)src_iter + iter_desc.get_size(),
      nullptr,
      (uint8_t *)src_iter_c + iter_c_desc.get_size(),
      nullptr,
      (uint8_t *)weight + weight_size,
      (uint8_t *)workspace + workspace_size,
      diff_src,
      diff_dst,
      (uint8_t *)diff_src_iter + iter_desc.get_size(),
      (uint8_t *)diff_dst_iter + iter_desc.get_size(),
      (uint8_t *)diff_src_iter_c + iter_c_desc.get_size(),
      (uint8_t *)diff_dst_iter_c + iter_c_desc.get_size(),
      (uint8_t *)diff_weight + weight_size,
      scratchpad};

  desc.get(&mode, &bias_mode, &direction, &dt, &input_size, &hidden_size,
           &projection_size, &layer_size);

  get_rnn_configuration(src_desc.get_desc(), direction, mode, dt, hidden_size,
                        &src_dt, &src_format_tag, &projection_size,
                        &output_size, &seq_length, &batch_size, &direction_num,
                        &gate_num);

  if (direction == rnn_direction::bidirectional) {
    if (layer_size > 1) {
      last_layer_cache = allocate(src_desc);
      hidden_layer_cache = allocate(src_desc);
      data[8] = last_layer_cache;
    }
    e = execute_rnn_backward_primitive(
        mode, ::dnnl::rnn_direction::bidirectional_concat, bias_mode, src_dt,
        src_format_tag, seq_length, batch_size, output_size, 2 * output_size, 1,
        direction_num, hidden_size, gate_num, projection_size, data, offset, 1);
    if (layer_size > 1) {
      data[8] = hidden_layer_cache;
      data[9] = last_layer_cache;
      e = execute_rnn_backward_primitive(
          mode, ::dnnl::rnn_direction::bidirectional_sum, bias_mode, src_dt,
          src_format_tag, seq_length, batch_size, output_size, output_size, 1,
          direction_num, hidden_size, gate_num, projection_size, data, offset,
          layer_size - 1);
      _q->memcpy(diff_src,
                 ((layer_size - 1) % 2 == 0) ? last_layer_cache
                                             : hidden_layer_cache,
                 src_desc.get_size());
    }
  } else {
    e = execute_rnn_backward_primitive(
        mode, ::dnnl::rnn_direction::unidirectional_left2right, bias_mode,
        src_dt, src_format_tag, seq_length, batch_size, output_size,
        output_size, layer_size, direction_num, hidden_size, gate_num,
        projection_size, data, offset, 1);
  }

  return exit_primitive(e);
}

inline
size_t engine_ext::get_dropout_state_size(){
#ifndef __INTEL_MKL__
  throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                           "Interfaces Project does not support this API.");
#else
  auto r = get_internal_resource(_q);
  if(r->random_engine_state_size == -1){
    auto rand_engine = rng_engine_t(*_q, 0);
    r->random_engine_state_size =
        oneapi::mkl::rng::get_state_size(rand_engine);
  }
  return r->random_engine_state_size;
#endif
}

inline size_t
engine_ext::get_dropout_workspace_size(const memory_desc_ext &src_desc) {
  return src_desc.get_size();
}

inline
sycl::event engine_ext::async_dropout_forward(dropout_desc &desc,
                                              const memory_desc_ext &src_desc,
                                              void *src,
                                              const memory_desc_ext &dst_desc,
                                              void *dst, void *workspace,
                                              size_t workspace_size) {
  if (workspace_size < src_desc.get_size()) {
    throw std::runtime_error("async_dropout_forward: no sufficient workspace.");
  }
  enter_primitive(src_desc.get_size() * 2 + dst_desc.get_size() * 2);
  float p = desc.get_probability();
  if (p == 1.f) {
    return _q->memset(dst, 0, dst_desc.get_size());
  } else if (p == 0.f) {
    return async_reorder(1.f, src_desc, src, 0.f, dst_desc, dst);
  }

  float scale_factor = 1.f / (1.f - p);
  void *cache = workspace;

  memory_desc_ext rng_data_desc(
      ::dnnl::memory::desc(src_desc.get_dims(), ::dnnl::memory::data_type::s32,
                           src_desc.get_strides()));
  if (src_desc.get_desc().get_data_type() != ::dnnl::memory::data_type::s32) {
    cache = allocate(rng_data_desc);
  }

  desc.generate(_q, get_dropout_state_size(), rng_data_desc.get_element_num(),
                (std::int32_t *)cache);

  if (cache == workspace) {
    async_scale(scale_factor, src_desc, workspace);
  } else {
    async_reorder(scale_factor, rng_data_desc, cache, 0.f, src_desc, workspace);
  }

  auto primitive_args = create_primitive_args_or_get<::dnnl::binary>(
      ::dnnl::algorithm::binary_mul, src_desc.get_desc(), src_desc.get_desc(),
      dst_desc.get_desc());

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC_0, src_desc.get_desc(),
             src);
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC_1, src_desc.get_desc(),
             workspace);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, dst_desc.get_desc(),
             dst);

  return exit_primitive(execute_primitive<::dnnl::binary>(primitive_args));
}

inline
sycl::event engine_ext::async_dropout_backward(
    dropout_desc &desc, const memory_desc_ext &diff_dst_desc,
    void *diff_dst, const memory_desc_ext &diff_src_desc, void *diff_src,
    void *workspace, size_t workspace_size) {
  enter_primitive(2 * diff_src_desc.get_size());
  float p = desc.get_probability();
  if (p == 1.f) {
    return _q->memset(diff_src, 0, diff_src_desc.get_size());
  } else if (p == 0.f) {
    return async_reorder(1.f, diff_dst_desc, diff_dst, 0.f, diff_src_desc,
                         diff_src);
  }

  auto primitive_args = create_primitive_args_or_get<::dnnl::binary>(
      ::dnnl::algorithm::binary_mul, diff_dst_desc.get_desc(),
      diff_dst_desc.get_desc(), diff_src_desc.get_desc());

  insert_arg(primitive_args.second.args, DNNL_ARG_SRC_0,
             diff_dst_desc.get_desc(), diff_dst);
  insert_arg(primitive_args.second.args, DNNL_ARG_SRC_1,
             diff_dst_desc.get_desc(), workspace);
  insert_arg(primitive_args.second.args, DNNL_ARG_DST, diff_src_desc.get_desc(),
             diff_src);

  return exit_primitive(execute_primitive<::dnnl::binary>(primitive_args));
}
} // namespace dnnl
} // namespace dpct

#endif // __DPCT_DNNL_UTILS_HPP__
