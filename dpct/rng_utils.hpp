//==---- rng_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_RNG_UTILS_HPP__
#define __DPCT_RNG_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
#include <oneapi/mkl/rng/device.hpp>
#endif
#include "device.hpp"
#include "lib_common_utils.hpp"

namespace dpct {
namespace rng {
#ifdef __INTEL_MKL__ // The oneMKL Interfaces Project does not support this.
namespace device {
/// The random number generator on device.
/// \tparam engine_t The device random number generator engine. It can only be
/// oneapi::mkl::rng::device::mrg32k3a<1> or
/// oneapi::mkl::rng::device::mrg32k3a<4> or
/// oneapi::mkl::rng::device::philox4x32x10<1> or
/// oneapi::mkl::rng::device::philox4x32x10<4>.
template <typename engine_t> class rng_generator {
  static_assert(
      std::disjunction_v<
          std::is_same<engine_t, oneapi::mkl::rng::device::mrg32k3a<1>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::mrg32k3a<4>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::philox4x32x10<1>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::philox4x32x10<4>>,
          std::is_same<engine_t, oneapi::mkl::rng::device::mcg59<1>>>,
      "engine_t can only be oneapi::mkl::rng::device::mrg32k3a<1> or "
      "oneapi::mkl::rng::device::mrg32k3a<4> or "
      "oneapi::mkl::rng::device::philox4x32x10<1> or "
      "oneapi::mkl::rng::device::philox4x32x10<4> or "
      "oneapi::mkl::rng::device::mcg59<1>.");
  static constexpr bool _is_engine_vec_size_one = std::disjunction_v<
      std::is_same<engine_t, oneapi::mkl::rng::device::mrg32k3a<1>>,
      std::is_same<engine_t, oneapi::mkl::rng::device::philox4x32x10<1>>,
      std::is_same<engine_t, oneapi::mkl::rng::device::mcg59<1>>>;
  static constexpr std::uint64_t default_seed = 0;
  oneapi::mkl::rng::device::bits<std::uint32_t> _distr_bits;
  oneapi::mkl::rng::device::uniform_bits<std::uint32_t> _distr_uniform_bits;
  oneapi::mkl::rng::device::gaussian<float> _distr_gaussian_float;
  oneapi::mkl::rng::device::gaussian<double> _distr_gaussian_double;
  oneapi::mkl::rng::device::lognormal<float> _distr_lognormal_float;
  oneapi::mkl::rng::device::lognormal<double> _distr_lognormal_double;
  oneapi::mkl::rng::device::poisson<std::uint32_t> _distr_poisson;
  oneapi::mkl::rng::device::uniform<float> _distr_uniform_float;
  oneapi::mkl::rng::device::uniform<double> _distr_uniform_double;
  engine_t _engine;

public:
  /// Default constructor of rng_generator
  rng_generator() { _engine = engine_t(default_seed); }
  /// Constructor of rng_generator if engine type is not mcg59
  /// \param [in] seed The seed to initialize the engine state.
  /// \param [in] num_to_skip Set the number of elements need to be skipped.
  /// The number is calculated as: num_to_skip[0] + num_to_skip[1] * 2^64 +
  /// num_to_skip[2] * 2^128 + ... + num_to_skip[n-1] * 2^(64*(n-1))
  template <typename T = engine_t,
            typename std::enable_if<!std::is_same_v<
                T, oneapi::mkl::rng::device::mcg59<1>>>::type * = nullptr>
  rng_generator(std::uint64_t seed,
                std::initializer_list<std::uint64_t> num_to_skip) {
    _engine = engine_t(seed, num_to_skip);
  }
  /// Constructor of rng_generator if engine type is mcg59
  /// \param [in] seed The seed to initialize the engine state.
  /// \param [in] num_to_skip Set the number of elements need to be skipped.
  template <typename T = engine_t,
            typename std::enable_if<std::is_same_v<
                T, oneapi::mkl::rng::device::mcg59<1>>>::type * = nullptr>
  rng_generator(std::uint64_t seed, std::uint64_t num_to_skip) {
    _engine = engine_t(seed, num_to_skip);
  }

  /// Generate random number(s) obeys distribution \tparam distr_t.
  /// \tparam T The distribution of the random number. It can only be
  /// oneapi::mkl::rng::device::bits<std::uint32_t>,
  /// oneapi::mkl::rng::device::uniform_bits<std::uint32_t>,
  /// oneapi::mkl::rng::device::gaussian<float>,
  /// oneapi::mkl::rng::device::gaussian<double>,
  /// oneapi::mkl::rng::device::lognormal<float>,
  /// oneapi::mkl::rng::device::lognormal<double>,
  /// oneapi::mkl::rng::device::poisson<std::uint32_t>,
  /// oneapi::mkl::rng::device::uniform<float> or
  /// oneapi::mkl::rng::device::uniform<double>
  /// \tparam vec_size The length of the return vector. It can only be 1, 2
  /// or 4.
  /// \param distr_params The parameter(s) for lognormal or poisson
  /// distribution.
  /// \return The vector of the random number(s).
  template <typename distr_t, int vec_size, class... distr_params_t>
  auto generate(distr_params_t... distr_params) {
    static_assert(vec_size == 1 || vec_size == 2 || vec_size == 4,
                  "vec_size is not supported.");
    static_assert(
        std::disjunction_v<
            std::is_same<distr_t,
                         oneapi::mkl::rng::device::bits<std::uint32_t>>,
            std::is_same<distr_t,
                         oneapi::mkl::rng::device::uniform_bits<std::uint32_t>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::gaussian<float>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::gaussian<double>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::lognormal<float>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::lognormal<double>>,
            std::is_same<distr_t,
                         oneapi::mkl::rng::device::poisson<std::uint32_t>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::uniform<float>>,
            std::is_same<distr_t, oneapi::mkl::rng::device::uniform<double>>>,
        "distribution is not supported.");

    if constexpr (std::is_same_v<
                      distr_t, oneapi::mkl::rng::device::bits<std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_bits);
    }
    if constexpr (std::is_same_v<
                      distr_t,
                      oneapi::mkl::rng::device::uniform_bits<std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_uniform_bits);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::gaussian<float>>) {
      return generate_vec<vec_size>(_distr_gaussian_float);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::gaussian<double>>) {
      return generate_vec<vec_size>(_distr_gaussian_double);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::lognormal<float>>) {
      return generate_vec<vec_size>(_distr_lognormal_float, distr_params...,
                                    0.0f, 1.0f);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::lognormal<double>>) {
      return generate_vec<vec_size>(_distr_lognormal_double, distr_params...,
                                    0.0, 1.0);
    }
    if constexpr (std::is_same_v<distr_t, oneapi::mkl::rng::device::poisson<
                                              std::uint32_t>>) {
      return generate_vec<vec_size>(_distr_poisson, distr_params...);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::uniform<float>>) {
      return generate_vec<vec_size>(_distr_uniform_float);
    }
    if constexpr (std::is_same_v<distr_t,
                                 oneapi::mkl::rng::device::uniform<double>>) {
      return generate_vec<vec_size>(_distr_uniform_double);
    }
  }

  /// Get the random number generator engine.
  /// \return The reference of the internal random number generator engine.
  engine_t &get_engine() { return _engine; }

private:
  template <int vec_size, typename distr_t, class... distr_params_t>
  auto generate_vec(distr_t &distr, distr_params_t... distr_params) {
    if constexpr (sizeof...(distr_params_t)) {
      typename distr_t::param_type pt(distr_params...);
      distr.param(pt);
    }
    if constexpr (vec_size == 4) {
      if constexpr (_is_engine_vec_size_one) {
        sycl::vec<typename distr_t::result_type, 4> res;
        res.x() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.y() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.z() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.w() = oneapi::mkl::rng::device::generate(distr, _engine);
        return res;
      } else {
        return oneapi::mkl::rng::device::generate(distr, _engine);
      }
    } else if constexpr (vec_size == 1) {
      if constexpr (_is_engine_vec_size_one) {
        return oneapi::mkl::rng::device::generate(distr, _engine);
      } else {
        return oneapi::mkl::rng::device::generate_single(distr, _engine);
      }
    } else if constexpr (vec_size == 2) {
      if constexpr (_is_engine_vec_size_one) {
        sycl::vec<typename distr_t::result_type, 2> res;
        res.x() = oneapi::mkl::rng::device::generate(distr, _engine);
        res.y() = oneapi::mkl::rng::device::generate(distr, _engine);
        return res;
      } else {
        sycl::vec<typename distr_t::result_type, 2> res;
        res.x() = oneapi::mkl::rng::device::generate_single(distr, _engine);
        res.y() = oneapi::mkl::rng::device::generate_single(distr, _engine);
        return res;
      }
    }
  }
};

} // namespace device
#endif

namespace host {
namespace detail {
class rng_generator_base {
public:
  /// Set the seed of host rng_generator.
  /// \param seed The engine seed.
  virtual void set_seed(const std::uint64_t seed) = 0;

  /// Set the dimensions of host rng_generator.
  /// \param dimensions The engine dimensions.
  virtual void set_dimensions(const std::uint32_t dimensions) = 0;

  /// Set the queue of host rng_generator.
  /// \param queue The engine queue.
  virtual void set_queue(sycl::queue *queue) = 0;

  /// Generate unsigned int random number(s) with 'uniform_bits' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform_bits(unsigned int *output,
                                            std::int64_t n) = 0;

  /// Generate unsigned long long random number(s) with 'uniform_bits'
  /// distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform_bits(unsigned long long *output,
                                            std::int64_t n) = 0;

  /// Generate float random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  virtual inline void generate_lognormal(float *output, std::int64_t n, float m,
                                         float s) = 0;

  /// Generate double random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  virtual inline void generate_lognormal(double *output, std::int64_t n,
                                         double m, double s) = 0;

  /// Generate float random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  virtual inline void generate_gaussian(float *output, std::int64_t n,
                                        float mean, float stddev) = 0;

  /// Generate double random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  virtual inline void generate_gaussian(double *output, std::int64_t n,
                                        double mean, double stddev) = 0;

  /// Generate unsigned int random number(s) with 'poisson' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param lambda Lambda for the Poisson distribution.
  virtual inline void generate_poisson(unsigned int *output, std::int64_t n,
                                       double lambda) = 0;

  /// Generate float random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform(float *output, std::int64_t n) = 0;

  /// Generate double random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  virtual inline void generate_uniform(double *output, std::int64_t n) = 0;

  /// Skip ahead several random number(s).
  /// \param num_to_skip The number of random numbers to be skipped.
  virtual void skip_ahead(const std::uint64_t num_to_skip) = 0;

  /// Set the direction numbers of host rng_generator. Only Sobol engine
  /// supports this method.
  /// \param direction_numbers The engine direction numbers.
  virtual void set_direction_numbers(
      const std::vector<std::uint32_t> &direction_numbers) = 0;

protected:
  sycl::queue *_queue{&dpct::get_default_queue()};
  std::uint64_t _seed{0};
  std::uint32_t _dimensions{1};
  std::vector<std::uint32_t> _direction_numbers;
};

/// The random number generator on host.
template <typename engine_t = oneapi::mkl::rng::philox4x32x10>
class rng_generator : public rng_generator_base {
public:
  /// Constructor of rng_generator.
  rng_generator() : _engine(create_engine(_queue, _seed, _dimensions)) {}

  /// Set the seed of host rng_generator.
  /// \param seed The engine seed.
  void set_seed(const std::uint64_t seed) {
    if (seed == _seed) {
      return;
    }
    _seed = seed;
    _engine = create_engine(_queue, _seed, _dimensions);
  }

  /// Set the dimensions of host rng_generator.
  /// \param dimensions The engine dimensions.
  void set_dimensions(const std::uint32_t dimensions) {
    if (dimensions == _dimensions) {
      return;
    }
    _dimensions = dimensions;
    _engine = create_engine(_queue, _seed, _dimensions);
  }

  /// Set the queue of host rng_generator.
  /// \param queue The engine queue.
  void set_queue(sycl::queue *queue) {
    if (queue == _queue) {
      return;
    }
    _queue = queue;
    _engine = create_engine(_queue, _seed, _dimensions);
  }

  /// Set the direction numbers of Sobol host rng_generator.
  /// \param direction_numbers The user-defined direction numbers.
  void
  set_direction_numbers(const std::vector<std::uint32_t> &direction_numbers) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
    if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::sobol>) {
      if (direction_numbers == _direction_numbers) {
        return;
      }
      _direction_numbers = direction_numbers;
      _engine = oneapi::mkl::rng::sobol(*_queue, _direction_numbers);
    } else {
      throw std::runtime_error("Only Sobol engine supports this method.");
    }
#endif
  }

  /// Generate unsigned int random number(s) with 'uniform_bits' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform_bits(unsigned int *output, std::int64_t n) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
    static_assert(sizeof(unsigned int) == sizeof(std::uint32_t));
    generate<oneapi::mkl::rng::uniform_bits<std::uint32_t>>(
        (std::uint32_t *)output, n);
#endif
  }

  /// Generate unsigned long long random number(s) with 'uniform_bits'
  /// distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform_bits(unsigned long long *output,
                                    std::int64_t n) {
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
    static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
    generate<oneapi::mkl::rng::uniform_bits<std::uint64_t>>(
        (std::uint64_t *)output, n);
#endif
  }

  /// Generate float random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  inline void generate_lognormal(float *output, std::int64_t n, float m,
                                 float s) {
    generate<oneapi::mkl::rng::lognormal<float>>(output, n, m, s);
  }

  /// Generate double random number(s) with 'lognormal' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param m Mean of associated normal distribution
  /// \param s Standard deviation of associated normal distribution.
  inline void generate_lognormal(double *output, std::int64_t n, double m,
                                 double s) {
    generate<oneapi::mkl::rng::lognormal<double>>(output, n, m, s);
  }

  /// Generate float random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  inline void generate_gaussian(float *output, std::int64_t n, float mean,
                                float stddev) {
    generate<oneapi::mkl::rng::gaussian<float>>(output, n, mean, stddev);
  }

  /// Generate double random number(s) with 'gaussian' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param mean Mean of normal distribution
  /// \param stddev Standard deviation of normal distribution.
  inline void generate_gaussian(double *output, std::int64_t n, double mean,
                                double stddev) {
    generate<oneapi::mkl::rng::gaussian<double>>(output, n, mean, stddev);
  }

  /// Generate unsigned int random number(s) with 'poisson' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  /// \param lambda Lambda for the Poisson distribution.
  inline void generate_poisson(unsigned int *output, std::int64_t n,
                               double lambda) {
    generate<oneapi::mkl::rng::poisson<unsigned int>>(output, n, lambda);
  }

  /// Generate float random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform(float *output, std::int64_t n) {
    generate<oneapi::mkl::rng::uniform<float>>(output, n);
  }

  /// Generate double random number(s) with 'uniform' distribution.
  /// \param output The pointer of the first random number.
  /// \param n The number of random numbers.
  inline void generate_uniform(double *output, std::int64_t n) {
    generate<oneapi::mkl::rng::uniform<double>>(output, n);
  }

  /// Skip ahead several random number(s).
  /// \param num_to_skip The number of random numbers to be skipped.
  void skip_ahead(const std::uint64_t num_to_skip) {
#ifndef __INTEL_MKL__
    oneapi::mkl::rng::skip_ahead(_engine, num_to_skip);
#else
    if constexpr (std::is_same_v<engine_t, oneapi::mkl::rng::mt2203>)
      throw std::runtime_error("no skip_ahead method of mt2203 engine.");
    else
      oneapi::mkl::rng::skip_ahead(_engine, num_to_skip);
#endif
  }

private:
  static inline engine_t create_engine(sycl::queue *queue,
                                       const std::uint64_t seed,
                                       const std::uint32_t dimensions) {
#ifdef __INTEL_MKL__
    return std::is_same_v<engine_t, oneapi::mkl::rng::sobol>
               ? engine_t(*queue, dimensions)
               : engine_t(*queue, seed);
#else
    return engine_t(*queue, seed);
#endif
  }

  template <typename distr_t, typename buffer_t, class... distr_params_t>
  void generate(buffer_t *output, const std::int64_t n,
                const distr_params_t... distr_params) {
    auto output_buf = dpct::detail::get_memory<buffer_t>(output);
    oneapi::mkl::rng::generate(distr_t(distr_params...), _engine, n,
                               output_buf);
  }
  engine_t _engine{};
};
} // namespace detail
} // namespace host

enum class random_engine_type {
  philox4x32x10,
  mrg32k3a,
  mt2203,
  mt19937,
  sobol,
  mcg59
};

typedef std::shared_ptr<rng::host::detail::rng_generator_base> host_rng_ptr;

/// Create a host random number generator.
/// \param type The random engine type.
/// \return The pointer of random number generator.
inline host_rng_ptr create_host_rng(const random_engine_type type) {
  switch (type) {
  case random_engine_type::philox4x32x10:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::philox4x32x10>>();
  case random_engine_type::mrg32k3a:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mrg32k3a>>();
#ifndef __INTEL_MKL__
    throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) "
                             "Interfaces Project does not support this API.");
#else
  case random_engine_type::mt2203:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mt2203>>();
  case random_engine_type::mt19937:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mt19937>>();
  case random_engine_type::sobol:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::sobol>>();
  case random_engine_type::mcg59:
    return std::make_shared<
        rng::host::detail::rng_generator<oneapi::mkl::rng::mcg59>>();
#endif
  }
}
} // namespace rng
} // namespace dpct

#endif // __DPCT_RNG_UTILS_HPP__
