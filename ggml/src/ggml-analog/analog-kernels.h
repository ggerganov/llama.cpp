#pragma once

// Parallel kernels of the emulated analog array.
//
// The array model has three passes that touch every cell or every output of a matrix
// multiplication:
//
//   - programming:  weight tensor -> array snapshot (finite conductance levels, programming
//                   error, drift, IR drop)
//   - input DAC:    activations -> levels that the array can drive into the rows
//   - readout:      array partial sums -> ADC quantization + thermal noise
//
// All three are elementwise passes over the array, which is why they vectorize and parallelize
// without changing the result: the kernels below produce exactly the same values as the scalar
// reference implementations next to them, which is what tests/test-analog-kernels.cpp checks
// bit by bit. The reference implementations are part of the interface so that they can be used as
// the definition of the model, and so that a run can be repeated with the kernels disabled
// (GGML_ANALOG_KERNELS=0) to tell a kernel bug from a different model.

#include "ggml.h"
#include "analog-array.h"

#include <cstdint>

// "avx2+omp", "scalar+serial", ...
const char * ggml_analog_kernels_name(void);

// threads used by the kernels, 1 means that everything runs on the calling thread
void ggml_analog_kernels_set_n_threads(int n_threads);
int  ggml_analog_kernels_get_n_threads(void);

// the value of GGML_ANALOG_N_THREADS, 0 when it was not set
int  ggml_analog_kernels_get_env_n_threads(void);

// use the scalar reference implementations instead of the vectorized kernels
void ggml_analog_kernels_set_enabled(bool enabled);
bool ggml_analog_kernels_get_enabled(void);

// where the passes of the array model run: the CPU kernels of this file, the compute shaders of
// vulkan/ with their CPU mirror, or the shaders on a Vulkan device
enum ggml_analog_dispatch {
    GGML_ANALOG_DISPATCH_CPU    = 0,
    GGML_ANALOG_DISPATCH_SHADER = 1,
    GGML_ANALOG_DISPATCH_VULKAN = 2,
};

// selected with GGML_ANALOG_DISPATCH (cpu | shader | vulkan)
// re-read the GGML_ANALOG_* variables of this module (dispatch, threads, kernels on/off)
void ggml_analog_kernels_env_reload(void);

void ggml_analog_kernels_set_dispatch(enum ggml_analog_dispatch dispatch);
enum ggml_analog_dispatch ggml_analog_kernels_get_dispatch(void);
const char * ggml_analog_kernels_dispatch_name(void); // "cpu kernels", "shader mirror", "vulkan"

// row conversions used by the kernels and by the shader path
void ggml_analog_dequant_row(const struct ggml_tensor * w, const void * src, float * dst, int64_t k);
void ggml_analog_quant_row  (const struct ggml_tensor * w, void * dst, const float * src, int64_t k);

// table of unit-variance readout noise samples, shared by the kernels and the reference
#define GGML_ANALOG_NOISE_TABLE 8192
const float * ggml_analog_kernel_noise_table(void);

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

// quantize the activations of ncols columns of k elements to the levels of the input DAC
void ggml_analog_kernel_dac(
        const float * src,
              float * dst,
        int64_t k, int64_t ncols,
        const ggml_analog_config & cfg);

// readout of the array: `x` the quantized activations, `y` the partial sums of the core, overwritten
// by the ADC result; row_norm/row_inv_norm hold the L2 norm of every programmed row
void ggml_analog_kernel_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg);

// program a weight tensor into `snap` (same shape as `w`, possibly F16), filling row_norm and
// row_inv_norm with the L2 norm of every programmed row
void ggml_analog_kernel_program(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm);

// ---------------------------------------------------------------------------
// scalar reference implementations of the same three passes, single threaded
// ---------------------------------------------------------------------------

void ggml_analog_ref_dac(
        const float * src,
              float * dst,
        int64_t k, int64_t ncols,
        const ggml_analog_config & cfg);

void ggml_analog_ref_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg);

void ggml_analog_ref_program(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm);
