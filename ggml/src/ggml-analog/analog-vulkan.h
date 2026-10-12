#pragma once

// Vulkan dispatch of the compute shaders of vulkan/.
//
// The device is a plain compute device: the passes of the array model already arrived in host
// memory, so every call uploads its buffers, dispatches the kernel and downloads the result. That
// is the right shape for the emulation (the partial sums of the mixed-signal core are produced on
// the host), but it is not a fast path: the point of these kernels is that a real array controller
// would run exactly this code next to the tile, not that the emulation gets faster.
//
// The loader is resolved with dlopen, so a build with the Vulkan headers does not link against a
// Vulkan library and runs the CPU mirror of the shaders when no device can be initialized.

#include "ggml.h"
#include "analog-array.h"

#include <cstdint>

// whether a Vulkan device with the capabilities of the shaders was initialized, the device is
// created on the first call
bool ggml_analog_vulkan_available(void);

// "vulkan: <device name>", or the reason why the dispatch is not available
const char * ggml_analog_vulkan_status(void);

bool ggml_analog_vulkan_dac(
        const float * x,
              float * q,
        uint32_t k, uint32_t ncols,
        float step, float inv_step);

bool ggml_analog_vulkan_program(
        const float * w,
              float * snap,
        uint32_t k, uint32_t nrows,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm,
        float step, float inv_step);

bool ggml_analog_vulkan_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        uint32_t k, uint32_t m,
        uint32_t n1, uint32_t n2, uint32_t n3,
        uint32_t ne2, uint32_t ne3,
        float adc_sigma, float sigma, float adc_step, float adc_rstep);
