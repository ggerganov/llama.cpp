#pragma once

// Execution of the array kernels through the compute shaders of vulkan/.
//
// The shaders are the portable form of the three passes of the array model: the input DAC, the
// programming of a tile and the readout. Every shader has a line by line mirror in
// analog-shader.cpp, so the same model can be run on a machine without a Vulkan device, and the
// two can be compared against each other and against the CPU kernels. Which one runs is selected
// with GGML_ANALOG_DISPATCH (cpu | shader | vulkan).

#include "ggml.h"
#include "analog-array.h"

// what the shader kernels are executing with: "mirror", "mirror (no Vulkan device)", "vulkan: ..."
const char * ggml_analog_shader_status(void);

// whether the shaders were compiled into this build
bool ggml_analog_shader_available(void);

// the SPIR-V modules of the shaders, 0 when the build had no GLSL compiler
int ggml_analog_shader_spirv_count(void);
const char * ggml_analog_shader_spirv_name(int index);
const uint32_t * ggml_analog_shader_spirv_words(int index, size_t * n_words);

// prefer the Vulkan device over the mirror, ignored when no device can be initialized
void ggml_analog_shader_set_prefer_vulkan(bool prefer);

// the three passes of the array model, the same signature as the CPU kernels so that a dispatcher
// can pick either one
void ggml_analog_shader_program(
        const struct ggml_tensor * w,
              struct ggml_tensor * snap,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm);

void ggml_analog_shader_dac(
        const float * src,
              float * dst,
        int64_t k, int64_t ncols,
        const ggml_analog_config & cfg);

void ggml_analog_shader_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        int64_t k, int64_t m,
        int64_t n1, int64_t n2, int64_t n3,
        int64_t ne2, int64_t ne3,
        const ggml_analog_config & cfg);
