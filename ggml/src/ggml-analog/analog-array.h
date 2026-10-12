#pragma once

// Internal emulation of the analog crossbar array:
//
//   - programming: weight tensor -> array snapshot (finite levels, programming error, drift, IR drop)
//   - readout:     array partial sums -> ADC quantization + thermal noise
//
// See docs/backend/ANALOG.md for the device model and the meaning of the parameters.

#include "ggml.h"
#include "ggml-analog.h"

#include <cstdint>

struct ggml_analog_config {
    // cell precision
    int bits_w   = 4;    // conductance levels per cell (weights)
    int bits_x   = 8;    // levels of the input DAC
    int bits_adc = 8;    // levels of the readout ADC

    // non-idealities, all relative to the full scale of the affected quantity
    float sigma_prog  = 0.02f; // programming error of a cell
    float sigma_therm = 0.005f;// thermal / shot noise of a readout, per array pass
    float drift       = 0.00f; // conductance drift, applied at the end of programming
    float ir_drop     = 0.00f; // parasitic wire attenuation along the array columns
    float adc_sigma   = 4.00f; // readout range, in standard deviations of the partial sum

    // array geometry
    int tile_rows = 128; // rows (output channels) per crossbar tile
    int tile_cols = 128; // columns (reduction dimension) per crossbar tile

    // modeled device parameters
    float pj_per_mac     = 0.1f; // energy per MAC
    float mvm_latency_ns = 300;  // latency of one array pass

    uint64_t seed       = 42;
    bool     ideal      = false; // disable all non-idealities, used to validate the plumbing
    bool     verbose    = false;
    bool     all_matmul = false; // program all matrices, not only the weights
    int      n_threads = 1;     // threads used by the readout stage
};

struct ggml_analog_array;

// program a weight tensor into the array, returns nullptr if the tensor cannot be programmed
ggml_analog_array * ggml_analog_array_new(const struct ggml_tensor * w, const ggml_analog_config & cfg);

void ggml_analog_array_free(ggml_analog_array * arr);

// the programmed snapshot, used as the source of the emulated MVM
struct ggml_tensor * ggml_analog_array_weights(ggml_analog_array * arr);

int64_t ggml_analog_array_cells   (const ggml_analog_array * arr);
double  ggml_analog_array_program_ms(const ggml_analog_array * arr);

// apply the readout stage to dst, which holds sum_k w[i,k]*x[k,c] evaluated with the programmed weights
void ggml_analog_array_readout(
        const ggml_analog_array *  arr,
        const struct ggml_tensor * x,
              struct ggml_tensor * dst,
        const ggml_analog_config & cfg);

// quantize the activations to the levels of the input DAC, returns nullptr when the native
// quantization of the mixed-signal core is already equivalent (8 bit per 32-element block)
struct ggml_tensor * ggml_analog_quantize_input(const struct ggml_tensor * x, const ggml_analog_config & cfg);
void ggml_analog_quantize_input_free(struct ggml_tensor * q);

// modeled device cost of one emulated MVM
void ggml_analog_mvm_cost(
        const struct ggml_tensor * w,
        const struct ggml_tensor * dst,
        const ggml_analog_config & cfg,
              struct ggml_analog_stats * stats);

// the statistics are accumulated process-wide
void ggml_analog_stats_add_external  (const struct ggml_analog_stats & s);
void ggml_analog_stats_get_external  (      struct ggml_analog_stats * s);
void ggml_analog_stats_reset_external(      void);

void ggml_analog_config_load_from_env(ggml_analog_config & cfg);
void ggml_analog_config_print(const ggml_analog_config & cfg);
