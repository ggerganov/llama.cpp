#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Emulation of an analog in-memory computing accelerator (memristor / ReRAM crossbar arrays).
//
// The device is a weight-stationary ACCEL device, like BLAS: the weights stay in host memory and
// the device offloads GGML_OP_MUL_MAT operations whose first operand holds model weights. Before
// an operation is executed, the weights are "programmed" into the emulated array, i.e. they are
// snapshotted, snapped to a finite number of conductance levels and perturbed by the array's
// non-idealities (programming error, conductance drift, IR drop). The dot products themselves are
// evaluated by an embedded CPU backend (the emulated mixed-signal core), after which the array
// readout (input DAC quantization, ADC quantization, thermal noise) is applied to the results.
//
// The array is registered but inert by default. Enable it with the GGML_ANALOG_ENABLE environment
// variable or with ggml_backend_analog_set_enabled() before enabling the device, otherwise the
// device reports that it supports no operations and the CPU backend is used.
//
// All array parameters can be set with GGML_ANALOG_* environment variables, see docs/backend/ANALOG.md

// modeled device counters, accumulated process-wide
struct ggml_analog_stats {
    // array programming ("weight loading")
    int64_t  n_tensors;         // number of tensors programmed into arrays
    int64_t  n_cells;           // number of emulated cells
    double   bytes;             // bytes of emulated array storage
    double   time_program_ms;   // wall time spent programming the arrays

    // array operations
    int64_t  n_mvm;             // number of MUL_MAT operations offloaded to the device
    int64_t  n_macs;            // number of multiply-accumulate operations
    int64_t  n_column_tiles;    // number of crossbar column tiles (K dimension)
    int64_t  n_array_passes;    // modeled number of serialized array passes
    int64_t  n_adc_samples;     // number of ADC conversions
    double   time_compute_ms;   // wall time spent in the emulated MVM (incl. delegation)
    double   time_readout_ms;   // wall time spent in the readout stage

    // modeled hardware cost, see docs/backend/ANALOG.md for the model
    double   est_time_ms;       // modeled array execution time
    double   est_energy_mj;     // modeled energy
};

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_analog_reg(void);

GGML_BACKEND_API ggml_backend_t ggml_backend_analog_init(void);

GGML_BACKEND_API bool ggml_backend_is_analog(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_analog_set_n_threads(ggml_backend_t backend, int n_threads);

// enable/disable the emulation, i.e. whether the device reports support for GGML_OP_MUL_MAT
GGML_BACKEND_API void ggml_backend_analog_set_enabled(bool enabled);
GGML_BACKEND_API bool ggml_backend_analog_get_enabled(void);

// Re-read the parameters and the enable flag from the GGML_ANALOG_* variables, for callers that
// changed the environment after this backend was registered; set_enabled() stays the override
GGML_BACKEND_API void ggml_backend_analog_config_reload(void);

GGML_BACKEND_API void ggml_backend_analog_stats_reset(void);
GGML_BACKEND_API void ggml_backend_analog_stats_get(struct ggml_analog_stats * stats);
GGML_BACKEND_API void ggml_backend_analog_stats_print(void);

#ifdef  __cplusplus
}
#endif
