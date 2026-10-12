# llama.cpp for analog in-memory computing accelerators

The analog backend (`GGML_ANALOG`) emulates an analog in-memory computing accelerator, i.e. an
array of non-volatile memory cells (memristor / ReRAM / PCM crossbars) that stores the model
weights and computes matrix-vector products by physical laws (Ohm's law and Kirchhoff's current
law) instead of by digital multiply-accumulate circuits.

It is a **research and evaluation tool**: it runs entirely on the CPU, it is not faster than the
CPU backend and it is not bit-accurate. Its purpose is to answer the questions that come before
silicon exists:

- what does a given cell precision / array size / noise level do to model quality?
- how much of the inference time is array time and how much is converter (DAC/ADC) time?
- what is the modeled energy per token for a given device?
- where do the non-idealities (programming error, drift, IR drop, read noise) hurt the most?

## Device model

The device is registered as an **ACCEL** device, the same kind of device as the BLAS backend:

| layer | what it models | where it runs |
| --- | --- | --- |
| array | weights programmed into crossbar cells | a snapshot of the weight tensor, perturbed by the non-idealities |
| mixed-signal core | the digital part of the accelerator, i.e. the MAC operations themselves | the CPU backend |
| readout | column ADCs, charge sharing, thermal and shot noise | a post-pass over the operation outputs |

Weights are *programmed* (i.e. snapped to a finite number of conductance levels and perturbed)
lazily, the first time an operation needs them, and then kept in the emulated array. Each
`GGML_OP_MUL_MAT` whose first operand is a model weight is evaluated as:

```
y = ADC( sum_k  W_programmed[i,k] * DAC(x[k]) )
```

Implementing this as an ACCEL device means that nothing in `src/` changes: llama.cpp already knows
how to schedule operations onto extra accelerator devices, and `--n-gpu-layers` / `-ngl` do not
need to be touched, because the weights stay in host memory.

Matrix multiplications whose first operand is not a weight (attention scores, `KQ`, `V*attn`, ...)
are evaluated by the mixed-signal core in exactly the same way the CPU backend would, because on a
real chip they run on the digital part of the accelerator, not on the arrays.

Because the arrays are programmed from the weights in host memory, the device deliberately does not
claim operations that llama.cpp only keeps enabled when they are assigned to the same device as the
layer weights (`GGML_OP_FLASH_ATTN_EXT` and the other fused layer operations). Claiming them would
silently disable Flash Attention.

## Build

```sh
cmake -B build -DGGML_ANALOG=ON
cmake --build build --config Release -j
```

The backend is off by default, and it is inert until it is enabled at runtime, so a build with
`-DGGML_ANALOG=ON` behaves exactly like a stock build unless the environment asks otherwise:

```sh
GGML_ANALOG_ENABLE=1  ./llama-cli     -m model.gguf -p "Hello"
GGML_ANALOG_ENABLE=1  ./llama-bench   -m model.gguf -p 512 -n 128
GGML_ANALOG_ENABLE=1  ./llama-perplexity -m model.gguf -f wiki.test.raw --chunks 8
```

`GGML_ANALOG=ON` requires the CPU backend (`GGML_CPU`), because the emulated mixed-signal core is
the CPU backend. It is not compatible with `GGML_BACKEND_DL`.

The backend is plain C++ and builds on Windows as well: the passes fall back to `std::thread` when
the build has no OpenMP, the Vulkan loader is opened at runtime (`vulkan-1.dll` there,
`libvulkan.so.1` on posix) so no Vulkan library is linked, and the shaders only need a GLSL
compiler named by `GGML_ANALOG_GLSLC` (`glslangValidator.exe` or `glslc.exe`). On a Windows
machine with a Vulkan driver `--analog-dispatch vulkan` runs the SPIR-V modules of the passes on
it; without one it falls back to the CPU mirror exactly like on Linux.

## Configuration

All parameters are read from the environment when the backend is registered. The defaults model a
plausible 4 bit ReRAM array.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GGML_ANALOG_ENABLE` | `0` | use the device at all |
| `GGML_ANALOG_IDEAL` | `0` | disable every non-ideality, i.e. the array is a perfect copy of the weights |
| `GGML_ANALOG_BITS_W` | `4` | conductance levels per cell (weight precision) |
| `GGML_ANALOG_BITS_X` | `8` | levels of the input DAC (activation precision) |
| `GGML_ANALOG_BITS_ADC` | `8` | levels of the column ADC |
| `GGML_ANALOG_SIGMA_PROG` | `0.02` | programming error of a cell, relative to the block full scale |
| `GGML_ANALOG_SIGMA_THERM` | `0.005` | read noise of one array pass, relative to the readout full scale |
| `GGML_ANALOG_DRIFT` | `0.00` | conductance drift applied at the end of programming |
| `GGML_ANALOG_IR_DROP` | `0.00` | parasitic wire attenuation along the columns |
| `GGML_ANALOG_ADC_SIGMA` | `4.0` | readout range, in standard deviations of the partial sum |
| `GGML_ANALOG_TILE_ROWS` | `128` | rows (output channels) per crossbar tile |
| `GGML_ANALOG_TILE_COLS` | `128` | columns (reduction dimension) per crossbar tile |
| `GGML_ANALOG_PJ_PER_MAC` | `0.1` | modeled energy per MAC |
| `GGML_ANALOG_MVM_LATENCY_NS` | `300` | modeled latency of one array pass |
| `GGML_ANALOG_SEED` | `42` | seed of the noise and programming errors |
| `GGML_ANALOG_ALL_MATMUL` | `0` | program every matrix, not only the model weights |
| `GGML_ANALOG_N_THREADS` | `GGML_DEFAULT_N_THREADS` | threads of the array passes, overridden by `ggml_backend_set_n_threads()` |
| `GGML_ANALOG_DISPATCH` | `cpu` | where the passes run: `cpu`, `shader` (CPU mirror of the shaders) or `vulkan` |
| `GGML_ANALOG_KERNELS` | `1` | `0` forces the scalar reference instead of the CPU kernels |
| `GGML_ANALOG_VERBOSE` | `0` | log the programming and the readout of every operation |
| `GGML_ANALOG_STATS` | `0` | print the device statistics at exit |

The statistics are also available through `ggml_backend_analog_stats_get()`, and are printed when
the process exits when `GGML_ANALOG_STATS=1` (or `--analog-stats`) is set. That block is written to
stderr directly, so it is shown whatever the log level is; the rest of the backend log messages
only appear when the verbosity of the tool is raised, see [Logging](#logging) below.

### Command line

The tools that link `common` expose the same configuration as options. They write the variables
above while the command line is parsed and then ask the backend to re-read them, so the flags work
in any order and mix with the environment (the flag of a variable overrides it):

```sh
./llama-cli        -m model.gguf -p "Hello" --analog --analog-bits 2
./llama-bench      -m model.gguf -p 512 -n 128 --analog --analog-stats
./llama-perplexity -m model.gguf -f wiki.test.raw --chunks 8 --analog --analog-ideal
```

| Option | Meaning |
| --- | --- |
| `--analog` / `--no-analog` | use the emulated array at all, or not (`GGML_ANALOG_ENABLE`) |
| `--analog-ideal` | program the array without any error, identical to the CPU backend (`GGML_ANALOG_IDEAL`) |
| `--analog-bits N` | precision of a cell, 1 to 8 bits (`GGML_ANALOG_BITS_W`) |
| `--analog-tile N` | cells of a tile along the column (`GGML_ANALOG_TILE_COLS`) |
| `--analog-dispatch <cpu\|shader\|vulkan>` | where the passes run (`GGML_ANALOG_DISPATCH`) |
| `--analog-threads N` | threads of the passes (`GGML_ANALOG_N_THREADS`) |
| `--analog-latency-ns T` | modeled latency of one array pass, in nanoseconds (`GGML_ANALOG_MVM_LATENCY_NS`) |
| `--analog-stats` | print the device statistics at exit (`GGML_ANALOG_STATS`) |

Without the options, the environment variables of the table above work on every tool, including
the tests and `tools/analog-bench`.

### Example

A 2 bit cell with a 6 bit ADC, drifting by 3% after programming and with a 2% IR drop:

```sh
GGML_ANALOG_ENABLE=1 \
GGML_ANALOG_BITS_W=2 GGML_ANALOG_BITS_ADC=6 \
GGML_ANALOG_DRIFT=0.03 GGML_ANALOG_IR_DROP=0.02 \
./llama-perplexity -m model.gguf -f wiki.test.raw --chunks 8 -v
```

## Kernels

The device model has three passes that touch the cells of the array:

| pass | what it does | shape of the work |
| --- | --- | --- |
| input DAC | snaps the activations to the levels the drivers of the array can produce | one block of 32 values per column |
| programming | weights -> conductances: level snapping, programming error, drift, IR drop | every cell, once |
| readout | scales the partial sums to the full scale of the row, adds the noise of one array pass, clips and quantizes them with the ADC | every output element |

Every one of them is element-wise, which is what makes it possible to run them in parallel and on
a GPU without changing the result. They exist in three forms:

| where | files | notes |
| --- | --- | --- |
| scalar reference | `ggml/src/ggml-analog/analog-kernels.cpp` | the definition of the model, single threaded, `GGML_ANALOG_KERNELS=0` |
| parallel CPU kernels | `analog-kernels.cpp`, `analog-parallel.h` | AVX2, NEON or scalar inner loops, over the OpenMP runtime or std::thread |
| compute shaders | `ggml/src/ggml-analog/vulkan/*.comp` | GLSL 4.50 compute shaders, compiled to SPIR-V and embedded at build time |
| mirror of the shaders | `ggml/src/ggml-analog/analog-shader.cpp` | the same kernels, line by line, as C++ (for machines without a Vulkan device) |
| Vulkan dispatch | `analog-vulkan.cpp` | dispatches the embedded SPIR-V on a Vulkan 1.1 device, opened at runtime with `dlopen` |

The three implementations are required to agree **bit for bit**, not statistically: the passes are
element-wise, the summation order of the reference is preserved (one invocation per row for the
programming pass, one per column for the input RMS), and the random numbers are the same 64 bit
splitmix hash on every backend, with the same 32 bit mixing constants and the same triangular
readout noise. That is what makes it possible to say that a run on a GPU is the run of
`docs/backend/ANALOG.md` and not a similar-looking one.

What the tests check, and what they can not check without a device:

| | result |
| --- | --- |
| the CPU kernels against the scalar reference | bit for bit, over 3 array configurations and 5 shapes, for every pass |
| the mirror against the scalar reference | bit for bit, same matrix |
| the mirror on the whole model against the CPU kernels | bit for bit, `test-analog-backend -d shader`, 9 `MUL_MAT` cases in 2 array configurations |
| the SPIR-V modules | magic, version, bound, `GLCompute`/`main` entry point, the `Int64` capability of the two passes that draw the noise, and the mixing constants of the model among the literals |
| the push constants | `static_assert` on the size of every block in `analog-vulkan.cpp`, next to the `layout` of the shader that reads it |
| the Vulkan dispatch | compiled and linked (with the Vulkan headers) but **not executed here**: this repository was developed on a machine without a Vulkan loader or device, so the dispatch reports why it cannot initialize and the mirror runs instead |

The last row is the honest limit of the shaders in this tree: the SPIR-V is built, validated and
embedded, the same kernels are executed and compared in their mirror form, but nobody has run the
GLSL of `vulkan/` on silicon from here. On a machine with a Vulkan 1.1 device the same commands
below execute the SPIR-V, and `test-analog-backend -d vulkan` checks the result of the whole model
against the CPU kernels (it allows one ulp there: a driver may contract a multiply and an add,
which the CPU paths do not do).

```sh
build/bin/test-analog-kernels -r 5                  # parity and speed of the passes
GGML_ANALOG_DISPATCH=shader build/bin/test-analog-kernels   # same check through the shader mirror
build/bin/test-analog-backend -d shader             # the whole model through the mirror
build/bin/test-analog-backend -d vulkan             # the whole model through the SPIR-V modules
```

What the passes cost on a 2 vCPU machine, on a 1024 x 2048 F16 matrix with the default
non-idealities and on a 2048 x 64 element readout (medians of 15 runs, `test-analog-kernels -r 15`):

| pass | scalar reference | CPU kernels, 1 thread | CPU kernels, 2 threads | shader mirror, 1 thread | shader mirror, 2 threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| programming, 2.1 M cells | 46.9 ms | 46.7 ms | 23.7 ms | 58.6 ms | 38.0 ms |
| readout, 131 k elements | 487 us | 486 us | 260 us | 1733 us | 881 us |

Reading the table: the passes *scale with threads* (1.9x on 2 cores, far more on a machine with
more), but the programming pass is bound by the draw of the per-cell programming error (four
hashes per cell), which is why the vectorized kernel is not faster than the scalar one at one
thread. The readout gains from both. The shader mirror is not a fast path: it is the definition of
the shaders, kept on the CPU so that the model can be run and compared without a GPU.

On the test model of `benches/analog` this shows up as (4 bit array, 29 matrices, 2 threads, the
numbers are from `benches/analog/RESULTS.md`, where the same machine moves by +-30% between runs):

| | before, serial passes | now, parallel kernels | shader mirror | `vulkan` on a machine without a GPU |
| --- | ---: | ---: | ---: | ---: |
| programming of the arrays | 89.3 ms | 66.5 ms | 66.6 ms | 63.1 ms |
| readout | 59.0 ms | 13.6 ms | 56.0 ms | 55.4 ms |
| first evaluation | 114.5 ms | 83.9 ms | 89.1 ms | 84.8 ms |
| prompt processing, 128 tokens | 22.2 ms | 12.2 ms | 21.5 ms | 21.1 ms |
| token generation | 0.45 ms | 0.35 ms | 0.56 ms | 0.58 ms |

`GGML_ANALOG_DISPATCH=vulkan` falls back to the mirror when no device can be initialized (no
loader, no Vulkan 1.1 device, or no `shaderInt64`), and logs why; on a machine with a GPU the same
run executes the SPIR-V modules. Note that the Vulkan path uploads and downloads its buffers on
every call: the partial sums of the mixed-signal core are produced in host memory, so a GPU
dispatch is the emulation of a tile controller running its peripheral kernels, not a faster way to
run the emulation. The point of the shaders is that a real array controller would run exactly this
code next to the tile - the DAC, the programming and the readout of a crossbar are peripheral
circuits, not matrix arithmetic.

## Accuracy

With `GGML_ANALOG_IDEAL=1` the emulation is an exact copy of the weights and the results are
bit-identical to the CPU backend, which makes it easy to check that the plumbing is correct:

```sh
./llama-perplexity -m model.gguf -f wiki.test.raw --chunks 8                                # PPL = x
GGML_ANALOG_ENABLE=1 GGML_ANALOG_IDEAL=1 ./llama-perplexity -m model.gguf -f wiki.test.raw --chunks 8   # PPL = x
```

With the non-idealities enabled the results differ from the CPU backend: a 4 bit cell with a
programming error of 2% produces an error of 16-28% RMS on the output of a single matrix
multiplication, and logits that move by ~0.03 RMS, which is enough to change the sampled token
whenever the distribution is not extremely peaked. `benches/analog/RESULTS.md` has the measured
numbers, and `benches/analog/README.md` explains how to reproduce them.

## Performance

The emulation is CPU-bound, so the analog backend is *slower* than the CPU backend in every
configuration that programs real weights: every value has to be snapped, perturbed and possibly
re-quantized once, and each operation pays for the readout, which is a second pass over every
output element. On the test model of `benches/analog` the steady state is *not* where the cost of a
4 bit array shows up: prompt processing and token generation come out at 12.6 ms and 0.37 ms
against 14.1 ms and 0.45 ms of the CPU backend, both inside the +-30% that this machine moves
between runs. What shows up is the first evaluation, which programs 0.9 M weight cells in ~45 ms
(~50 us per 1000 cells, ~90 ms before the passes were parallel, see the previous section), and the
accuracy, which is the point of the whole exercise.

Model *loading* is not affected, because the weights are read into the same host buffers as before
and the arrays are programmed lazily, the first time a matrix is used. The cost of programming
therefore shows up in the first evaluation rather than in the load.

The useful numbers are the modeled ones - `est_time_ms`, `n_array_passes`, `est_energy_mj` - which
describe the device that is being emulated rather than the emulation. In the run above the modeled
device would have needed 10.6 ms and 0.161 mJ for work that took 144 ms to emulate.

### Digital kernel against the modeled array

The three numbers are easy to confuse: what the digital CPU kernel takes, what the emulation of
the analog device takes on that same CPU, and what the modeled hardware would take.

| on the test model of `benches/analog` | digital CPU | analog, emulated | analog, modeled |
| --- | ---: | ---: | ---: |
| prompt of 128 tokens | 14.1 ms | 12.6 ms | 2.38 ms |
| one generated token | 0.45 ms | 0.37 ms | 0.0222 ms |
| energy, generation | - | - | 0.347 uJ/token |

The modeled time of a matrix-vector product is `ncols * ceil(K/tile_cols) * mvm_latency_ns`: it is
set by how many times the input vector has to be swept through the array, not by the arithmetic.
That is why the array looks best for the shapes of token generation (one column, K wide, which the
digital kernel serves at memory bandwidth) - 3x to 20x faster than the CPU kernel in the `MUL_MAT`
table of `benches/analog/RESULTS.md` - and worst for wide batches, where the digital kernel has
enough arithmetic to amortize the loads.

The assumed latency is a free parameter of the model, and it decides whether the array is ahead at
all. With `GGML_ANALOG_MVM_LATENCY_NS` swept on the same model:

| assumed pass latency | prompt, modeled | generation, modeled | prompt vs digital | generation vs digital |
| --- | ---: | ---: | ---: | ---: |
| 100 ns | 0.795 ms | 0.0074 ms/token | 17.7x faster | 61x faster |
| 300 ns (default) | 2.384 ms | 0.0222 ms/token | 5.9x faster | 20x faster |
| 1000 ns | 7.948 ms | 0.0740 ms/token | 1.8x faster | 6.1x faster |
| 3000 ns | 23.844 ms | 0.2220 ms/token | 1.7x slower | 2.0x faster |

i.e. on this model the array stops being worth it somewhere around 1.7 us per pass for prompt
processing and 4.9 us per pass for token generation. Note that the model only counts the array
passes: the DAC and ADC energy and latency, which on real designs are comparable to the array
itself, are not in these numbers.

There is no point in comparing `t/s` of an analog emulation with `t/s` of the CPU backend to decide
which one is "better"; the point is to see how the accuracy and the modeled energy of a hypothetical
chip change with its parameters.

## API

The backend exposes a small C API in `ggml-analog.h`, the rest of the integration goes through
the regular backend registry:

```c
ggml_backend_reg_t ggml_backend_analog_reg(void);            // the backend registry entry
ggml_backend_t     ggml_backend_analog_init(void);           // an instance of the device
bool               ggml_backend_is_analog(ggml_backend_t);   // is this the analog device
void               ggml_backend_analog_set_n_threads(ggml_backend_t, int);
void               ggml_backend_analog_set_enabled(bool);    // off by default
bool               ggml_backend_analog_get_enabled(void);
void               ggml_backend_analog_config_reload(void);  // re-read the environment
void               ggml_backend_analog_stats_reset(void);
void               ggml_backend_analog_stats_get(struct ggml_analog_stats *);
void               ggml_backend_analog_stats_print(void);
```

`struct ggml_analog_stats` reports both the cost of the emulation (programmed tensors, cells,
wall time of the programming, of the emulated compute and of the readout) and the cost of the
modeled device (MACs, array passes, ADC samples, time, energy).

## Tests

`tests/test-analog-backend.cpp` compares the analog device against the CPU backend for a range of
matrix shapes and weight types, in two passes: with an ideal array, where the maximum error must
be exactly 0, and with the default non-ideal model, where the errors, the wall time and the
modeled energy are reported.

```sh
cmake -B build -DGGML_ANALOG=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j
ctest --test-dir build -R analog-backend --output-on-failure
```

`tests/test-analog-kernels.cpp` checks the three passes of the array model directly: the parallel
CPU kernels and the mirror of the compute shaders against the scalar reference (bit for bit), the
SPIR-V modules of the shaders (version, capabilities, entry point), and the speed of each pass at
one and at all threads.

```sh
ctest --test-dir build -R analog --output-on-failure          # both tests
build/bin/test-analog-kernels                                 # the passes, in detail
```

The end to end comparison against the CPU backend (loading, throughput, perplexity, logits) is
described in `benches/analog/README.md`.

## Limitations

- Only `GGML_OP_MUL_MAT` is emulated. On a real accelerator the attention part, the KV cache and
  the samplers run on the digital core too; here they simply run on the CPU backend.
- `GGML_OP_MUL_MAT_ID` (MoE) is not emulated yet: expert weights are evaluated on the CPU backend.
- Weight formats that cannot be re-quantized without an importance matrix (`IQ2_XXS`, `IQ2_XS`,
  `IQ1_S`) are programmed into the array as F16, because the array has no access to the importance
  matrix. The emulation is therefore not measuring what those formats would cost on the device, it
  is measuring the array model applied to a more precise copy of the weights.
- Crossbar-level effects that need the actual circuit, such as sneak paths, cell-to-cell coupling
  and saturation of the column current, are not modeled. IR drop and noise are modeled as
  independent per-cell / per-readout perturbations.
- The modeled time and energy ignore the digital part of the accelerator, i.e. the ADC and DAC
  energy, which on real designs contributes as much as the array itself.
- Per-channel (per-row, per-column) programming errors, which show up in hardware as fixed pattern
  noise, are not modeled: the errors are per cell and independent.
- The compute shaders cover the three passes of the array model, i.e. the peripheral circuits of the
  tile. They are not a second implementation of the mixed-signal core, which runs the `MUL_MAT`
  itself on the CPU backend in this emulation.

## Logging

The tools filter out `GGML_LOG_LEVEL_INFO` messages of backends unless their verbosity is raised,
so run them with `-v` (or redirect the log with `--log-file`) to see the messages of the backend:

```sh
GGML_ANALOG_ENABLE=1 GGML_ANALOG_VERBOSE=1 GGML_ANALOG_STATS=1 \
    ./llama-perplexity -m model.gguf -f wiki.test.raw --chunks 1 -v

analog: programmed blk.0.attn_q.weight (q8_0 -> q8_0, 256 rows x 256, 65536 cells) in 2.31 ms
...
analog: ===== array statistics =====
analog: programmed tensors:       29 (0.004 GiB, 0.920 M cells) in 54.8 ms
analog: array operations  :     1508 (0.344 G MAC, 0.055 G ADC samples)
analog: emulated compute  :     99.3 ms (readout 13.9 ms, 35344 array passes)
analog: modeled device    :    10.604 ms, 0.161 mJ
```
