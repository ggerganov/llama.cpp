# Comparing the analog backend with the CPU backend

These tools measure the analog backend against the CPU backend at three levels:

| level | tool | what it shows |
| --- | --- | --- |
| one operation | `test-analog-backend` | error and time of a single `MUL_MAT` |
| the array passes | `test-analog-kernels` | accuracy and speed of the DAC, programming and readout kernels, including their compute shaders |
| whole model | `llama-analog-bench` | loading, array programming, prompt processing, token generation, logits |
| corpus | `llama-perplexity` | quality of the model |

Nothing has to be downloaded: the test model is generated from scratch and the text corpus is
taken from the repository itself.

## Running everything

```sh
cmake -B build -DGGML_ANALOG=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j

benches/analog/compare.sh          # writes the tables to stdout, logs into /tmp/llama-analog-bench
```

`compare.sh` needs `python3` with `numpy` and `pyyaml` for the model generator
(`pip install numpy pyyaml`).

The result of a run on a 2 vCPU machine is in [RESULTS.md](RESULTS.md).

## Step by step

The model, a 4 layer llama with random weights and a byte level vocabulary (so that any text can be
tokenized without a trained vocabulary):

```sh
python3 benches/analog/make-tiny-model.py --out /tmp/medium-f32.gguf \
    --n-embd 256 --n-layer 4 --n-head 8 --n-ff 768
build/bin/llama-quantize /tmp/medium-f32.gguf /tmp/medium-q8_0.gguf q8_0
```

A single matrix multiplication, comparing the CPU backend against the emulated array:

```sh
build/bin/test-analog-backend                       # every case
build/bin/test-analog-backend -f "q4_0 1024"        # one case
build/bin/test-analog-backend -v                    # with the per operation log
```

The kernels and the compute shaders of the three passes of the array model. The test reports the
difference against the scalar reference (it must be exactly 0) and the speed of each pass with one
and with all threads:

```sh
build/bin/test-analog-kernels                      # parity + speed of every pass
build/bin/test-analog-kernels -r 20                # more repetitions for the timings
GGML_ANALOG_DISPATCH=shader build/bin/test-analog-kernels   # run the passes through the shader mirror
GGML_ANALOG_KERNELS=0 build/bin/test-analog-kernels         # force the scalar reference
```

The same model loaded by both backends, with the timing of every stage, and with the passes
dispatched to the CPU kernels, to the mirror of the compute shaders, or to a Vulkan device:

```sh
GGML_ANALOG_DISPATCH=cpu    build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -n 16 -r 3 -t 2 -c 512   # default
GGML_ANALOG_DISPATCH=shader build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -n 16 -r 3 -t 2 -c 512
GGML_ANALOG_DISPATCH=vulkan build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -n 16 -r 3 -t 2 -c 512
```

The same model loaded by both backends, with the timing of every stage:

```sh
build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -n 16 -r 3 -t 2 -c 512
GGML_ANALOG_ENABLE=1 build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -n 16 -r 3 -t 2 -c 512
```

The logits of the same prompts on both backends, to see what the array model does to the output:

```sh
build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -c 512 -q --logits /tmp/cpu.bin
GGML_ANALOG_ENABLE=1 build/bin/llama-analog-bench -m /tmp/medium-q8_0.gguf -p 128 -c 512 -q --logits /tmp/analog.bin
build/bin/llama-analog-bench --compare /tmp/cpu.bin /tmp/analog.bin
```

Perplexity, including the explicit check that an ideal array reproduces the CPU result exactly:

```sh
build/bin/llama-perplexity -m /tmp/medium-q8_0.gguf -f README.md -c 512 -b 512 --chunks 8
GGML_ANALOG_ENABLE=1 GGML_ANALOG_IDEAL=1 build/bin/llama-perplexity -m /tmp/medium-q8_0.gguf -f README.md -c 512 -b 512 --chunks 8
GGML_ANALOG_ENABLE=1 build/bin/llama-perplexity -m /tmp/medium-q8_0.gguf -f README.md -c 512 -b 512 --chunks 8
```

## Notes on the test model

The weights are random, so the model is a random function of its input: it writes nonsense and its
perplexity sits at the size of the vocabulary no matter how accurate the backends are. That is
enough to check that everything is wired up and to measure time, energy and the errors of the
intermediate results, which is what the harnesses above report, but it cannot show the *quality*
cost of a given array configuration. For that, a trained model is needed:

```sh
GGML_ANALOG_ENABLE=1 GGML_ANALOG_BITS_W=4 build/bin/llama-perplexity -m model.gguf -f wiki.test.raw --chunks 8
```
