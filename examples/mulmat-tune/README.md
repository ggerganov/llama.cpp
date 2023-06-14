# Mulmat Benchmark and Tunning

Apart from the standalone tool `mulmat-tune`, mulmat tune is also integrated into
`main` and `perplexity`. To avoid too many new cli options, I just added two options.
To make it run faster, the `m_num` is set as 8 thus max M is 128, and the `n_pass`
is set as 1.

With the newly added cli options, we can use `main` and `perplexity` with the
following three ways:

* bench and run:  --tune
* bench and exit: --tune --tune-file <FILE>
* load  and run:  --tune-file <FILE>

The `load` mode reads existing data file. Although this is fine because we can
run bench ahead of time (saving tens of seconds), but there are two shortcomings:
- have to re-run when format changed, this is OK because we are acknowledged.
- the most subtle problem is algorithm was changed silently but we are using the
  outdated format. So I integrated mulmat tune into `main` and `perplexity` as
  a complementary solution.

## Build into main and perplexity

Makefile:
```
make clean && LLAMA_MULMAT_TUNE=1 make
```

CMake (with BLAS):
```
cmake --build . --target clean
cmake .. -DLLAMA_BLAS=ON -DLLAMA_MULMAT_TUNE=ON
cmake --build . --config Release
```

Run examples:

```
# bench and run:

./main -m ./models/3B/open-llama-3b-q4-0.bin -c 512 -b 1024 -n 256 --keep 48 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt -t 4 --tune

# bench then exit:
./main -m ./models/3B/open-llama-3b-q4-0.bin --tune --tune-file <FILE>

# load and run

./main -m ./models/3B/open-llama-3b-q4-0.bin -c 512 -b 1024 -n 256 --keep 48 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt -t 4 --tune-file <FILE>
```

# Build the standalone `mulmat-tune`

Makefile:
```
make clean && LLAMA_MULMAT_TUNE=1 make
```

CMake (with BLAS)
```
cmake --build . --target clean
cmake .. -DLLAMA_BLAS=ON -DLLAMA_MULMAT_TUNE=ON
cmake --build . --config Release
```

Run examples:

```
./mulmat-tune -h

# run with default params (7B, Q4_0, ...)
./mulmat-tune

# set model
./mulmat-tune --model 13B

# set ggml ftype, 2 for Q4_0, 3 for Q4_1, run `mulmat-tune -h` for help.
./mulmat-tune --ftype 3

# customized m_num
./mulmat-tune --m_num 8

# customized n_pass: run 1 pass only instead of the default 3.
./mulmat-tune --n_pass 1

# customized n_threads instead of the default 1.
./mulmat-tune --n_threads 4

# save to file
./mulmat-tune --file <FILE>

# save to file, always override if exists (CAUTION!)
./mulmat-tune --file <FILE> -y

```

# End to End Test

## Compare With Master

You may want to run the following commands. Make sure the tune result file is
setup properly.

General steps:

1. run `./mulmat-tune -h` to see how to build for misc vendors.
   you can build with `GGML_MULMAT_TUNE_NDEBUG=` to enable the the debug, e.g:
   ```
   make clean; LLAMA_MULMAT_TUNE=1 LLAMA_MULMAT_TUNE_NDEBUG=1 LLAMA_NO_ACCELERATE=1 LLAMA_CLBLAST=1 make
   ```
   On `macOS`, `ACCELERATE` is enabled by default. When `ACCELERATE` is built along
   with `CUDA` or `CL`, you may not see `CUDA` or `CL` from debug because `CPU`
   or `CPU_BLAS` is more faster (as of the estimation from mulmat tune).
2. create a small prompt file:
   ```
   head -n 5 ./models/wikitext-2-raw/wiki.valid.raw > ./models/wiki.valid-5.raw
   ```
3. run any of the following example commands.
   ```
   ./perplexity -m models/7B/ggml-model-q4_0.bin -f ./models/wiki.valid-5.raw -c 128 --mlock -t 1 -b 32
   ./perplexity -m models/7B/ggml-model-q4_0.bin -f ./models/wiki.valid-5.raw -c 128 --mlock -t 4 -b 64
   ```
   * `--mlock` is recommended for `macOS`, you may not want to use it.
   * don't change `-c 128`: too large `context size` causes 0 perplexity trunk.
   * `-t` is the number of threads, recommend `1`, `2`, `4` or `6`.
   * you can change the batch size (`-b`) between `1` and `128`.
   * you may want to add other cli options.

The following results are generated with Accelerate compiled.

### 1 thread

**Master (2d43387d)**

```
| M   | perplexity (seconds per pass) | prompt eval time (ms per token) |
| --- | --------------- |
|  8  |  43.53 | 339.95 |
|  16 |  44.31 | 346.12 |
|  24 |  43.14 | 336.90 |
|  32 |  33.59 | 262.25 |
|  40 |  27.64 | 215.77 |
|  48 |  24.52 | 191.42 |
```

**This branch (tune)**

```
|  M  |  perplexity (seconds per pass) | prompt eval time (ms per token) |
| --- | --------------- |
|  8  |  43.78 | 341.96 |
|  16 |  42.88 | 334.93 |
|  24 |  42.06 | 328.42 |
|  32 |  33.07 | 258.25 |
|  40 |  28.69 | 223.98 |
|  48 |  25.65 | 200.19 |
```

### 4 threads

**Master (2d43387d)**

```
|  M  |  perplexity (seconds per pass) | prompt eval time (ms per token) |
| --- | --------------- |
|   8 |  12.43 |  96.99 |
|  16 |  12.10 |  94.44 |
|  24 |  12.81 |  99.95 |
|  32 |  31.64 | 247.04 |
|  48 |  24.55 | 191.63 |
|  64 |  17.56 | 137.09 |
|  96 |  17.59 | 137.25 |
| 128 |  10.73 |  83.74 |
```

**This branch (no tune)**

```
|  M  |  perplexity (seconds per pass) | prompt eval time (ms per token) |
| --- | --------------- |
|   8 |  12.31 |  96.07 |
|  16 |  12.00 |  93.63 |
|  24 |  12.07 |  94.15 |
|  32 |  20.34 | 158.76 |
|  48 |  15.86 | 123.73 |
|  64 |  10.98 |  85.69 |
|  96 |  11.24 |  87.66 |
| 128 |   7.53 |  58.77 |
```

**This branch (tune)**

```
|  M  |  perplexity (seconds per pass) | prompt eval time (ms per token) |
| --- | --------------- |
|   8 |  12.48 |  97.37 |
|  16 |  12.26 |  95.70 |
|  24 |  12.25 |  95.53 |
|  32 |  11.98 |  93.58 |
|  48 |  12.57 |  98.12 |
|  64 |  11.28 |  88.05 |
|  96 |   9.55 |  74.53 |
| 128 |   7.51 |  58.61 |
```

# Bench Data Format

**Example**

```
5 3B 2 6 1

3200 3200  2 0 3 10
16 0 0 0  16 1 0 1   0 0 0 0
16 1 0 2  17 0 1 0   0 0 0 0
 0 0 0 0  34 0 1 0   0 0 0 0
   1        1      793 0     9103     2102 0 0     6014 0
   2        2     1591 0     8034     2305 0 0    30982 0
   4        4     2236 0     6476     2484 0 0    31388 0
   8        7     4161 0     6623     2389 0 0    29204 0
  16       15     8339 0     6434     2752 0 0    34303 0
  32       32    16919 0     6915     3651 0 0    42511 0
  64      200    34270 0     6574     4528 0 0    68212 0
 128      188    69400 0     6325     6839 0 0    74437 0
 256      303   134597 0     6168    11544 0 0   110180 0
 512      687   279685 0     6337    29712 0 0   159728 0

3200 8640  2 0 2 10

 ...

 ```

**Informal Explanation**

```
head
groups+

head := version model ggml_ftype n_shapes n_threads
shape+

# head
version: 1
model: "3B" | "7B" | "13B" | "30B" | "65B"
ggml_ftype: 0 - 4, 7 - 14
n_shapes: number of shapes
n_threads: number of threads

shape := N K  m_num n_profiles
task_conf_profile+
bench_item+

task_conf_profile: stage_conf(init) stage_conf(compute) stage_conf(finalize)
stage_conf: backend parallel wait
backend: 0 (NONE) | 16 (CPU) | 17 (CPU_BLAS) | 32 (GPU) | 33 (GPU_CUDA) | 34 (GPU_CL)
parallel: 0 (false) | 1 (true)
wait: 0 (false) | 1 (true)

bench_item: M profile_time+
profile_time := stage_time[3]
stage_time[3]: init_time, compute_time, finalize_time
```

A task stage is invalid if it's backend equals to `GGML_TASK_BACKEND_NONE`.
Time unit is `us`. A column is all zeros when that stage does not exist.

# NOTE

1. "3B" is [open-llama 3B](https://github.com/ggerganov/llama.cpp/pull/1588).
2. Model names are subject to change: we may support something like X-3B, Y-4B, ...
3. As of Jun 1, this tool is still in early stage, will be changed frequently in
   recent couple of days (or weeks).
