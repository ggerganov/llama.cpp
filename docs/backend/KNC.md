# Knights Corner (experimental)

This CPU implementation targets Xeon Phi 3120A, 5110P and 7120P using MPSS GNU GCC 5.1.1 and K1OM IMCI assembly. It does not target Knights Landing or AVX-512.

Q3_K, Q4_K, Q5_K and Q6_K matrix products reuse a staged Q8_K activation block across 1, 2, 4 or 8 weight rows. Q6_K is needed by mixed Q3_K_M, Q4_K_M and Q5_K_M recipes. Tensor types and GGUF layouts remain unchanged. The specialization accepts contiguous, unbatched weights and F32 inputs/outputs; other operations and layouts use the existing CPU paths. Reference mode bypasses the specialization. Packed weights may be unaligned, and prefetches are bounded by the remaining blocks.

The scalar metadata expressions retain x87 intermediate precision until each float accumulator update. Changing block helpers to return float can change subsequent activation quantization and model logits.

## Build

Use an isolated Linux environment with the MPSS 3.8.6 sysroot and the GNU K1OM toolchain installed. No compiler, firmware, driver or runtime binaries are distributed here. The compiler must report `k1om-mpss-linux` and `5.1.1`; the directory name alone is insufficient.

```sh
/usr/linux-k1om-4.7/bin/x86_64-k1om-linux-gcc -dumpmachine
/usr/linux-k1om-4.7/bin/x86_64-k1om-linux-gcc -dumpversion
cmake -S . -B build-knc-7120p \
  -DCMAKE_TOOLCHAIN_FILE=cmake/k1om.cmake \
  -DGGML_KNC=ON -DGGML_KNC_PROFILE=7120P \
  -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF \
  -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-knc-7120p -j4
```

Select `3120A` or `5110P` in a separate build directory for those profiles. The supported build includes libllama, llama-simple and focused tests. The common library, server and multimodal frontends are not qualified with GCC 5. The C++14 compatibility prerequisite is separate from the KNC kernels; ordinary builds retain C++17.

`GGML_KNC_PREFETCH_L1` and `GGML_KNC_PREFETCH_L2` are leads in 16-value iterations, each in 0..256. Zero disables that level. `GGML_KNC_PREFETCH_STEP` is 16 or 64 values; `GGML_KNC_PREFETCH_Q8` selects activation prefetches. Defaults are 4/16 on 3120A and 8/32 on 5110P/7120P, step 64 and Q8 enabled. These are starting policies, not proof of optimal performance on each card.

Do not substitute `-march=native`, AVX-512 flags, or global fast math. The toolchain uses `-mknc`. Activation reuse and the extended K-quant glue use strict arithmetic; only the inherited Q5/F32 glue retains its established compilation options.

## Execution and tuning

Stage executables and the matching MPSS `libgomp.so.1` in a new directory on the card. Preserve any installed runtime. The GNU OpenMP runtime uses `GOMP_CPU_AFFINITY`, not Intel's `KMP_AFFINITY`.

`scripts/knc-affinity.sh` reads actual sibling groups, reserves the entire core containing logical CPU 0, respects the current allowed CPU mask, and spreads workers across the remaining cores. It does not assume that every sibling zero is reserved for interrupts. Pass 1..4 to choose siblings per worker core. The resulting thread counts are:

| Card | Worker cores | 1 / 2 / 3 / 4 siblings |
|---|---:|---|
| 3120A | 56 | 56 / 112 / 168 / 224 |
| 5110P | 59 | 59 / 118 / 177 / 236 |
| 7120P | 60 | 60 / 120 / 180 / 240 |

Set the caller's inference thread count to the exported `OMP_NUM_THREADS`. Explicit library context/threadpool settings still control the GGML worker count. Use `GGML_KNC_MQ=0` for a same-binary control and `GGML_KNC_MQ_ROWS=1`, `2`, `4` or `8` to tune reuse. Settings are read once during CPU initialization. Start with three siblings and compare prompt evaluation and single-token decoding separately.

```sh
GGML_KNC_MQ_ROWS=8 bash scripts/knc-affinity.sh 3 sh -c \
  'exec ./build-knc-7120p/bin/llama-simple -m "$1" -ngl 0 -t "$OMP_NUM_THREADS" -n 8 "Hello"' sh small-model.gguf
```

Use a short fixed prompt and a small trained model, identical context/batch sizes, greedy generation and at least three alternating repetitions. Report model and binary SHA-256, source revision, compiler, OpenMP library, actual worker affinities, temperatures, prompt tokens and timed decode evaluations. Keep traces off for throughput measurements. A small synthetic model can check quant layouts but cannot establish model quality or representative throughput. No result for one SKU qualifies another SKU.

## Tests and provenance

Run `test-knc-dot`, `test-knc-mq` at each tile size, `test-quantize-fns` and `test-sampling` on matching hardware, with site thermal, memory and workload ownership guards. Cross-compilation is not native correctness evidence. The graph tests cover multiple widths, tails, worker counts, unaligned blocks, reference mode, strided inputs and batched fallback; end-to-end tests must also compare full-vocabulary logits, chunked prefill and state reset.

The small K-quant cases also run through upstream's backend comparison harness:

```sh
./build-knc-7120p/bin/test-backend-ops -b CPU -o MUL_MAT \
  -p 'type_a=q[3456]_K,type_b=f32,m=(1|7|8|17|33|65),n=(1|3),k=512,'
```

The filter selects 52 small cases. Filtering removes rejected cases in one pass, avoiding repeated vector compaction on a slow card.

On a Linux host, `bash scripts/test-knc-host.sh /tmp/knc-host-check 7120P` checks the emitted assembly's source semantics, activation staging counts and every executed prefetch address under ASan/UBSan. The output directory must be new. It requires a host C compiler and Python 3, and does not emulate the native ISA or measure KNC throughput.

`python3 tests/knc/test_affinity.py` checks all three card sizes, contiguous and interleaved logical CPU numbering, 1..4 siblings and rejected CPU masks using synthetic topology. Native worker affinity must still be inspected during hardware tests.

The Q5/F32 GNU assembly comes from the MIT KNC GCC kernels 0.2.1 work based on julialongtin's historical KNC implementation at `03a8b80c7d502797325788f604248dd4aabdf5ef`. The related upstream proposal is [llama.cpp PR #6440](https://github.com/ggml-org/llama.cpp/pull/6440). Q2/Q3/Q4/Q6 assembly, staging reuse and the GCC compatibility work were developed in BluePower. Original kernel license notices are retained in `ggml/src/ggml-cpu/arch/knc/LICENSE`.

The recovered `knc-atomics.h` has SHA-256 `a92cc7bf637bac437751f05eb6def7142efe09347faa4d9b4b60ae441aeae887`. It maps C11 atomic operations to GCC builtins with the same memory orders and avoids GCC 5's OpenMP `_Atomic` parser limitation.

This port needs human review and maintenance ownership before an upstream submission. Coordinate with the existing KNC proposal, review the broad compiler compatibility prerequisite separately, and disclose substantive AI assistance. Multimodal quality, reasoning templates, long-context behavior and per-SKU native performance require separate evidence.
