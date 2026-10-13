# KNC rebase qualification

The candidate is based on upstream `4413a192410fad1f054e9216114087a3e01527a0`, verified as upstream `master` on 2026-10-13. The compiler compatibility prerequisite and the opt-in KNC CPU implementation are separate commits.

## Completed host checks

The ordinary CPU path uses GCC 12 and C++17 in an isolated x86-64 Linux VM. Its emulated CPU exposes neither SSE4.2/BMI2 nor AVX, so those build options are disabled explicitly.

| Check | Result |
|---|---|
| Upstream quantization tests | Pass |
| Upstream GGUF tests | Pass |
| Upstream sampling tests | Pass |
| Small Q3_K/Q4_K/Q5_K/Q6_K matrix cases against CPU reference | 52/52 pass |
| Tiny mixed Q3_K_M/Q4_K_M/Q5_K_M model state checks | Pass; zero logit differences across prefill, incremental decoding, three-token chunks and reset |
| Synthetic affinity policy | 24 card/layout/sibling combinations pass; excluded CPUs and unexpected hardware rejected |
| Python type checking for added scripts | Pass with ty 0.0.84 |
| EditorConfig and Git whitespace checks | Pass |

The assembly source-semantics tests use an independent packed-value oracle with ASan/UBSan. Each profile passes 27,648 numerical row checks and verifies activation staging once per block per tile. The bounded prefetch checks number 1,444,608 for 3120A and 1,165,824 each for 5110P and 7120P. These tests execute translated source semantics on the host; they do not emulate the native ISA.

The three model fixtures are synthetic two-layer models, about 12-14 MB each, with a five-token prompt, 128-token context and batch size 8. Each comparison covers 160,010 logits. They establish loader and state consistency, not model quality or representative throughput. Fixture SHA-256 values, in Q3/Q4/Q5 order:

```
a2931b8e61040369cf107e878e02745ccaa4ebec5376a355bf49257497a584ef
f2ce5fe9d8d7c75d299f9b2dd0dc55a90080b177738546aee0c9f96e58291eed
a56c35c9157bb6d3f90be2606402a4851bd9407a37f74fd633fc3ba6c6fce895
```

## Completed cross-build checks

All three profiles build and link with MPSS 3.8.6 GNU GCC 5.1.1 in the isolated Linux VM:

| Profile | L1 / L2 leads | Core, simple example and five focused tests | ELF and IMCI audit |
|---|---|---|---|
| 3120A | 4 / 16 | Pass | Pass |
| 5110P | 8 / 32 | Pass | Pass |
| 7120P | 8 / 32 | Pass | Pass |

The packaged executables are ELF64 Intel K1OM, use the K1OM dynamic loader, require no GLIBC version newer than 2.14, and resolve the matching MPSS OpenMP library through `$ORIGIN/../lib`. Compiled Q3/Q4/Q5/Q6 kernels contain native IMCI vector operations and both L1 and L2 prefetch instructions. These checks validate the emitted binaries and dependencies, not execution on the card.

The local preservation audit compares 839 release files and 336 installed-runtime files with the pre-task hashes; all are unchanged. Build products and detailed receipts remain separate from the source contribution and installed runtime.

## Native qualification gate

The connected card's SSH identity differs from saved trust. Native correctness, actual worker affinity and small trained-model throughput remain pending verification through a trusted console or recovery connection. No trust records or installed runtime are changed to bypass this gate. There are no new KNC tokens/sec results for this rebase, and previous-base measurements do not qualify it.

After identity verification, run the operator and graph checks before short, alternating baseline/control/staging model runs. Use the same model hash, prompt, context, batch, worker layout and matching OpenMP library; capture full-vocabulary logits, state reset, prompt/decode timings, worker affinities and thermal guard evidence. Record source and binary hashes. Repeat on each of 3120A, 5110P and 7120P before claiming per-card native qualification.

The supported compiler target is the core library, simple example and focused tests. Common/server and multimodal frontends, reasoning templates and long-context quality remain outside this qualification. Human review and maintenance ownership are required before an upstream submission, especially for the broad GCC 5 compatibility prerequisite.
