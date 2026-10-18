# Cache-aware MoE routing (`--moe-cache-prior`)

`--moe-cache-mib` keeps the experts of the host-resident MoE layers in an LRU cache on the GPU
(`src/llama-moe-cache.cpp`). Every selected expert that is not in that cache is a *miss* and has to
be uploaded over PCIe before the token can be computed, which is what limits the decode speed of a
partially offloaded MoE model.

`--moe-cache-prior` makes the router prefer the experts that are already in the cache. It is the
*Cache-Prior Reranking* of:

> A. Skliar, T. van Rozendaal, R. Lepert, T. Boinovski, M. van Baalen, M. Nagel, P. Whatmough,
> B. Ehteshami Bejnordi (Qualcomm AI Research), *Mixture of Cache-Conditional Experts for Efficient
> Mobile Device Inference*, TMLR 06/2025 — https://arxiv.org/abs/2412.00099 (section 3.3)

The paper reports a cache miss rate reduced by more than 50 %, a perplexity impact of 0.1–3 %, a
downstream accuracy impact below 0.1 % and about 2× decode speedup on a phone, with no training and
no change to the model file.

## What it does

For every MoE layer and every token, let

* `s[e]` — the score the router gives to expert `e` (the `selection_probs` of `build_moe_ffn`, i.e.
  the softmax/sigmoid of the gate logits, after the `e_score_correction_bias` of DeepSeek and after
  the expert-group masking),
* `m[e]` — 1 if the weights of expert `e` are currently in the MoE cache of that layer, else 0,
* `Δ` — the largest score of the token,
* `λ` — `--moe-cache-prior`, in `[0, 1]`, the only knob.

The ranking of the experts is done with

```
s'[e] = s[e] + λ · Δ · m̃[e]        m̃ = m | [ s[e] >= max(s) ]
```

**`s'` is only used to rank**: the weights the selected experts are combined with are still read
from the unbiased `s`, exactly like the `e_score_correction_bias` of DeepSeek V3 does it, so an
expert that is selected either way keeps the weight the router gave it.

`λ = 0` is the original routing bit for bit, `λ = 1` is the fastest routing that still selects the
expert the router likes most in every layer. There is no second knob: the promotion of the best
expert of the router is always on, because it is what makes `λ` safe to push to 1, and turning it
off would be a different feature rather than a bigger `λ`.

`Δ` is equation (10) of the paper estimated per token instead of with a running average over a
calibration set, which appendix D of it finds equivalent. For the gates llama.cpp uses the smallest
score of a token is close to 0 — a softmax over many experts, or a sigmoid far from saturation — so
the largest score is the range. It is extracted in the graph with a sharp softmax: the kernel
subtracts the maximum of the row before the exponent, so `softmax(s · 16384)` is a one-hot at the
argmax and `sum_rows(s · onehot)` is that score. It is clamped at 0, so a token whose scores are all
negative gets no bias at all and is routed as before.

Consequences of `bias = λ · Δ`:

* a cached expert wins a comparison exactly when it beats the other one by less than `λ · Δ`, so `λ`
  is "how much of the confidence of the router the cache may spend";
* for a peaked router `Δ` is large and the prior swaps more experts, for a flat one `Δ` is small and
  it swaps fewer, which is the adaptivity the paper asks for;
* the best expert of the router is promoted together with the cached ones, so it never loses its
  place: an expert below it scores less than `max(s)` and gets the same bias, therefore
  `s_e + λΔ·m̃_e < max(s) + λΔ = s'_argmax`.

## Architectures

The prior lives in `build_moe_ffn`, so it covers every architecture that routes through it: the
Qwen MoE family (`qwen2moe`, `qwen3moe`, `qwen35moe`, `qwen3vlmoe`, `qwen3next`, `qwen4exp`),
DeepSeek V2/V3 and its group masking, Mixtral, Phi-3.5-MoE, LFM2/LFM2-MoE, GPT-OSS, and the rest.
It needs the layer to keep all of its experts in host memory, so the merged `ffn_gate_up_exps` path
and the per-expert scales of Qwen3.5 are covered, and the shared experts are not routed and
therefore not touched. What it does not cover:

* an architecture that passes its own `selected_experts_in` — the selection is already made there,
  so there is nothing to rerank;
* `LLM_ARCH_LLAMA4` and `LLM_ARCH_GROVEMOE`, which rank the raw logits instead of the gate scores.
  The prior still applies, but `Δ` is then the largest logit rather than a probability, so it is much
  larger and `λ` has to be chosen smaller;
* a batch larger than 32 tokens, where the MoE cache itself is bypassed.

## When it is active

The prior needs the state of the cache, so it is applied only where `--moe-cache-mib` is in effect:

* the model must be MoE and the layer must keep **all** of its experts in host memory
  (`--n-cpu-moe`, `--cpu-moe`, `-ot ffn_.*_exps=CPU`), on a device that has a MoE cache;
* the ubatch must be small enough that its experts are read from the cache
  (`llama_moe_cache::get_slot_map`: at most 32 tokens, and at most as many distinct experts as there
  are slots). Large prompt chunks are routed as before, so prompt processing keeps its behaviour and
  only fills the cache;
* `λ > 0`.

Every tensor of the prior is built from the shape of the router scores, not from the one of the
ubatch: a hybrid architecture (LFM2 for example) can route a different number of tokens than the
ubatch carries, and the prior follows the scores. When the two differ, the layer still gets its
prior and `LLAMA_LOG_DEBUG` reports it (`-lv 4`).

The mask `m` is a host input tensor of the graph (`inp_moe_cache_prior-<il>`) that is refreshed
before every ubatch from the slot map the LRU maintains in host memory, so it always describes the
cache state left by the previous ubatch. Nothing is read back from the GPU and no graph split is
added: the mask only feeds the ranking, and the ranking still happens in the split that precedes the
slot-map lookup of `build_moe_cache_slots`, which the copy callback of the cache asserts.

## Usage

```bash
llama-server -m qwen3-235b-a22b-q4_k_m.gguf \
    -ngl 99 -ncmoe 99 \
    --moe-cache-mib 6144 \
    --moe-cache-prior 1.0
```

| flag | meaning |
| --- | --- |
| `--moe-cache-prior F` | `λ` in `[0, 1]`, `0` (default) = the original routing, `1` = the fastest routing that still selects the best expert of the router in every layer |
| `LLAMA_ARG_MOE_CACHE_PRIOR` | the same as an environment variable |

Start at `1.0` and measure: the promotion of the best expert is always on, so the deviation is
bounded by construction, and `λ` only decides how much of the rest of the selection the cache may
take over. If the quality drops, `0.3–0.5` is the conservative range; the miss rate saturates well
before `1.0`, so there is usually nothing to gain above `0.7`.

At exit the cache logs what happened:

```
llama_moe_cache: ubatch <= 8: hits = 120453, misses = 8312, hit rate = 93.55%, uploaded = 512.31 MiB
llama_moe_cache: ubatch  > 8: hits = 40211, misses = 31904, hit rate = 55.76%, uploaded = 1971.44 MiB
llama_moe_cache: cache prior: lambda = 0.20, routed 48/48 cached layers
```

## Verifying it

* `tests/test-moe-cache-prior.cpp` — checks the host side mask (`llama_moe_cache_prior_mask`) and
  rebuilds the graph side bias on the CPU backend, comparing the selection with a plain C++
  reference of the algorithm for gaussian, softmax-like, peaked and all-negative scores, and for
  `λ = 0, 0.1, 0.3, 0.5, 1.0`. `λ = 0`, an empty cache and a full cache must all reproduce the
  original selection exactly.
* The tensors of the routing are named for the graph callbacks (`llm_graph_cb`, the `cb_eval` of
  `llama_context_params`): `ffn_moe_probs` (unbiased), `ffn_moe_prior_delta`, `ffn_moe_prior_bias`,
  `ffn_moe_probs_cache_prior` (the ranking that is used), `ffn_moe_topk` (the selection),
  `ffn_moe_weights` (still unbiased).
* `llama-perplexity` with and without the flag measures the quality side, the `hit rate` /
  `uploaded` lines above measure the traffic side.
* `scripts/simulate-moe-cache-prior.py` simulates the LRU and
  the reranking and prints miss rate, top-1 expert match and expert-set match against `λ`, which is
  the fastest way to pick a `λ` before running a model.

## The promotion of the best expert, and why it is not a knob

The paper promotes the top-`J` experts of the router together with the cached ones, `m̃ = m ∪ top-J`.
Writing that union into the mask would need a scatter over the rows that an index tensor selects per
token, and ggml does not have one: `ggml_set_rows` indexes the *second* dimension of the destination
while it iterates the third, so it would need `n_expert ≥ id` with `n_tokens` iterations, and
`ggml_add_id` reads `src1[id, i2]`, so its source would need `n_expert` rows per index value, i.e.
`O(n_expert²)`. Both were tried and both abort (`GGML_ASSERT(i1 >= 0 && i1 < ne1)` and
`GGML_ASSERT(i11 >= 0 && i11 < ne11)`).

No scatter is needed, because the top-`J` of a token is a **threshold**:

```
top-J = { e : s[e] >= s[J] }        m̃ = m + ind − m·ind,  ind = [s >= s[J]]
```

and for `J = 1` the threshold is `max(s)`, which the sharp softmax above already extracted. `ind` is
written as `1 − step(max(s) − s)` rather than `step(s − max(s))`, because the step of ggml is the
strict `x > 0` and the best expert scores exactly `max(s)`. `ggml_sgn` does not work either, it is
`0` at `0`. So the union costs one `step`, a `neg`, an `add` and three elementwise passes over
`[n_expert, n_tokens]` per layer, with no argsort and no scatter, which next to the matmuls of the
experts is free.

The guarantee needs **no cap on the bias**. A first attempt here capped it at the gap,
`min(λ·Δ, s[J] − s[J+1])`, which keeps the top-`J` as well — the bias can then never exceed the lead
of the `J`-th expert — but it caps the reranking with it, and on a router with flat scores that is
most tokens. Measured with `scripts/simulate-moe-cache-prior.py`, 64 experts, top-4, 15 slots, a flat
sigmoid-like router: the cap costs **2.4× of the miss rate** for the same 100 % top-1 agreement.

### What the promotion costs

Almost nothing, and the residual miss rate is the floor it leaves: with `k` experts selected and one
of them pinned, `k − 1` can still be swapped. `scripts/simulate-moe-cache-prior.py`, 1500 tokens, a
cache of a quarter of the experts:

| model shape | λ | miss rate | vs base | top-1 expert match |
| --- | --- | --- | --- | --- |
| 128 experts, top-8 (Qwen3-MoE-like) | 0.5 | 12.7 % | −80 % | 100 % |
| 128 experts, top-8 | **1.0** | **7.4 %** | **−88 %** | **100 %** |
| 64 experts, top-4, flat router (LFM2-like) | 0.5 | 21.0 % | −68 % | 100 % |
| 64 experts, top-4, flat router | **1.0** | **14.8 %** | **−77 %** | **100 %** |

Without the promotion (`--top-j 0` of the script) the same runs reach 0.3–0.4 % miss, i.e. −99 %, but
the top-1 expert of the router changes on 55–76 % of the tokens. That is what the promotion buys out,
and the reason `λ = 1` is a sane thing to try rather than a footgun.

The floor grows with `J`: the paper uses `J = 2` for the fine-grained MoEs, and the same run with two
pinned experts lands at 31 % instead of 15 %. `J` is deliberately not exposed — one knob that goes
from "the original routing" to "the fastest routing that still follows the router where it is
confident" is the whole feature, and a `J` that trades that floor for more of the original selection
is a different one. In the code `J` is the `n_top` of `build_moe_cache_prior`, so raising it later is
a matter of restoring the peeling loop that extracts `s[J]` with `J − 1` more sharp softmaxes.

## Known limits

* The top-1 expert of every layer stays selected, so a greedy decode keeps the strongest signal of
  the router, but the experts below it are swapped freely at a large `λ`: measure with
  `llama-perplexity` before settling on one. The top-1 **token** agreement is not measured here, it
  depends on the model and on how much of the output the swapped experts carry.
* `Δ` is the largest score of the token, not `max − min` as in equation (10), and it is computed per
  token rather than as a running average per layer. Both are approximations the paper itself
  considers equivalent (appendix D).
* The bias is applied to the scores of the gate (softmax or sigmoid). For the architectures that rank
  the raw logits (`LLM_ARCH_LLAMA4`, `LLM_ARCH_GROVEMOE`) `Δ` is the largest logit, which can be
  much larger than a probability range, so `λ` has to be chosen smaller there.
* Nothing changes when the experts are on the GPU: without host-resident experts there is no cache,
  no mask and no bias.

## Files

| file | change |
| --- | --- |
| `src/llama-moe-cache.h/.cpp` | `llama_moe_cache_prior_mask()`, `prior()`, `is_cached()`, `get_cached_slots()`, `n_expert()`, the `λ` plumbing and the log line |
| `src/llama-graph.h/.cpp` | `llm_graph_input_moe_cache_prior` (the mask input), `build_moe_cache_prior()`, `build_moe_cache_slot_map()` split out of `build_moe_cache_slots()`, and the hook in `build_moe_ffn()` |
| `src/llama-cparams.h`, `src/llama-context.cpp`, `include/llama.h` | `llama_context_params::moe_cache_prior` |
| `common/common.h/.cpp`, `common/arg.cpp` | `--moe-cache-prior`, `LLAMA_ARG_MOE_CACHE_PRIOR` |
| `tests/test-moe-cache-prior.cpp`, `tests/CMakeLists.txt` | the test |
| `tools/{cli,completion,server}/README.md` | the flag table |
