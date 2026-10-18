#!/usr/bin/env python3
"""
Simulation of the cache-aware MoE routing ("cache prior") that this llama.cpp fork adds.

Model of the world (mirrors the llama.cpp MoE cache, src/llama-moe-cache.cpp):
  * a MoE layer has n_expert experts, the router picks the top n_expert_used of them for every token
  * the experts live in host memory, only n_slots of them fit in the device cache (LRU eviction)
  * a selected expert that is not in the cache is a miss: it has to be uploaded from the host memory

Routing (arXiv:2412.00099, section 3.3 "Cache-Prior Reranking"), as implemented by
llm_graph_context::build_moe_cache_prior:
  * m_t : bitmask of the experts that are in the cache when token t is routed
  * the scores s of the router are biased:  s' = s + lambda * delta * m_t
  * delta is the score range of the token, here its largest score (clamped at 0)
  * s' is used ONLY to rank the experts, the weights of the experts still come from s

Router scores are generated with temporal locality: a per-layer preference that drifts slowly
(AR(1)) plus a per-token i.i.d. component. That is what makes an LRU cache useful at all, and it
is what the paper measures as the "temporal consistency of expert selections" (section 4.6).
The lower --drift, the stronger the locality.

usage:
  python3 simulate_cache_prior.py                          # 64 experts, top-8, cache = 1/2
  python3 simulate_cache_prior.py --drift 0.01             # stronger temporal locality
  python3 simulate_cache_prior.py --n-expert 8 --n-used 2  # Mixtral-like
"""

import argparse
import math
import random
from collections import OrderedDict


class LRU:
    """The LRU of llama_moe_cache: a slot per expert, evict the least recently used."""

    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.order = OrderedDict()  # expert -> None, least recently used first

    def state(self):
        return set(self.order.keys())

    def access(self, experts):
        """Touch the selected experts, upload the misses. Returns (n_hit, n_miss)."""
        n_hit = n_miss = 0
        for e in experts:  # the hits first, like moe_cache_lru::plan
            if e in self.order:
                self.order.move_to_end(e)
                n_hit += 1
        for e in experts:
            if e in self.order:
                continue
            if len(self.order) >= self.n_slots:
                self.order.popitem(last=False)
            self.order[e] = None
            n_miss += 1
        return n_hit, n_miss


def top_k(scores, k):
    """Stable top-k, highest score first, ties broken by expert id."""
    return sorted(range(len(scores)), key=lambda e: (-scores[e], e))[:k]


def route(scores, cache_state, n_used, lam, top_j=1, delta_mode="max"):
    """The expert selection of build_moe_ffn, without and with the cache prior."""
    orig = top_k(scores, n_used)
    if lam <= 0.0:
        return orig, orig

    if delta_mode == "max":
        delta = max(0.0, max(scores))       # what build_moe_cache_prior extracts
    elif delta_mode == "range":
        delta = max(scores) - min(scores)   # equation (10) of the paper
    else:
        delta = 1.0

    bias = lam * delta

    # m~ = m | top-J of section 3.3: the top-J experts of the router are promoted together with the
    # cached ones, and the top-J of a token is the threshold { e : s[e] >= s[J] }, which is what the
    # graph computes because ggml has no scatter over the rows an index tensor selects per token.
    # No cap on the bias is needed: an expert outside of the top-J scores at most s[J], so with the
    # same bias on both sides it cannot overtake one.
    promoted = set(cache_state)
    j = min(top_j, len(scores) - 1)
    if j > 0:
        promoted.update(orig[:j])

    biased = [s + (bias if e in promoted else 0.0) for e, s in enumerate(scores)]
    return orig, top_k(biased, n_used)


def run(args, lam):
    """One pass over the tokens with a fixed lambda: the routing feeds back into the cache."""
    rnd = random.Random(args.seed)
    n_layer, n_expert, n_used = args.n_layer, args.n_expert, args.n_used
    n_slots = max(int(round(n_expert * args.cache_frac)), n_used)

    lru = [LRU(n_slots) for _ in range(n_layer)]
    # slowly drifting per-layer preference (temporal locality) + per-token noise
    pref = [[rnd.gauss(0.0, 1.0) for _ in range(n_expert)] for _ in range(n_layer)]

    st = dict(hit=0, miss=0, top1=0, n=0, same=0, same_n=0, changed=0)

    for _ in range(args.n_tokens):
        for il in range(n_layer):
            p = pref[il]
            for e in range(n_expert):
                p[e] = (1.0 - args.drift) * p[e] + args.drift * rnd.gauss(0.0, 1.0)
            scores = [p[e] + args.noise * rnd.gauss(0.0, 1.0) for e in range(n_expert)]

            orig, sel = route(scores, lru[il].state(), n_used, lam, args.top_j, args.delta_mode)
            n_hit, n_miss = lru[il].access(sel)

            st["hit"] += n_hit
            st["miss"] += n_miss
            st["top1"] += orig[0] == sel[0]
            st["n"] += 1
            st["same"] += len(set(orig) & set(sel))
            st["same_n"] += n_used
            st["changed"] += set(orig) != set(sel)

    return st, n_slots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-expert", type=int, default=64)
    ap.add_argument("--n-used", type=int, default=8)
    ap.add_argument("--cache-frac", type=float, default=0.5, help="cached experts / experts, the paper uses 1/2 and 1/4")
    ap.add_argument("--n-tokens", type=int, default=2000)
    ap.add_argument("--drift", type=float, default=0.05, help="how fast the layer preference drifts (temporal locality)")
    ap.add_argument("--noise", type=float, default=1.0, help="per-token router noise")
    ap.add_argument("--top-j", type=int, default=1, help="J: the top-J experts of the router stay selected, 0 = no guarantee")
    ap.add_argument("--delta-mode", choices=["max", "range", "one"], default="max")
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(f"layers={args.n_layer} experts={args.n_expert} top-{args.n_used} tokens={args.n_tokens} "
          f"cache={args.cache_frac:.2f} drift={args.drift} noise={args.noise} J={args.top_j} delta={args.delta_mode}")
    print()
    print(f"{'lambda':>7} {'slots':>6} {'miss rate':>10} {'vs base':>9} {'top-1 match':>12} "
          f"{'expert match':>13} {'changed sel':>12}")

    base = None
    for lam in [0.0] + args.lambdas:
        st, n_slots = run(args, lam)
        n = st["hit"] + st["miss"]
        miss = 100.0 * st["miss"] / n
        if base is None:
            base, vs = miss, "  (base)"
        else:
            vs = f"{100.0 * (miss - base) / base:+8.1f}%"
        print(f"{lam:7.2f} {n_slots:6d} {miss:9.2f}% {vs:>9} "
              f"{100.0 * st['top1'] / st['n']:11.2f}% "
              f"{100.0 * st['same'] / st['same_n']:12.2f}% "
              f"{100.0 * st['changed'] / st['n']:11.2f}%")


if __name__ == "__main__":
    main()
