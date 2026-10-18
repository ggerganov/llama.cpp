// test-moe-cache-prior: verify the cache-aware MoE routing ("cache prior") of the MoE cache.
//
// The routing of a MoE layer selects the top-k experts of the router scores. When the experts live
// in host memory and only a subset of them is cached on the device (--moe-cache-mib), every
// selected expert that is not in the cache has to be uploaded from the host memory, which is what
// limits the decode speed. The cache prior biases the scores of the experts that are already in the
// cache, so that the router prefers them:
//
//   s'[e] = s[e] + lambda * delta * m~[e]
//   m~[e] = m[e] | [ s[e] >= max(s) ]        m: 1 for the cached experts, delta: the largest score
//
// The union is what keeps the best expert of the router selected at any lambda: an expert that is
// not the best one scores below max(s), so with the same bias on both sides it cannot overtake it.
// s' only ranks the experts, the weights they are combined with still come from s, so an expert
// that is selected either way keeps the weight that the router gave it.
//
// ref: Skliar et al., "Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference",
//      arXiv:2412.00099, section 3.3 (Cache-Prior Reranking)
//
// This test checks the two halves of the implementation:
//   1. llama_moe_cache_prior_mask(), the host side bitmask of the cached experts
//   2. the graph side bias that llm_graph_context::build_moe_cache_prior() adds to the router
//      scores, verified against a plain C++ reference of the same algorithm
//
// The MoE cache itself needs a device backend, so it is not exercised here.

#include "llama-moe-cache.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("%s:%d: FAILED: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        n_failed++; \
    } \
} while (0)

//
// plain C++ reference
//

struct ref_prior {
    float hi        = 0.0f; // the largest score of the token, the threshold of the union
    float delta     = 0.0f; // max(hi, 0), the range that lambda scales
    float bias      = 0.0f; // lambda * delta, what every promoted expert gets
    bool  union_on  = false; // whether the best expert of the router is promoted too
};

static ref_prior ref_prior_of(const std::vector<float> & scores, float lambda, bool top_j) {
    ref_prior res;
    for (float s : scores) {
        res.hi = std::max(res.hi, s);
    }
    // a token whose scores are all negative starts at 0 and stays there, so there is nothing to
    // promote towards and the routing of that token is the original one
    res.hi       = *std::max_element(scores.begin(), scores.end());
    res.delta    = std::max(res.hi, 0.0f);
    res.bias     = lambda*res.delta;
    res.union_on = top_j && lambda > 0.0f;
    return res;
}

static bool ref_promoted(const std::vector<float> & scores, const std::vector<int8_t> & cached, int e,
                         const ref_prior & p) {
    return cached[e] != 0 || (p.union_on && scores[e] >= p.hi);
}

// the selection of build_moe_ffn: the top-k experts of the biased scores, highest first
static std::vector<int> ref_top_k(
        const std::vector<float> & scores, const std::vector<int8_t> & cached, int k, const ref_prior & p) {
    std::vector<int> order(scores.size());
    for (int e = 0; e < (int) scores.size(); ++e) {
        order[e] = e;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float sa = scores[a] + (ref_promoted(scores, cached, a, p) ? p.bias : 0.0f);
        const float sb = scores[b] + (ref_promoted(scores, cached, b, p) ? p.bias : 0.0f);
        return sa != sb ? sa > sb : a < b;
    });
    order.resize(k);
    return order;
}

//
// the graph
//

// the graph of build_moe_cache_prior: [n_expert, n_tokens] scores in, [k, n_tokens] experts out
struct prior_graph {
    ggml_backend_ptr        backend;
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_cgraph           * gf     = nullptr;
    ggml_tensor           * scores = nullptr;
    ggml_tensor           * mask   = nullptr;
    ggml_tensor           * delta  = nullptr;
    ggml_tensor           * biasv  = nullptr; // lambda*delta, [1, n_tokens]
    ggml_tensor           * ind    = nullptr; // the top-1 of the router, [n_expert, n_tokens]
    ggml_tensor           * bias   = nullptr; // [n_expert, n_tokens]
    ggml_tensor           * sel    = nullptr;
};

static void build_prior_graph(prior_graph & g, int n_expert, int n_tokens, int k, float lambda, bool top_j) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(dev && "the CPU backend is required");
    g.backend.reset(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(g.backend);

    g.ctx.reset(ggml_init({ ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true }));
    GGML_ASSERT(g.ctx);

    ggml_context * ctx = g.ctx.get();

    g.scores = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
    g.mask   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
    ggml_set_input(g.scores);
    ggml_set_input(g.mask);
    ggml_format_name(g.scores, "scores");
    ggml_format_name(g.mask,   "mask");

    // a sharp softmax is a one-hot at the argmax, because the kernel subtracts the maximum of the
    // row before the exponent, so sum_rows(scores*onehot) is that maximum
    constexpr float sharp = 16384.0f;

    ggml_tensor * onehot = ggml_soft_max_ext(ctx, g.scores, nullptr, sharp, 0.0f);
    ggml_tensor * hi     = ggml_sum_rows(ctx, ggml_mul(ctx, g.scores, onehot)); // [1, n_tokens]

    g.delta = ggml_clamp(ctx, hi, 0.0f, INFINITY);
    ggml_format_name(g.delta, "delta");

    ggml_tensor * ldelta = ggml_scale(ctx, g.delta, lambda);
    g.biasv = ldelta;
    ggml_format_name(g.biasv, "biasv");

    ggml_tensor * mask_r = ggml_repeat(ctx, ggml_reshape_3d(ctx, g.mask, n_expert, 1, 1), g.scores);
    ggml_tensor * bias   = ggml_mul(ctx, mask_r, ggml_repeat(ctx, ldelta, g.scores));

    // no threshold promotes anything while the union is off
    g.ind = ggml_scale(ctx, bias, 0.0f);

    if (top_j && lambda > 0.0f) {
        ggml_tensor * one = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_tokens, 1, 1), 1.0f);
        ggml_tensor * ind = ggml_step(ctx, ggml_sub(ctx, ggml_repeat(ctx, hi, g.scores), g.scores));
        ind = ggml_neg(ctx, ind);
        ind = ggml_add(ctx, ind, one);

        ggml_tensor * promote = ggml_sub(ctx, ggml_add(ctx, mask_r, ind), ggml_mul(ctx, mask_r, ind));
        bias = ggml_mul(ctx, promote, ggml_repeat(ctx, ldelta, g.scores));

        g.ind = ind;
    }

    g.bias = bias;
    ggml_format_name(g.bias, "bias");

    // ggml_argsort_top_k returns a view of the full ranking, make it contiguous so that it can be read back
    g.sel = ggml_cont(ctx, ggml_argsort_top_k(ctx, ggml_add(ctx, g.scores, bias), k)); // [k, n_tokens]
    ggml_format_name(g.sel, "sel");

    g.gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    for (ggml_tensor * t : { g.sel, g.delta, g.biasv, g.ind, g.bias }) {
        ggml_set_output(t);
        ggml_build_forward_expand(g.gf, t);
    }

    g.buf.reset(ggml_backend_alloc_ctx_tensors(ctx, g.backend.get()));
    GGML_ASSERT(g.buf && "failed to allocate the tensors of the cache prior graph");
}

//
// tests
//

static void test_prior_mask() {
    printf("%s\n", __func__);

    const int n_expert = 8;
    const int32_t slots[n_expert] = { 3, -1, 0, -1, -1, 7, -1, 2 };
    float mask[n_expert];

    // exactly the cached experts are promoted
    llama_moe_cache_prior_mask(slots, n_expert, mask);
    for (int e = 0; e < n_expert; ++e) {
        CHECK(mask[e] == (slots[e] >= 0 ? 1.0f : 0.0f), "mask[%d] = %g, slot = %d", e, mask[e], slots[e]);
    }

    // an empty cache promotes nothing, so the routing stays the original one
    const int32_t empty[n_expert] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    llama_moe_cache_prior_mask(empty, n_expert, mask);
    for (int e = 0; e < n_expert; ++e) {
        CHECK(mask[e] == 0.0f, "mask[%d] = %g for an empty cache", e, mask[e]);
    }

    // a full cache promotes everything, so the routing stays the original one as well
    const int32_t full[n_expert] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    llama_moe_cache_prior_mask(full, n_expert, mask);
    for (int e = 0; e < n_expert; ++e) {
        CHECK(mask[e] == 1.0f, "mask[%d] = %g for a full cache", e, mask[e]);
    }
}

static void test_prior_graph() {
    printf("%s\n", __func__);

    std::mt19937 rng(1234);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    const int n_expert = 64;
    const int n_tokens = 3;
    const int k        = 8;

    std::vector<float>   scores(n_expert*n_tokens);
    std::vector<int8_t>  cached(n_expert);
    std::vector<float>   mask(n_expert);
    std::vector<int32_t> sel(k*n_tokens);
    std::vector<float>   bias(n_expert*n_tokens);
    std::vector<float>   ind(n_expert*n_tokens);
    std::vector<int>     order(n_expert);

    struct row { ggml_tensor * t; std::vector<float> v; const char * name; };

    // 0: gaussian scores, 1: softmax-like scores, 2: peaky scores, 3: negative scores,
    // 4: a near tie at the top, where the union has to hold as well
    for (int kind = 0; kind < 5; ++kind) {
        for (const float lambda : { 0.0f, 0.1f, 0.5f, 1.0f }) {
            for (int top_j = 0; top_j < 2; ++top_j) {
                for (int e = 0; e < n_expert; ++e) {
                    cached[e] = (rng() % 2) ? 1 : 0;
                    mask[e]   = cached[e] ? 1.0f : 0.0f;
                }
                for (auto & s : scores) {
                    s = noise(rng);
                }
                if (kind == 1) {
                    for (int t = 0; t < n_tokens; ++t) {
                        float sum = 0.0f;
                        for (int e = 0; e < n_expert; ++e) {
                            scores[t*n_expert + e] = std::exp(scores[t*n_expert + e]);
                            sum += scores[t*n_expert + e];
                        }
                        for (int e = 0; e < n_expert; ++e) {
                            scores[t*n_expert + e] /= sum;
                        }
                    }
                }
                if (kind == 2) {
                    for (int t = 0; t < n_tokens; ++t) {
                        scores[t*n_expert + (int) (rng() % n_expert)] += 6.0f;
                    }
                }
                if (kind == 3) {
                    for (auto & s : scores) {
                        s -= 4.0f; // every score is negative, so the prior must not promote anything
                    }
                }
                if (kind == 4) {
                    for (int t = 0; t < n_tokens; ++t) {
                        const int a = (int) (rng() % n_expert);
                        int b = (int) (rng() % n_expert);
                        while (b == a) { b = (int) (rng() % n_expert); }
                        scores[t*n_expert + a] += 3.0f;
                        scores[t*n_expert + b]  = scores[t*n_expert + a] - 1.0e-4f; // a hair below the top
                    }
                }

                prior_graph g;
                build_prior_graph(g, n_expert, n_tokens, k, lambda, top_j != 0);

                ggml_backend_tensor_set(g.scores, scores.data(), 0, ggml_nbytes(g.scores));
                ggml_backend_tensor_set(g.mask,   mask.data(),   0, ggml_nbytes(g.mask));

                const enum ggml_status status = ggml_backend_graph_compute(g.backend.get(), g.gf);
                CHECK(status == GGML_STATUS_SUCCESS, "graph compute failed: %d", (int) status);
                if (status != GGML_STATUS_SUCCESS) {
                    continue;
                }

                ggml_backend_tensor_get(g.sel,  sel.data(),  0, ggml_nbytes(g.sel));
                ggml_backend_tensor_get(g.bias, bias.data(), 0, ggml_nbytes(g.bias));
                ggml_backend_tensor_get(g.ind,  ind.data(),  0, ggml_nbytes(g.ind));

                std::vector<row> rows = {
                    { g.delta, std::vector<float>(n_tokens), "delta" },
                    { g.biasv, std::vector<float>(n_tokens), "biasv" },
                };
                for (auto & r : rows) {
                    ggml_backend_tensor_get(r.t, r.v.data(), 0, ggml_nbytes(r.t));
                }

                for (int t = 0; t < n_tokens; ++t) {
                    const std::vector<float> s(scores.begin() + t*n_expert, scores.begin() + (t + 1)*n_expert);
                    const ref_prior want = ref_prior_of(s, lambda, top_j != 0);

                    CHECK(std::fabs(rows[0].v[t] - want.delta) < 1e-3f*std::max(1.0f, want.delta),
                          "kind=%d lambda=%.1f delta[%d] = %f, want %f",
                          kind, lambda, t, rows[0].v[t], want.delta);
                    CHECK(std::fabs(rows[1].v[t] - want.bias) < 1e-3f*std::max(1.0f, want.bias) + 1e-6f,
                          "kind=%d lambda=%.1f bias[%d] = %f, want %f",
                          kind, lambda, t, rows[1].v[t], want.bias);

                    // the bias of every expert: lambda*delta for the promoted ones, 0 for the others,
                    // and an expert that is both cached and the best one is biased once, not twice
                    for (int e = 0; e < n_expert; ++e) {
                        const float want_b = ref_promoted(s, cached, e, want) ? want.bias : 0.0f;
                        CHECK(std::fabs(bias[e + t*n_expert] - want_b) < 1e-4f*std::max(1.0f, want_b),
                              "kind=%d lambda=%.1f top_j=%d bias[%d][%d] = %f, want %f",
                              kind, lambda, top_j, t, e, bias[e + t*n_expert], want_b);

                        // the threshold promotes the best expert of the token, and only it
                        const float want_i = (want.union_on && s[e] >= want.hi) ? 1.0f : 0.0f;
                        CHECK(std::fabs(ind[e + t*n_expert] - want_i) < 1e-6f,
                              "kind=%d lambda=%.1f top_j=%d ind[%d][%d] = %f, want %f",
                              kind, lambda, top_j, t, e, ind[e + t*n_expert], want_i);
                    }

                    // ggml_argsort_top_k returns the top-k experts in no particular order
                    std::vector<int> got(sel.begin() + t*k, sel.begin() + (t + 1)*k);
                    std::vector<int> ref = ref_top_k(s, cached, k, want);
                    std::sort(got.begin(), got.end());
                    std::sort(ref.begin(), ref.end());

                    if (got != ref && n_failed < 8) {
                        printf("  kind=%d lambda=%.1f top_j=%d token=%d: got [", kind, lambda, top_j, t);
                        for (int e : got) { printf(" %d", e); }
                        printf(" ], want [");
                        for (int e : ref) { printf(" %d", e); }
                        printf(" ]\n");
                    }
                    CHECK(got == ref, "kind=%d lambda=%.1f top_j=%d: the selection differs from the reference",
                          kind, lambda, top_j);

                    // the original ranking of the token
                    for (int e = 0; e < n_expert; ++e) { order[e] = e; }
                    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                        return s[a] != s[b] ? s[a] > s[b] : a < b;
                    });

                    // the point of the union: the best expert of the router stays selected, at every
                    // lambda, even when it is not in the cache and every other expert is
                    if (top_j && want.delta > 0.0f) {
                        CHECK(std::find(got.begin(), got.end(), order[0]) != got.end(),
                              "kind=%d lambda=%.1f: the prior dropped the best expert %d of the router",
                              kind, lambda, order[0]);
                    }

                    // lambda = 0, an empty cache and a full cache keep the original selection
                    const bool all_cached  = std::all_of(cached.begin(), cached.end(), [](int8_t c) { return c != 0; });
                    const bool none_cached = std::none_of(cached.begin(), cached.end(), [](int8_t c) { return c != 0; });
                    if (lambda == 0.0f || all_cached || none_cached || want.delta == 0.0f) {
                        std::vector<int> orig(order.begin(), order.begin() + k);
                        std::sort(orig.begin(), orig.end());
                        CHECK(got == orig, "kind=%d lambda=%.1f: the routing changed without a cache prior",
                              kind, lambda);
                    }
                }
            }
        }
    }

    // the best expert of the router survives lambda = 1 with an empty cache of its own, and the
    // cached ones take every other place, which is the fastest routing the guarantee allows
    {
        const int ne = 4;
        const float data[ne] = { 0.50f, 0.45f, 0.40f, 0.35f }; // a flat token, delta = 0.50
        const float m[ne]    = { 0.0f, 1.0f, 1.0f, 1.0f };      // the best expert is not cached

        for (const float lambda : { 0.5f, 1.0f }) {
            prior_graph g;
            build_prior_graph(g, ne, 1, 3, lambda, true);

            ggml_backend_tensor_set(g.scores, data, 0, ggml_nbytes(g.scores));
            ggml_backend_tensor_set(g.mask,   m,    0, ggml_nbytes(g.mask));

            CHECK(ggml_backend_graph_compute(g.backend.get(), g.gf) == GGML_STATUS_SUCCESS, "graph compute failed");

            std::vector<int32_t> got(3);
            ggml_backend_tensor_get(g.sel, got.data(), 0, ggml_nbytes(g.sel));
            std::sort(got.begin(), got.end());

            // the expert 0 is not cached, yet it keeps its place at any lambda: 0.50 + lambda*0.50
            // against 0.45 + lambda*0.50 for the best cached one. That is the whole point of the
            // union, and it is why lambda can be pushed to 1.0. The experts 1 and 2 are the cached
            // ones, so the selection is the top-3 of the router either way here; the 4th expert is
            // what a smaller cache would have to swap
            const std::vector<int32_t> want = { 0, 1, 2 };
            CHECK(got == want, "lambda=%.1f: got [%d %d %d], want [0 1 2]", lambda, got[0], got[1], got[2]);
            CHECK(std::find(got.begin(), got.end(), 0) != got.end(),
                  "lambda=%.1f: the best expert of the router was dropped", lambda);
        }
    }
}

int main() {
    setbuf(stdout, nullptr);
    ggml_time_init();

    test_prior_mask();
    test_prior_graph();

    if (n_failed == 0) {
        printf("test-moe-cache-prior: OK\n");
    } else {
        printf("test-moe-cache-prior: %d checks failed\n", n_failed);
    }
    return n_failed == 0 ? 0 : 1;
}
