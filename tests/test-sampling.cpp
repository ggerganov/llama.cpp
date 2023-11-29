#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <numeric>
#include <cassert>
#include <vector>
#include <algorithm>

static void dump(const llama_token_data_array * candidates) {
    for (size_t i = 0; i < candidates->size; i++) {
        printf("%d: %f (%f)\n", candidates->data[i].id, candidates->data[i].p, candidates->data[i].logit);
    }
}

#define DUMP(__candidates) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__candidates)); printf("-\n"); } while(0)

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & expected_probs, int k) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        float logit = log(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_top_k(nullptr, &candidates_p, k, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-5);
    }
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & expected_probs, float p) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        float logit = log(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_top_p(nullptr, &candidates_p, p, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_tfs(const std::vector<float> & probs, const std::vector<float> & expected_probs, float z) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        float logit = log(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    DUMP(&candidates_p);
    llama_sample_tail_free(nullptr, &candidates_p, z, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & expected_probs, float p) {
    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        float logit = log(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    DUMP(&candidates_p);
    llama_sample_typical(nullptr, &candidates_p, p, 1);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

static void test_repetition_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    size_t n_vocab = probs.size();
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
        float logit = log(probs[token_id]);
        candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);
    llama_sample_repetition_penalties(nullptr, &candidates_p, (const llama_token *) last_tokens.data(), last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);
    llama_sample_softmax(nullptr, &candidates_p);
    DUMP(&candidates_p);

    GGML_ASSERT(candidates_p.size == expected_probs.size());
    for (size_t i = 0; i < candidates_p.size; i++) {
        GGML_ASSERT(fabs(candidates_p.data[i].p - expected_probs[i]) < 1e-3);
    }
}

int main(void) {
    ggml_time_init();

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f}, 3);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 1);

    test_tfs({0.1f, 0.15f, 0.2f, 0.25f, 0.3f}, {0.3f}, 0.25f);
    test_tfs({0.1f, 0.15f, 0.2f, 0.25f, 0.3f}, {0.3f, 0.25f}, 0.75f);
    test_tfs({0.1f, 0.15f, 0.2f, 0.25f, 0.3f}, {0.3f, 0.25f}, 0.99f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f}, 0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f}, {0.2f, 0.2f, 0.2f}, 0.5f);

    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0.25f, 0.25f, 0.25f, 0.25f, 0},   50.0f, 0.0f, 0.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0.5f, 0.5f, 0, 0, 0},       50.0f, 0.0f, 0.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.5f, 0.5f, 0, 0, 0}, 50.0f, 0.0f, 0.0f);

    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.249997f, 0.249997f, 0.249997f, 0.249997f, 0.000011f}, 1.0f, 5.0f, 5.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.499966f, 0.499966f, 0.000023f, 0.000023f, 0.000023f}, 1.0f, 5.0f, 5.0f);
    test_repetition_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.499977f, 0.499977f, 0.000023f, 0.000023f, 0.000000f}, 1.0f, 5.0f, 5.0f);

    printf("OK\n");

    return 0;
}
