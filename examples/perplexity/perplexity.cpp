#include "common.h"
#include "llama.h"

#include <cmath>
#include <ctime>

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}

void perplexity(llama_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    auto tokens = ::llama_tokenize(ctx, params.prompt, true);

    int count = 0;
    int seq_count = tokens.size() / params.n_ctx;
    int n_vocab = llama_n_vocab(ctx);

    double nll = 0.0;
    fprintf(stderr, "%s : calculating perplexity over %d chunks, batch_size=%d\n", __func__, seq_count, params.n_batch);

    for (int i = 0; i < seq_count; ++i) {
        int start = i * params.n_ctx;
        int end = start + params.n_ctx;

        std::vector<float> logits;
        int num_batches = (params.n_ctx + params.n_batch - 1) / params.n_batch;
        auto start_t = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < num_batches; ++j) {
            int batch_start = start + j * params.n_batch;
            int batch_size = std::min(end - batch_start, params.n_batch);
            if (llama_eval(ctx, tokens.data() + batch_start, batch_size, j * params.n_batch, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return;
            }
            auto batch_logits = llama_get_logits(ctx);
            logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
        }
        auto end_t = std::chrono::high_resolution_clock::now();
        if (i == 0) {
            const float seconds = std::chrono::duration<float>(end_t - start_t).count();
            printf("%.2f seconds per pass - ETA ", seconds);
            int total_seconds = (int)(seconds * seq_count);
            if (total_seconds >= 60*60) {
                printf("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            printf("%d minutes\n", total_seconds / 60);
        }
        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.
        for (int j = std::min(512, params.n_ctx / 2); j < params.n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            std::vector<float> tok_logits(
                logits.begin() + j * n_vocab,
                logits.begin() + (j + 1) * n_vocab);
            float prob = softmax(tok_logits)[tokens[start + j + 1]];
            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        fflush(stdout);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    params.n_batch = 512;
    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.perplexity = true;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.logits_all = params.perplexity;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;
        lparams.embedding  = params.embedding;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    if (!params.lora_adapter.empty()) {
        int err = llama_apply_lora_from_file(ctx,
                                             params.lora_adapter.c_str(),
                                             params.lora_base.empty() ? NULL : params.lora_base.c_str(),
                                             params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    perplexity(ctx, params);

    llama_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
