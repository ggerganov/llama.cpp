#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;

    struct common_sampler * smpl = nullptr;
};

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sparams.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.model_draft.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);
    model_tgt = llama_init_tgt.model;
    ctx_tgt = llama_init_tgt.context;

    // load the draft model
    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    if (params.draft_cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.draft_cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.draft_cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params);
    model_dft = llama_init_dft.model;
    ctx_dft = llama_init_dft.context;

    const bool vocab_type_tgt = llama_vocab_type(model_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(model_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_add_bos_token(model_tgt) != llama_add_bos_token(model_dft) ||
        llama_add_eos_token(model_tgt) != llama_add_eos_token(model_dft) ||
        llama_token_bos(model_tgt) != llama_token_bos(model_dft) ||
        llama_token_eos(model_tgt) != llama_token_eos(model_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
            const char * token_text_dft = llama_token_get_text(model_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sparams);

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    int n_past = inp.size() - 1;

    // init the speculator
    struct common_speculative_params params_spec;
    params_spec.n_draft   = n_draft;
    params_spec.n_min     = 5;
    params_spec.model_dft = model_dft;
    params_spec.ctx_dft   = ctx_dft;

    struct common_speculative * spec = common_speculative_init(params_spec);

    // feed the prompt to the speculator
    //
    // this has to be kept synchronized with the target context
    //
    // TODO: simplify this by moving the context management logic in the common_speculative instance
    //       for example, the common_speculative_add_draft can pass the entire context (or part of it) and the
    //       speculator will automatically compute any new tokens that are not present in its context
    //
    common_speculative_set_prompt(spec, inp.data(), n_input - 1);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // always have a token to evaluate from before
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past, { 0 }, true);

        // optionally, append draft tokens to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        common_speculative_add_draft(spec, batch_tgt, id_last, n_past);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const auto ids = common_speculative_sample(spec, smpl, ctx_tgt);

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_past    += ids.size();
        n_drafted += batch_tgt.n_tokens - 1;
        n_accept  += ids.size() - 1;

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        {
            llama_token id;
            std::string token_str;

            for (size_t i = 0; i < ids.size(); ++i) {
                id = ids[i];

                ++n_predict;

                if (llama_token_is_eog(model_tgt, id)) {
                    has_eos = true;
                    break;
                }

                token_str = common_token_to_piece(ctx_tgt, id);

                if (params.use_color && i + 1 < ids.size()) {
                    LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
                } else {
                    LOG("%s", token_str.c_str());
                }
            }

            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
                break;
            }

            LOG_DBG("accepted %d draft tokens, the last target token is: (%d, '%s')\n", (int) ids.size() - 1, id, token_str.c_str());

            {
                LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

                llama_kv_cache_seq_rm(ctx_tgt, 0, n_past, -1);
                llama_kv_cache_seq_rm(ctx_dft, 0, n_past, -1);
            }

            // remember the last accepted token for the next iteration
            id_last = id;
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
