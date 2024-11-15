#include "speculative.h"

#include "log.h"
#include "common.h"
#include "sampling.h"

#include <vector>

struct seq_draft {
};

struct common_speculative {
    struct common_speculative_params params;

    llama_batch batch_dft;

    struct common_sampler * smpl;

    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
};

struct common_speculative * common_speculative_init(struct common_speculative_params params) {
    auto * result = new common_speculative {
        /* .params      = */ params,
        /* .batch_dft   = */ llama_batch_init(llama_n_batch(params.ctx_dft), 0, 1),
        /* .smpl        = */ nullptr,
        /* .i_batch_tgt = */ {},
        /* .tokens      = */ {},
    };

    // TODO: optimize or pass from outside?
#if 0
    {
        common_sampler_params sparams;
        sparams.no_perf = false;

        sparams.top_k = 40;
        sparams.top_p = 0.9;

        sparams.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = common_sampler_init(params.model_dft, sparams);
    }
#else
    {
        common_sampler_params sparams;
        sparams.no_perf = false;

        sparams.top_k = 10;

        sparams.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        result->smpl = common_sampler_init(params.model_dft, sparams);
    }
#endif

    result->batch_dft = llama_batch_init(llama_n_batch(params.ctx_dft), 0, 1);

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    common_sampler_free(spec->smpl);

    llama_batch_free(spec->batch_dft);

    delete spec;
}

void common_speculative_set_prompt(struct common_speculative * spec, llama_token * tokens, int32_t n_tokens) {
    llama_kv_cache_clear(spec->params.ctx_dft);

    // TODO: error handling
    llama_decode(spec->params.ctx_dft, llama_batch_get_one(tokens, n_tokens));
}

void common_speculative_add_draft(
        struct common_speculative * spec,
        struct llama_batch & batch_tgt,
        llama_token id_last,
        int n_past) {
    spec->tokens.clear();

    spec->i_batch_tgt.clear();
    spec->i_batch_tgt.push_back(0);

    common_sampler_reset(spec->smpl);

    common_batch_clear(spec->batch_dft);
    common_batch_add  (spec->batch_dft, id_last, n_past, { 0 }, true);

    llama_decode(spec->params.ctx_dft, spec->batch_dft);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < spec->params.n_draft; ++i) {
        common_batch_clear(spec->batch_dft);

        common_sampler_sample(spec->smpl, spec->params.ctx_dft, 0, true);

        const auto * cur_p = common_sampler_get_candidates(spec->smpl);

        for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
            LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                    k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(spec->params.ctx_dft, cur_p->data[k].id).c_str());
        }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        // only collect very high-confidence draft tokens
        if (cur_p->data[0].p < 0.75 && spec->tokens.size() >= 0) {
            break;
        }

        common_sampler_accept(spec->smpl, id, true);

        spec->tokens.push_back(id);

        // add unique drafted tokens to the target batch
        spec->i_batch_tgt.push_back(batch_tgt.n_tokens);

        common_batch_add(batch_tgt, id, n_past + i + 1, { 0 }, true);

        if (batch_tgt.n_tokens > spec->params.n_draft) {
            break;
        }

        common_batch_add(spec->batch_dft, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(spec->params.ctx_dft, spec->batch_dft);
    }

    // don't waste time on small batches
    // TODO: do not evaluate the draft model for tha many rounds
    if (batch_tgt.n_tokens < spec->params.n_min) {
        batch_tgt.n_tokens = 1;
        spec->tokens.resize(0);
        spec->i_batch_tgt.resize(1);
    }

    // print current draft sequences
    LOG_DBG("draft %s\n", string_from(spec->params.ctx_dft, spec->tokens).c_str());
}

std::vector<llama_token> common_speculative_sample(
        struct common_speculative * spec,
        struct common_sampler * smpl,
        struct llama_context * ctx_tgt) {
    return common_sampler_sample_n(smpl, ctx_tgt, spec->i_batch_tgt, spec->tokens);
}
