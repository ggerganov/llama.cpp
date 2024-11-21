#include "speculative.h"

#include "log.h"
#include "common.h"
#include "sampling.h"

struct common_speculative {
    struct common_speculative_params params;

    llama_batch batch_dft;

    struct common_sampler * smpl;

    llama_tokens prompt_last;
};

struct common_speculative * common_speculative_init(struct common_speculative_params params) {
    auto * result = new common_speculative {
        /* .params      = */ params,
        /* .batch_dft   = */ llama_batch_init(llama_n_batch(params.ctx_dft), 0, 1),
        /* .smpl        = */ nullptr,
    };

    // TODO: optimize or pass from outside?
#if 1
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

void common_speculative_add_draft(
        struct common_speculative * spec,
        struct llama_batch & batch_tgt,
        const llama_tokens & prompt,
        llama_token id_last,
        llama_token n_past_tgt) {

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(spec->params.ctx_dft) - spec->params.n_draft;

    const int i_start = std::max<int>(0, (int) prompt.size() - n_ctx);

    for (int i = 0; i < (int) spec->prompt_last.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt.size() &&
               i       + cur < (int) spec->prompt_last.size() &&
               prompt[i_start + cur] == spec->prompt_last[i + cur]) {
            cur++;
        }

        if ((cur >= spec->params.n_reuse || prompt.size() <= n_ctx) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LOG_DBG("%s: reuse_i = %d, reuse_n = %d\n", __func__, reuse_i, reuse_n);

    if (reuse_n == 0) {
        llama_kv_cache_clear(spec->params.ctx_dft);

        spec->prompt_last.clear();
    } else {
        llama_kv_cache_seq_rm (spec->params.ctx_dft, 0, 0, reuse_i);
        llama_kv_cache_seq_rm (spec->params.ctx_dft, 0, reuse_i + reuse_n, -1);
        llama_kv_cache_seq_add(spec->params.ctx_dft, 0, reuse_i, -1, -reuse_i);

        spec->prompt_last.erase(spec->prompt_last.begin(), spec->prompt_last.begin() + reuse_i);
        spec->prompt_last.erase(spec->prompt_last.begin() + reuse_n, spec->prompt_last.end());
    }

    common_batch_clear(spec->batch_dft);

    for (int i = i_start + reuse_n; i < (int) prompt.size(); ++i) {
        //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt[i]);
        common_batch_add(spec->batch_dft, prompt[i], i - i_start, { 0 }, false);

        spec->prompt_last.push_back(prompt[i]);
    }

    const llama_pos n_past = prompt.size() - i_start;

    LOG_DBG("%s: n_past = %d\n", __func__, n_past);

    if (spec->batch_dft.n_tokens > 0) {
        LOG_DBG("%s: draft batch: %s\n", __func__, string_from(spec->params.ctx_dft, spec->batch_dft).c_str());

        llama_decode(spec->params.ctx_dft, spec->batch_dft);
    }

    common_batch_clear(spec->batch_dft);
    common_batch_add  (spec->batch_dft, id_last, n_past, { 0 }, true);

    spec->prompt_last.push_back(id_last);

    LOG_DBG("%s: prompt_last: %s\n", __func__, string_from(spec->params.ctx_dft, spec->prompt_last).c_str());

    llama_decode(spec->params.ctx_dft, spec->batch_dft);

    common_sampler_reset(spec->smpl);

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
        if (cur_p->data[0].p < spec->params.p_min) {
            break;
        }

        common_sampler_accept(spec->smpl, id, true);

        common_batch_add(batch_tgt, id, n_past_tgt + i, { 0 }, true);

        if (batch_tgt.n_tokens > spec->params.n_draft) {
            break;
        }

        common_batch_add(spec->batch_dft, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(spec->params.ctx_dft, spec->batch_dft);

        spec->prompt_last.push_back(id);
    }

    // don't waste time on small batches
    // TODO: do not evaluate the draft model for that many rounds
    if (batch_tgt.n_tokens < spec->params.n_min) {
        batch_tgt.n_tokens = 1;
    }
}
