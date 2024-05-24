#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static void dbg_color(const std::string & s, const std::string & fg)
{
    static const std::string kReset = "\033[0m";
    static const std::string kBold[] = { "", "\033[1m" };
    static size_t index = 0;
    std::cout << kBold[index] << fg << s << kReset << std::flush;
    index = 1 - index;
}

static void dbg_accepted(const std::string & accepted)
{
    dbg_color(accepted, /* green */ "\033[32m");
}

static void dbg_default(const std::string & accepted)
{
    dbg_color(accepted, "");
}

static void dbg_rejected(const std::string & rejected)
{
    dbg_color(rejected, /* red */ "\033[31m");
}

template<typename Iterator>
static std::string to_string(llama_context * ctx, Iterator from, Iterator to)
{
    std::string res = "";
    for (auto it = from; it != to; ++it)
    {
        res += llama_token_to_piece(ctx, *it);
    }
    return res;
} 

using llama_tokens = std::vector<llama_token>;

struct speculation_context
{
    llama_tokens candidate;
    int32_t      active_id;
    std::mutex   mtx;
    bool         done;
};

speculation_context spec_ctx;

static void split_done_cb(int split)
{
    if (split == 1 || split == 2)
    {
        std::lock_guard<std::mutex> guard(spec_ctx.mtx);
        spec_ctx.active_id = split - 1;
    }
}

// this ignores all the other sampling criteria
static std::vector<llama_token> greedy_tokens(
        llama_model   * model,
        llama_context * ctx,
        int32_t from_idx,
        int32_t to_idx)
{
    auto n_vocab = llama_n_vocab(model);
    std::vector<llama_token> res;
    if (n_vocab <= 0)
    {
        return res;
    }

    for (int idx = from_idx; idx < to_idx; idx++)
    {
        auto * logits  = llama_get_logits_ith(ctx, idx);
        llama_token new_token_id = 0;
        for (llama_token token_id = 1; token_id < n_vocab; token_id++)
        {
            if (logits[token_id] > logits[new_token_id])
            {
                new_token_id = token_id;
            }
        }

        res.push_back(new_token_id);
    }
    return res;
}

static int speculation(
    std::vector<llama_model *> model,
    speculation_context * spec_ctx,
    std::vector<llama_context *> ctx,
    llama_tokens input /* copy here */) {

    int32_t active = 1;

    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < input.size(); i++)
    {
        llama_batch_add(batch, input[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx[active], batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    int logit_idx = batch.n_tokens - 1;
    llama_tokens local = input;
    size_t match_len;

    // TODO: here we need to not generate too many and wait
    while (true) {
        auto next_tokens = greedy_tokens(model[active], ctx[active], logit_idx, logit_idx + 1);
        if (next_tokens.size() != 1) {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        local.push_back(next_tokens[0]);

        {
            std::lock_guard<std::mutex> _lock(spec_ctx->mtx);
            if (spec_ctx->done)
            {
                break;
            }
            auto& shared = spec_ctx->candidate;
            bool match = true;
            match_len = local.size() - 1;
            for (size_t i = 0; i < std::min(shared.size(), local.size()); i++)
            {
                if (shared[i] != local[i])
                {
                    match = false;
                    match_len = i;
                    // here we need to clear both contexts
                    llama_kv_cache_seq_rm(ctx[0], 0, i, -1);
                    llama_kv_cache_seq_rm(ctx[1], 0, i, -1);
                    break;
                }
            }
            if (match && shared.size() < local.size()) 
            {
                shared = local;
            } 
            else 
            {
                local = shared;
            }
            active = spec_ctx->active_id;
        }

        llama_batch_clear(batch);
        // TODO theoretically this can be empty?
        for (size_t i = match_len; i < local.size(); i++)
        {
            llama_batch_add(batch, local[i], i, { 0 }, true);
        }

        logit_idx = batch.n_tokens - 1;

        if (llama_decode(ctx[active], batch) != 0)
        {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    llama_batch_free(batch);
    return 0;
}

static int target(
    llama_model * model, 
    llama_context * ctx,
    const llama_tokens& input,
    size_t n_predict)
{
    dbg_default(to_string(ctx, input.begin(), input.end()));
    // TODO: batch size 
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < input.size(); i++)
    {
        llama_batch_add(batch, input[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    // how many tokens are currently accepted 
    // TODO: rename to n_accepted
    size_t n_cur = input.size();
    size_t n_decode = 0;

    const auto t_main_start = ggml_time_us();

    // we'll use logits from this position to determine next token
    int logits_from = batch.n_tokens - 1;
    int logits_to   = batch.n_tokens;

    llama_tokens input_seq, next_tokens;
    input_seq.push_back(input.back());

    while (n_decode <= n_predict)
    {
        next_tokens = greedy_tokens(model, ctx, logits_from, logits_to);
        if (next_tokens.size() != input_seq.size())
        {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        size_t next_tokens_pos = n_cur;
        // we always accept at least one new token
        n_cur    += 1;
        n_decode += 1;
        for (size_t i = 0; i + 1 < input_seq.size(); i++)
        {
            if (next_tokens[i] == input_seq[i + 1])
            {
                n_cur += 1;
                n_decode += 1;
            }
            else
            {
                // reject. next_tokens[i] is the last correct one.
                next_tokens.erase(next_tokens.begin() + i + 1, next_tokens.end());
                break;
            }
        }

        // empty the non-matching portion of kv cache. 
        // n_cur is incremented at least once and will be > 0
        llama_kv_cache_seq_rm(ctx, 0, n_cur - 1, -1);

        bool done = false;
        for (size_t i = 0; i < next_tokens.size(); i++)
        {
            // TODO: what should we do here, is this correct
            if (next_tokens[i] == llama_token_eos(model) || llama_token_is_eog(model, next_tokens[i]))
            {
                done = true;
                next_tokens.erase(next_tokens.begin() + i, next_tokens.end());
                break;
            }
        }

        {
            std::lock_guard<std::mutex> _lock(spec_ctx.mtx);
            auto & spec = spec_ctx.candidate;
            size_t n_match = 0;
            for (size_t i = 0; i < next_tokens.size() && i + next_tokens_pos < spec.size(); i++)
            {
                if (next_tokens[i] == spec[i + next_tokens_pos])
                {
                    n_match++;
                }
                else
                {
                    break;
                }
            }

            dbg_accepted(to_string(ctx, spec.begin() + next_tokens_pos, spec.begin() + next_tokens_pos + n_match));
            if (n_match != next_tokens.size())
            {
                dbg_rejected(to_string(ctx, spec.begin() + next_tokens_pos + n_match, spec.end()));
                dbg_default(to_string(ctx, next_tokens.begin() + n_match, next_tokens.end()));
                spec.erase(spec.begin() + next_tokens_pos, spec.end());
                for (const auto tok: next_tokens)
                {
                    spec.push_back(tok);
                }
            }
            input_seq.assign(spec.begin() + n_cur - 1, spec.end());
        }
        if (n_decode >= n_predict || done)
        {
            break;
        }

        llama_batch_clear(batch);
        for (size_t i = 0; i < input_seq.size(); i++)
        {
            llama_batch_add(batch, input_seq[i], n_cur - 1 + i, { 0 }, true);
        }
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        logits_from = 0;
        logits_to   = input_seq.size();
    }

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "decoded %zu tokens in %.2f s, speed: %.2f t/s\n",
            n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);
    fprintf(stderr, "\n");
    {
        std::lock_guard<std::mutex> _lock(spec_ctx.mtx);
        spec_ctx.done = true;
    }

    llama_batch_free(batch);
    return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    std::string draft_rpcs = params.rpc_servers_draft;
    size_t i = draft_rpcs.find(',');
    if (i == std::string::npos || draft_rpcs.find(',', i + 1) != std::string::npos)
    {
        fprintf(stderr, "drpc must contain exactly two servers\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // main model and context
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    params.cb_split_done = split_done_cb;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    llama_tokens input = llama_tokenize(ctx, params.prompt, true);
    spec_ctx.candidate = input;

    // prepare draft model and contexts. No need for two model instances?
    std::vector<llama_model *> draft_models = {nullptr, nullptr};
    std::vector<llama_context *> draft_ctx  = {nullptr, nullptr};

    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    if (params.n_threads_draft > 0) 
    {
        params.n_threads = params.n_threads_draft;
    }
    params.n_threads_batch = params.n_threads_batch_draft;
    
    params.rpc_servers = draft_rpcs.substr(0, i);
    std::tie(draft_models[0], draft_ctx[0]) = llama_init_from_gpt_params(params);
    params.rpc_servers = draft_rpcs.substr(i + 1);
    std::tie(draft_models[1], draft_ctx[1]) = llama_init_from_gpt_params(params);
    std::thread spec_thread = std::thread(speculation, draft_models, &spec_ctx, draft_ctx, input);

    target(model, ctx, input, params.n_predict);

    spec_thread.join();
    
    llama_free(ctx);
    llama_free(draft_ctx[0]);
    llama_free(draft_ctx[1]);

    llama_free_model(model);
    llama_free_model(draft_models[0]);
    llama_free_model(draft_models[1]);

    llama_backend_free();

    return 0;
}
