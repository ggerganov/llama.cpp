#include "common.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static void dbg_color(const std::string & s, const std::string & fg = "")
{
    static const std::string kReset = "\033[0m";
    static const std::string kBold[] = { "", "\033[1m" };
    static size_t index = 0;
    std::cout << kBold[index] << fg << s << kReset << std::flush;
    index = 1 - index;
}

template<typename iter_t>
static std::string to_string(llama_context * ctx, iter_t from, iter_t to)
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
    int32_t      vacant_id; // not running main model
    std::mutex   mtx;
    bool         done;
};

static void split_done_cb(int split, void * p_spec_ctx)
{
    if (split == 1 || split == 2)
    {
        auto * spec_ctx = static_cast<speculation_context*>(p_spec_ctx);
        std::lock_guard<std::mutex> guard(spec_ctx->mtx);
        spec_ctx->vacant_id = split - 1;
    }
}

// this ignores all the other sampling criteria
static llama_tokens greedy_tokens(llama_model * model, llama_context * ctx, int32_t from, int32_t to)
{
    auto n_vocab = llama_n_vocab(model);
    std::vector<llama_token> res;

    for (int idx = from; idx < to; idx++)
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

template<typename iter_t>
static int decode(llama_context * ctx, iter_t from, iter_t to, int offset, bool all_logits, llama_batch & batch)
{
    llama_batch_clear(batch);
    size_t i = offset;
    for (auto it = from; it != to; ++it)
    {
        llama_batch_add(batch, *it, i++, { 0 }, all_logits);
    }
    batch.logits[batch.n_tokens - 1] = true;
    int res = 0;
    if (llama_decode(ctx, batch) != 0)
    {
        fprintf(stderr, "llama_decode() failed\n");
        res = 1;
    }
    return res;
}

static int speculation(
    llama_model * model,
    speculation_context * spec_ctx,
    llama_context * ctx,
    const llama_tokens & input) {

    // TODO: check that input is non-empty
    llama_batch batch = llama_batch_init(512, 0, 1);
    decode(ctx, input.begin(), input.end(), 0, false, batch);

    int logit_idx = input.size() - 1;
    llama_tokens local = input;
    size_t match_len;

    // TODO: here we need to not generate too many and wait
    while (true) 
    {
        // TODO: cond var instead
        bool wait = false;
        {
            std::lock_guard<std::mutex> g(spec_ctx->mtx);
            if (spec_ctx->done)
            {
                break;
            }
            if (spec_ctx->vacant_id != 0)
            {
                wait = true;
            }
        }
        if (wait)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds{5});
            continue;
        }

        auto next_tokens = greedy_tokens(model, ctx, logit_idx, logit_idx + 1);
        if (next_tokens.size() != 1)
        {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        local.push_back(next_tokens[0]);
        {
            std::lock_guard<std::mutex> _lock(spec_ctx->mtx);
            auto& shared = spec_ctx->candidate;
            bool match = true;
            match_len = local.size() - 1;
            for (size_t i = 0; i < std::min(shared.size(), local.size()); i++)
            {
                if (shared[i] != local[i])
                {
                    match = false;
                    match_len = i;
                    llama_kv_cache_seq_rm(ctx, 0, i, -1);
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
        }

        decode(ctx, local.begin() + match_len, local.end(), match_len, false, batch);
        logit_idx = local.size() - match_len - 1;
    }

    llama_batch_free(batch);
    return 0;
}

static int target(
    llama_model * model,
    speculation_context * spec_ctx,
    llama_context * ctx,
    const llama_tokens& input,
    size_t n_predict)
{
    dbg_color(to_string(ctx, input.begin(), input.end()));

    llama_batch batch = llama_batch_init(512, 0, 1);
    decode(ctx, input.begin(), input.end(), 0, false, batch);

    size_t n_accepted = input.size();
    size_t n_decoded  = 0;

    const auto t_main_start = ggml_time_us();

    // we'll use logits from this position to determine next token
    int logits_from = input.size() - 1;
    int logits_to   = input.size();

    llama_tokens input_seq, next_tokens;
    input_seq.push_back(input.back());

    while (n_decoded < n_predict)
    {
        next_tokens = greedy_tokens(model, ctx, logits_from, logits_to);
        if (next_tokens.size() != input_seq.size())
        {
            fprintf(stderr, "invalid next tokens\n");
            return 1;
        }

        size_t next_tokens_pos = n_accepted;
        // we always accept at least one new token
        n_accepted += 1;
        n_decoded  += 1;
        for (size_t i = 0; i + 1 < input_seq.size(); i++)
        {
            if (next_tokens[i] == input_seq[i + 1])
            {
                n_accepted += 1;
                n_decoded  += 1;
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
        llama_kv_cache_seq_rm(ctx, 0, n_accepted - 1, -1);

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
            std::lock_guard<std::mutex> _lock(spec_ctx->mtx);
            auto & spec = spec_ctx->candidate;
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

            dbg_color(to_string(ctx, spec.begin() + next_tokens_pos, spec.begin() + next_tokens_pos + n_match), /* green */ "\033[32m");
            if (n_match != next_tokens.size())
            {
                dbg_color(to_string(ctx, spec.begin() + next_tokens_pos + n_match, spec.end()), /* red */ "\033[31m");
                dbg_color(to_string(ctx, next_tokens.begin() + n_match, next_tokens.end()));
                spec.erase(spec.begin() + next_tokens_pos, spec.end());
                for (const auto tok: next_tokens)
                {
                    spec.push_back(tok);
                }
            }
            input_seq.assign(spec.begin() + n_accepted - 1, spec.end());
        }
        if (n_decoded >= n_predict || done)
        {
            break;
        }

        decode(ctx, input_seq.begin(), input_seq.end(), n_accepted - 1, true, batch);

        logits_from = 0;
        logits_to   = input_seq.size();
    }

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "decoded %zu tokens in %.2f s, speed: %.2f t/s\n",
            n_decoded, (t_main_end - t_main_start) / 1000000.0f, n_decoded / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);
    fprintf(stderr, "\n");
    {
        std::lock_guard<std::mutex> _lock(spec_ctx->mtx);
        spec_ctx->done = true;
    }

    llama_batch_free(batch);
    return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    if (params.seed == LLAMA_DEFAULT_SEED)
    {
        params.seed = time(NULL);
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    speculation_context spec_ctx;

    // main model and context
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    params.cb_split_done = split_done_cb;
    params.cb_split_done_user_data = &spec_ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    llama_tokens input = llama_tokenize(ctx, params.prompt, true);
    spec_ctx.candidate = input;

    // prepare draft model and contexts.
    llama_model * draft_model = nullptr;
    llama_context * draft_ctx = nullptr;

    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    if (params.n_threads_draft > 0) 
    {
        params.n_threads = params.n_threads_draft;
    }
    params.n_threads_batch = params.n_threads_batch_draft;

    params.cb_split_done = nullptr;
    params.rpc_servers = params.rpc_servers_draft;
    std::tie(draft_model, draft_ctx) = llama_init_from_gpt_params(params);
    std::thread spec_thread = std::thread(speculation, draft_model, &spec_ctx, draft_ctx, input);

    target(model, &spec_ctx, ctx, input, params.n_predict);

    spec_thread.join();
    
    llama_free(ctx);
    llama_free(draft_ctx);

    llama_free_model(model);
    llama_free_model(draft_model);

    llama_backend_free();

    return 0;
}