// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"

#include "console.h"
#include "llama.h"
#include "grammar-parser.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define SI_DUMP_SEQUENCES_INTERVAL 40

static std::atomic<bool> interrupted {false};
static std::atomic<bool> done {false};

typedef struct tokens_chunk {
    bool is_input;
    size_t consumed;
    std::vector<llama_token> tokens;

    tokens_chunk(const bool is_input = false, const size_t consumed = 0, const std::vector<llama_token> & tokens = {})
        : is_input(is_input)
        , consumed(consumed)
        , tokens(tokens)
        {}
} tokens_chunk;

enum seq_state {
    SEQ_GENERATING,
    SEQ_SHARE_PROMPT,
    SEQ_INPUT,
    SEQ_DONE,
};

typedef struct seq_ctx {
    llama_seq_id seq_id;
    int32_t batch_idx;
    enum seq_state state;
    size_t n_remain;
    size_t n_toks; // Note: Does not include initial prompt size.
    llama_sampling_context *ctx_sampling;

    llama_token last_sampled;
    std::vector<tokens_chunk> chunks;
#ifndef LLAMA_NO_SEQREP_SAMPLER
    size_t high_water_mark;
    struct seqrep_rewind_state rewind_state;
    size_t rewind_count;
    size_t rewind_tokens;
#endif
} seq_ctx;


typedef struct gen_ctx {
    llama_context            * ctx          = nullptr;
    llama_model              * model        = nullptr;
    llama_sampling_context   * ctx_sampling = nullptr;

    llama_batch                batch;
    gpt_params                 params;
    llama_sampling_params    & sparams = params.sparams;


    int n_ctx;
    int n_vocab;

    std::vector<llama_token> scratch;
    std::vector<llama_token> prompt_tokens;
    size_t prompt_size = 0;

    llama_seq_id focused_sequence = 0;

    size_t decode_count = 0;
    int64_t decode_time_total = 0, decode_time_last = 0;

    std::vector<seq_ctx> ctxs_seq;

    private:
    bool init_params(const int argc, char ** argv);
    bool init_model();
    bool init_prompt();
    bool init_handlers();
    bool init_sampling();
    bool init_batches();

    public:
    gen_ctx(const int argc, char ** argv);
    ~gen_ctx();
    void dump_batches(const size_t prompt_start = 0);
    void dump_chunks(const std::vector<tokens_chunk> & chunks, const size_t start_offset = 0);
    void handle_seq(seq_ctx & sctx);
#ifndef LLAMA_NO_SEQREP_SAMPLER
    void handle_seq_seqrep(seq_ctx & sctx);
#endif
    bool feed_prompt(
            const std::vector<llama_token> & tokens,
            llama_pos pos = 0,
            llama_seq_id seq = 0);
    bool go();
} gen_ctx;


static void concat_chunks(const std::vector<tokens_chunk> & chunks, std::vector<llama_token> & dst, const size_t start_offset) {
    size_t offset = 0;

    for (const tokens_chunk & chunk : chunks) {
        if (offset + chunk.tokens.size() <= start_offset) {
            offset += chunk.tokens.size();
            continue;
        }

        const size_t chunk_offset = offset < start_offset ? start_offset - offset : 0;
        const size_t chunk_size = chunk.tokens.size() - chunk_offset;
        const llama_token * tp = chunk.tokens.data() + chunk_offset;

        for (size_t i = 0; i < chunk_size; i++, tp++) {
            dst.push_back(*tp);
        }
        offset += chunk.tokens.size();
    }
}


static void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const std::vector<llama_token> input_tokens, const std::string output, const std::vector<llama_token> output_tokens) {

    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = get_sortable_timestamp();

    const bool success = create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: simple-inference\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_string_yaml_multiline(logfile, "output", output.c_str());
    dump_vector_int_yaml(logfile, "output_tokens", output_tokens);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (interrupted) {
            done.store(true);
        } else {
            interrupted.store(true);
        }
    }
}
#endif


static bool check_unsupported(const gpt_params * params) {
    std::string nope;
    const llama_sampling_params & sparams = params->sparams;

    if (params->embedding)
        nope = "embedding";
    else if (!sparams.grammar.empty())
        nope = "grammar"; // Currently broken most likely
    else if (sparams.cfg_scale != 1.0f)
        nope = "cfg_scale";
    else if (!sparams.cfg_negative_prompt.empty())
        nope = "cfg_negative_prompt";
    else if (!params->path_prompt_cache.empty())
        nope = "prompt cache";
    else if (params->escape)
        nope = "prompt escaping";
    else if (params->interactive_first || params->instruct)
        nope = "interactive first or instruct mode";
    else if (!params->input_prefix.empty() || !params->input_suffix.empty() || params->input_prefix_bos)
        nope = "input prefix or suffix";
    else if (params->hellaswag)
        nope = "hellaswag";
    else if (params->n_keep != 0)
        nope = "keep";
    else if (!params->antiprompt.empty())
        nope = "reverse prompt";
    if (!nope.empty()) {
        printf("%s: error: We don't support %s here.\n", __func__, nope.c_str());
        return false;
    }
    return true;
}

bool gen_ctx::init_params(const int argc, char ** argv) {
    if (gpt_params_parse(argc, argv, params) == false) {
        return false;
    }

    if (!check_unsupported(&params)) {
        return false;
    }

    if (params.rope_freq_base != 10000.0) {
        printf("%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        printf("%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx < 8) {
        printf("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    printf("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    return true;
}

bool gen_ctx::init_model() {
    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        printf("%s: error: unable to load model\n", __func__);
        return false;
    }

    // print system information
    {
        printf("\n");
        printf("system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    n_ctx = llama_n_ctx(ctx);
    n_vocab = llama_n_vocab(llama_get_model(ctx));

    return true;
}

bool gen_ctx::init_prompt() {
    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos: %d\n", add_bos);

    if (!params.prompt.empty()) {
        LOG("tokenize the prompt\n");
        prompt_tokens = ::llama_tokenize(ctx, params.prompt, add_bos, true);
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, prompt_tokens).c_str());

    // Should not run without any tokens
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(llama_token_bos(model));
        LOG("input was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, prompt_tokens).c_str());
    }

    LOG("n_ctx: %d\n", n_ctx);

    if ((int) prompt_tokens.size() > n_ctx - 4) {
        printf("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) prompt_tokens.size(), n_ctx - 4);
        return false;
    }
    prompt_size = prompt_tokens.size();

    if (params.verbose_prompt) {
        printf("\n");
        printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        printf("%s: number of tokens in prompt = %zu\n", __func__, prompt_tokens.size());
        for (int i = 0; i < (int) prompt_tokens.size(); i++) {
            printf("%6d -> '%s'\n", prompt_tokens[i], llama_token_to_piece(ctx, prompt_tokens[i]).c_str());
        }

        printf("\n");
    }
    return true;
}

bool gen_ctx::init_handlers() {
    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    return true;
}

bool gen_ctx::init_sampling() {
    printf("sampling: %s\n", llama_sampling_print(sparams).c_str());
#ifndef LLAMA_NO_SEQREP_SAMPLER
    for (auto & sr_params : sparams.seqrep_params) {
        seqrep_sampler_params_dump(&sr_params);
    }
#endif
    ctx_sampling = llama_sampling_init(sparams);
    return true;
}

bool gen_ctx::init_batches() {
    batch = llama_batch_init(std::max(int32_t(prompt_size), params.n_batch), 0, 1);
    int n_remain = params.n_predict;

    if (n_remain < 0 || n_remain + int(prompt_size) > n_ctx) {
        n_remain = n_ctx - prompt_size;
    }

    ctxs_seq.reserve(params.n_parallel);
    for (int32_t i = 0; i < params.n_parallel; i++) {
        seq_ctx && bs = {
            llama_seq_id(i),
            -1,
            i == 0 ? SEQ_INPUT : SEQ_SHARE_PROMPT,
            size_t(n_remain),
            0,
            llama_sampling_init(params.sparams),
            -1,
            {},
#ifndef LLAMA_NO_SEQREP_SAMPLER
            prompt_size + 1,
            seqrep_rewind_state(n_vocab, n_ctx, 2000),
            0,
            0,
#endif
        };
        GGML_ASSERT(prompt_size > 0);
        bs.chunks.emplace_back(true, 0, prompt_tokens);
        if (i > 0) {
            bs.chunks.emplace_back(false, 0, std::vector<llama_token>());
        }
#ifndef LLAMA_NO_SEQREP_SAMPLER
        seqrep_rewind_slot & rw_slot = bs.rewind_state.get_rewind_slot(0);
        rw_slot.ctx_sampling = llama_sampling_init(params.sparams);
#endif
        ctxs_seq.push_back(bs);
    }
    if (!ctxs_seq.empty()) {
        focused_sequence = ctxs_seq.size() - 1;
    }

    return true;
}


gen_ctx::gen_ctx(const int argc, char ** argv) {
    bool result = true;

    result = result && init_params(argc, argv);
    result = result && init_model();
    result = result && init_prompt();
    result = result && init_handlers();
    result = result && init_sampling();
    result = result && init_batches();
    if (!result) {
        throw std::runtime_error("Initialization failed");
    }
}

gen_ctx::~gen_ctx() {
    for (auto & sctx : ctxs_seq) {
        llama_sampling_free(sctx.ctx_sampling);
#ifndef LLAMA_NO_SEQREP_SAMPLER
        for (auto & rs : sctx.rewind_state.rewind_slots) {
            if (rs.ctx_sampling != nullptr) {
                llama_sampling_free(rs.ctx_sampling);
                rs.ctx_sampling = nullptr;
            }
        }
#endif
    }
    llama_sampling_free(ctx_sampling);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();
}


bool gen_ctx::feed_prompt(const std::vector<llama_token> & tokens, llama_pos pos, llama_seq_id seq) {
    int32_t tokens_remain           = tokens.size();
    const llama_token * tokens_curr = tokens.data();

    console::set_display(console::prompt);
    while (tokens_remain > 0 && !interrupted) {
        const int32_t chunk_size = std::min(int32_t(tokens_remain), params.n_batch);
        llama_batch_clear(batch);
        for (int i = 0; i < chunk_size; i++) {
            llama_batch_add(batch, tokens_curr[i], pos + i, {seq}, false);
        }
        pos += batch.n_tokens;
        tokens_remain -= batch.n_tokens;
        batch.logits[batch.n_tokens - 1] = tokens_remain < 1;

        if (llama_decode(ctx, batch) != 0) {
            console::set_display(console::reset);
            printf("%s : failed to eval\n", __func__);
            return false;
        }
        decode_count++;

        // display text
        for (int i = 0; i < batch.n_tokens; i++) {
            const std::string token_str = llama_token_to_piece(ctx, tokens_curr[i]);
            fputs(token_str.c_str(), stdout);
        }
        fflush(stdout);

        tokens_curr += batch.n_tokens;
    }
    console::set_display(console::reset);
    return true;
}

void gen_ctx::dump_chunks(const std::vector<tokens_chunk> & chunks, const size_t start_offset) {
    size_t offset = 0;
    bool prompt_mode = false;
    console::set_display(console::reset);

    for (const tokens_chunk & chunk : chunks) {
        if (offset + chunk.tokens.size() <= start_offset) {
            offset += chunk.tokens.size();
            continue;
        }

        const size_t chunk_offset = offset < start_offset ? start_offset - offset : 0;
        const size_t chunk_size = chunk.tokens.size() - chunk_offset;
        const llama_token * tp = chunk.tokens.data() + chunk_offset;

        if (chunk.is_input != prompt_mode) {
            prompt_mode = chunk.is_input;
            console::set_display(prompt_mode ? console::prompt : console::reset);
        }

        for (size_t i = 0; i < chunk_size; i++, tp++) {
            const std::string token_str = llama_token_to_piece(ctx, *tp);
            fputs(token_str.c_str(), stdout);
        }
        offset += chunk.tokens.size();
    }
    if (prompt_mode) {
        console::set_display(console::reset);
    }
    fflush(stdout);
}

void gen_ctx::dump_batches(const size_t prompt_start) {

    bool first = true;

    for (seq_ctx & sctx : ctxs_seq) {
        if (sctx.seq_id == focused_sequence) continue;
        printf("\n\n%s Result #%d (size: %zu",
            !first ? "====================" : "####################",
            sctx.seq_id + 1, prompt_size + sctx.n_toks);
#ifndef LLAMA_NO_SEQREP_SAMPLER
        printf(", rewind cnt/toks: %zu/%zu", sctx.rewind_count, sctx.rewind_tokens);
#endif
        printf("%s):", sctx.state == SEQ_DONE ? ", DONE" : "");
        dump_chunks(sctx.chunks, prompt_start);
        first = false;
    }
    seq_ctx & sctx = ctxs_seq[focused_sequence];
    printf("\n\n%s Result #%d (size: %zu",
            !first ? "====================" : "####################",
            sctx.seq_id + 1, prompt_size + sctx.n_toks);
#ifndef LLAMA_NO_SEQREP_SAMPLER
    printf(", rewind cnt/toks: %zu/%zu", sctx.rewind_count, sctx.rewind_tokens);
#endif
    puts("):");
    dump_chunks(sctx.chunks, prompt_start);
}

void gen_ctx::handle_seq(seq_ctx & sctx) {
    switch (sctx.state) {
        case SEQ_DONE:
        case SEQ_SHARE_PROMPT: break;

        case SEQ_GENERATING: {
            GGML_ASSERT(sctx.batch_idx >= 0);
            scratch.resize(prompt_size);
            concat_chunks(sctx.chunks, scratch, prompt_size);
#ifndef LLAMA_NO_SEQREP_SAMPLER
            handle_seq_seqrep(sctx);
#endif
            sctx.last_sampled = llama_sampling_sample(ctx_sampling, ctx, NULL, sctx.batch_idx, scratch);
            llama_sampling_accept(sctx.ctx_sampling, ctx, sctx.last_sampled, true);
            if (sctx.seq_id == focused_sequence) {
                const std::string token_str = llama_token_to_piece(ctx, sctx.last_sampled);
                fputs(token_str.c_str(), stdout);
                fflush(stdout);
            }
            sctx.n_toks++;
            sctx.n_remain--;
            if (sctx.chunks.empty() || sctx.chunks.back().is_input) {
                sctx.chunks.emplace_back(0, false, std::vector<llama_token>());
            }
            sctx.chunks.back().tokens.push_back(sctx.last_sampled);
            if (sctx.last_sampled == llama_token_eos(model) || sctx.n_remain == 0) {
                sctx.state = SEQ_DONE;
                llama_kv_cache_seq_rm(ctx, sctx.seq_id, -1, -1);
                sctx.batch_idx = -1;
                // printf(" [end of text]\n");
                // break;
            } else {
                sctx.batch_idx = batch.n_tokens;
                llama_batch_add(batch, sctx.last_sampled, prompt_size + sctx.n_toks, {sctx.seq_id}, true);
            }
        } break;

        case SEQ_INPUT: {
            sctx.last_sampled = -1;
            GGML_ASSERT(!sctx.chunks.empty());
            tokens_chunk & chunk = sctx.chunks.back();
            GGML_ASSERT(chunk.is_input);
            GGML_ASSERT(chunk.consumed < chunk.tokens.size());
            GGML_ASSERT(!chunk.tokens.empty());

            const size_t remain = chunk.tokens.size() - chunk.consumed;
            const size_t to_consume = std::min(size_t(params.n_batch), remain);
            for (size_t i = chunk.consumed; i < chunk.consumed + to_consume; ++i) {
                llama_batch_add(batch, chunk.tokens[i], llama_pos(prompt_size + sctx.n_toks + i), {sctx.seq_id}, false);
            }
            chunk.consumed += to_consume;
            sctx.n_remain -= to_consume;
            sctx.n_toks += to_consume;
            if (chunk.consumed == chunk.tokens.size()) {
#ifndef LLAMA_NO_SEQREP_SAMPLER
                // FIXME: Move this logic to a more appropriate place.
                for (size_t i = 0; i < chunk.consumed; i++) {
                    sctx.rewind_state.logit_slots.emplace_back(n_vocab);
                }
                sctx.high_water_mark = sctx.n_toks + 1;
#endif
                sctx.batch_idx = batch.n_tokens - 1;
                batch.logits[sctx.batch_idx] = true;
                sctx.chunks.emplace_back(false, 0, std::vector<llama_token>());
                sctx.chunks.back().tokens.reserve(sctx.n_remain);
                sctx.state = SEQ_GENERATING;
            } else {
                sctx.batch_idx = -1;
            }
        } break;

        default:
        throw std::runtime_error("Unexpected state in handle_seq");
    }
}

#ifndef LLAMA_NO_SEQREP_SAMPLER
    void gen_ctx::handle_seq_seqrep(seq_ctx & sctx) {
        if (sctx.n_toks > 0) {
            seqrep_rewind_slot & rw_slot = sctx.rewind_state.get_rewind_slot(sctx.n_toks);
            if (rw_slot.ctx_sampling == nullptr) {
                rw_slot.ctx_sampling = llama_sampling_init(params.sparams);
            }
            llama_sampling_cp(sctx.ctx_sampling, rw_slot.ctx_sampling);
            sctx.rewind_state.set_logits_slot(ctx, sctx.n_toks, sctx.batch_idx);
        } else {
            return;
        }
        std::vector<llama_token> seq_last_tokens;
        seq_last_tokens.reserve(sctx.n_toks);
        concat_chunks(sctx.chunks, seq_last_tokens, prompt_size);

        size_t rewind_distance =
            llama_seqrep_handle_rewind(
                ctx, sctx.rewind_state, seq_last_tokens, sctx.n_toks, prompt_tokens,
                sparams.seqrep_params, &sctx.high_water_mark, sctx.batch_idx);
        if (rewind_distance < 1) {
            return;
        }
        GGML_ASSERT(rewind_distance <= sctx.n_toks && "Rewind index out of bounds somehow?");
        const size_t slot_idx = sctx.n_toks - rewind_distance;
        const llama_token nl_id = llama_token_nl(model);

        seqrep_rewind_slot & rw_slot = sctx.rewind_state.get_rewind_slot(slot_idx);
        llama_sampling_cp(rw_slot.ctx_sampling, sctx.ctx_sampling);

        if (sctx.seq_id == focused_sequence) {
            console::set_display(console::error);
            fputs("\u3010", stdout);
            for (size_t i = seq_last_tokens.size() - rewind_distance; i < seq_last_tokens.size(); i++) {
                if (seq_last_tokens[i] == nl_id) {
                    fputs("\\n", stdout);
                    continue;
                }
                const std::string token_str = llama_token_to_piece(ctx, seq_last_tokens[i]);
                fputs(token_str.c_str(), stdout);
            }
            fputs("\u3011", stdout);
            console::set_display(console::reset);
            fflush(stdout);
        }

        sctx.n_remain += rewind_distance;
        sctx.n_toks -= rewind_distance;
        sctx.rewind_count++;
        sctx.rewind_tokens += rewind_distance;
        llama_kv_cache_seq_rm(ctx, sctx.seq_id, prompt_size + sctx.n_toks + 1, -1);
        while (!sctx.chunks.empty() && rewind_distance > 0) {
            tokens_chunk & last_chunk = sctx.chunks.back();
            GGML_ASSERT(!last_chunk.is_input);

            if (last_chunk.tokens.size() >= rewind_distance) {
                last_chunk.tokens.resize(last_chunk.tokens.size() - rewind_distance);
                rewind_distance = 0;
                break;
            }
            rewind_distance -= last_chunk.tokens.size();
            sctx.chunks.pop_back();
        }
    }
#endif

bool gen_ctx::go() {
    if (ctxs_seq.empty()) {
        return false;
    }

    if (decode_count == 0) {
        scratch.reserve(n_ctx);
        scratch.resize(prompt_size);
        std::copy(prompt_tokens.begin(), prompt_tokens.end(), scratch.begin());
        // FIXME: Hacky.
        if (!feed_prompt(prompt_tokens)) {
            throw std::runtime_error("Prompt processing failed");
        }
        for (auto & sctx : ctxs_seq) {
            sctx.batch_idx = batch.n_tokens - 1;
            sctx.state = SEQ_GENERATING;
            if (sctx.seq_id == 0) {
                sctx.chunks.back().consumed = prompt_size;
                sctx.chunks.emplace_back(false, 0, std::vector<llama_token>());
            } else {
                sctx.chunks.front().consumed = prompt_size;
                llama_kv_cache_seq_cp(ctx, 0, sctx.seq_id, 0, prompt_size);
            }
#ifndef LLAMA_NO_SEQREP_SAMPLER
            seqrep_rewind_slot & rw_slot = sctx.rewind_state.get_rewind_slot(0);
            rw_slot.ctx_sampling = llama_sampling_init(params.sparams);
            llama_sampling_cp(sctx.ctx_sampling, rw_slot.ctx_sampling);
            sctx.rewind_state.set_logits_slot(ctx, 0, sctx.batch_idx);
#endif
        }
    }

    llama_batch_clear(batch);
    for (auto & sctx : ctxs_seq) {
        handle_seq(sctx);
    }
    if (batch.n_tokens == 0) return false;

    decode_time_last = ggml_time_us();
    const int decode_result = llama_decode(ctx, batch);
    decode_time_last = std::max(int64_t(0), ggml_time_us() - decode_time_last);
    decode_time_total += decode_time_last;

    // FIXME: Handle KV cache pressure better.
    if (decode_result != 0) {
        fprintf(stderr, "%s : failed to eval batch of size %d: %s\n", __func__, batch.n_tokens,
            decode_result == 1 ? "couldn't find slot" : "unknown error");
        return false;
    }
    decode_count++;
    return true;
}

static bool handle_commands(gen_ctx & gctx) {
    std::string line;
    line.reserve(1024);


    printf("\n- Entering command mode. Use /help for help, blank line to exit. Focused sequence: %d\n", gctx.focused_sequence + 1);
    fflush(stdout);
    while (1) {
        printf("> ");
        fflush(stdout);
        console::readline(line, false);
        console::set_display(console::reset);
        while (!line.empty() && std::isspace(line.back())) {
            line.pop_back();
        }
        if (line.empty()) break;
        if (line.size() < 2 || line.front() != '/') {
            printf("\n- Bad command\n");
            continue;
        }
        size_t sep_idx = line.find(' ');
        std::string command, rest;
        if (sep_idx != std::string::npos) {
            command = line.substr(1, sep_idx - 1);
            rest = line.substr(sep_idx + 1);
        } else {
            command = line.substr(1);
        }
        for (char & c : command) c = std::tolower(c);

        if (command == "h" || command == "help") {
            printf("- Help: For commands with [SEQ], optionally specify a sequence number here to set the target.\n");
            printf("        If sequence isn't specified, then the current focus is used if possible.\n");
            printf("        One of any punctuation character is allowed after the number.\n");
            printf("        For example, '/1add hello' and '/1,add hello' both add 'hello' to sequence 1.\n");
            printf("- Available commands:\n");
            printf("  /[SEQ]add TEXT     : Adds the specified text to the focused sequence. Alias: /a\n");
            printf("  /[SEQ]addesc TEXT  : Same as /add but handles escapes (\\n, \\x20, etc) and tokenizes without a leading space. Alias: /ae\n");
            printf("  /[SEQ]addline TEXT : Same as /add but appends a newline. Alias: /al\n");
            printf("  /help              : Show this help. Alias: /h\n");
            printf("  /[SEQ]dump [N]     : Dump the last N tokens of SEQ showing offsets from the end. N defaults to 200 if not specified. Alias: /d\n");
            printf("  /[SEQ]dumptokens N : Same as /dump but displays token IDs as well. Alias: /dt\n");
            printf("  /[SEQ]kill         : Stop sequence SEQ. Alias: /k\n");
            printf("  /list              : List sequences and their state. Alias: /l\n");
            printf("  /[SEQ]focus        : Focus sequence SEQ. Alias: Just use /1, /2, etc\n");
            printf("  /[SEQ]print        : Display the content of SEQ. Alias: /p\n");
            printf("  /quit              : Exit the program. Alias: /q\n");
            printf("- End listing\n");
            continue;
        }

        if (command == "q" || command == "quit") return false;

        llama_seq_id target = gctx.focused_sequence;

        // Focus
        if (isdigit(command[0])) {
            char * parse_end = nullptr;
            target = std::strtol(command.c_str(), &parse_end, 10);
            if (target < 1 || size_t(target) > gctx.ctxs_seq.size()) {
               printf("! Bad seq id\n");
               continue;
            }
            target--;
            if (std::ispunct(*parse_end)) parse_end++;
            command = std::string(parse_end);
        }

        if (command.empty() || command == "focus") {
            printf("- Focus changed from %d to %d\n", gctx.focused_sequence + 1, target + 1);
            gctx.focused_sequence = llama_seq_id(target);
            continue;
        }

        if (command == "k" || command == "kill") {
            if (target == gctx.focused_sequence) {
                printf("! Kill: Can't kill focus\n");
            } else {
                printf("- Killed sequence %d\n", target + 1);
                gctx.ctxs_seq[target].state = SEQ_DONE;
                llama_kv_cache_seq_rm(gctx.ctx, target, -1, -1);
            }
            continue;
        }

        if (command == "l" || command == "list") {
            printf("- Listing %zu sequence%s:\n",
                gctx.ctxs_seq.size(),
                gctx.ctxs_seq.size() != 1 ? "s" : "");
            for (const seq_ctx & sctx : gctx.ctxs_seq) {
                std::string label;
                switch (sctx.state) {
                    case SEQ_DONE:         label = "DONE"; break;
                    case SEQ_GENERATING:   label = "LIVE"; break;
                    case SEQ_INPUT:        label = "FEED"; break;
                    case SEQ_SHARE_PROMPT: label = "WAIT"; break;
                    default: GGML_ASSERT(false);
                }
                printf("  %s%3d (%s): generated %5zu, remain %5zu. chunks: ",
                    sctx.seq_id == gctx.focused_sequence ? "*" : " ",
                    sctx.seq_id + 1, label.c_str(),
                    sctx.n_toks, sctx.n_remain);
                for (const tokens_chunk & chunk : sctx.chunks) {
                    if (chunk.is_input) {
                        printf("INP(%5zu,%5zu), ", chunk.tokens.size(), chunk.consumed);

                    } else {
                        printf("GEN(%5zu), ", chunk.tokens.size());
                    }
                }
                printf("\n");
            }
            continue;
        }

        if (   command == "al" || command == "a"  || command == "ae"
            || command == "add" || command == "addline" || command == "addesc") {
            bool is_special = false;
            seq_ctx & sctx = gctx.ctxs_seq[target < 0 ? gctx.focused_sequence : target];

            if (command == "al" || command == "addline") {
                rest.push_back('\n');
            } else if (command == "ae" || command == "addesc") {
                process_escapes(rest);
                is_special = true;
            }
            std::vector<llama_token> input_tokens = ::llama_tokenize(gctx.model, rest, false, is_special);
            if (input_tokens.size() > sctx.n_remain) {
                printf("! Input is %zu token(s) but sequence %d only has space for %zu\n",
                    input_tokens.size(), gctx.focused_sequence + 1, sctx.n_remain);
                continue;
            }
            if (!sctx.chunks.back().is_input) {
                sctx.chunks.emplace_back(true, 0, input_tokens);
            } else {
                tokens_chunk & chunk = sctx.chunks.back();
                const size_t old_size = chunk.tokens.size();

                chunk.tokens.resize(old_size + input_tokens.size());
                std::copy(input_tokens.begin(), input_tokens.end(), chunk.tokens.begin() + old_size);
            }
            sctx.state = SEQ_INPUT;
            continue;
        }

        if (command == "p" || command == "print") {
            seq_ctx & sctx = gctx.ctxs_seq[target < 0 ? gctx.focused_sequence : target];
            std::string label;
            switch (sctx.state) {
                case SEQ_DONE:         label = "DONE"; break;
                case SEQ_GENERATING:   label = "LIVE"; break;
                case SEQ_INPUT:        label = "FEED"; break;
                case SEQ_SHARE_PROMPT: label = "WAIT"; break;
                default: GGML_ASSERT(false);
            }

            printf("- Showing sequence %3d%s: state %s, generated %5zu, remain %5zu. chunks: ",
                    sctx.seq_id + 1,
                    sctx.seq_id == gctx.focused_sequence ? "(focus)" : " ",
                    label.c_str(), sctx.n_toks, sctx.n_remain);
            for (const tokens_chunk & chunk : sctx.chunks) {
                if (chunk.is_input) {
                    printf("INP(%5zu,%5zu), ", chunk.tokens.size(), chunk.consumed);

                } else {
                    printf("GEN(%5zu), ", chunk.tokens.size());
                }
            }
            printf("\n");
            gctx.dump_chunks(sctx.chunks);
            printf("\n- Done\n");
            continue;
        }

        if (command == "d" || command == "dt" || command == "dump" || command == "dumptokens") {
            seq_ctx & sctx = gctx.ctxs_seq[target < 0 ? gctx.focused_sequence : target];
            const bool with_id = command == "dt" || command == "dumptokens";
            const size_t max_n = sctx.n_toks + gctx.prompt_size;
            size_t dump_n = size_t(std::max(0, atoi(rest.c_str())));
            if (dump_n == 0) dump_n = 200;
            dump_n = std::min(dump_n, max_n);

            printf("- Dumping last %zu token%s from sequence %d\n",
                dump_n, dump_n != 1 ? "s" : "", target + 1);

            std::vector<llama_token> result;
            result.reserve(dump_n);
            concat_chunks(sctx.chunks, result, max_n - dump_n);
            GGML_ASSERT(result.size() == dump_n);
            for (size_t i = 0; i < dump_n; i++) {
                const llama_token tid = result[i];
                console::set_display(console::user_input);
                printf("[%zu", dump_n - i);
                if (with_id) {
                    printf(",%d", tid);
                }
                fputs("]", stdout);
                console::set_display(console::reset);
                fputs(llama_token_to_piece(gctx.ctx, tid).c_str(), stdout);

            }
            console::set_display(console::reset);
            printf("\n\n- Dump complete.\n");
            continue;
        }

        printf("! Bad command\n");
    }
    return true;
}

int main(int argc, char ** argv) {
    gen_ctx gctx(argc, argv);

    // This might look weird but done can get set while go() is running.
    while (!done && gctx.go() && !done) {
        bool need_dump = gctx.params.n_parallel > 1 && gctx.decode_count % SI_DUMP_SEQUENCES_INTERVAL == 0;
        if (interrupted) {
            if (!gctx.params.interactive || !handle_commands(gctx)) break;
            // Double check that ^C wasn't hit again.
            if (done) break;
            interrupted = false;
            need_dump = true;
        }
        if (need_dump) {
            printf("\n-- Last decode[%zu]: %.3f, avg: %.3f",
                gctx.decode_count, double(gctx.decode_time_last) / 1000000,
                (double(gctx.decode_time_total) / 1000000) / double(gctx.decode_count));
            gctx.dump_batches((gctx.prompt_size > 20) ? (gctx.prompt_size - 10) : 0);
        }
    }
    gctx.focused_sequence = gctx.ctxs_seq.size() - 1;
    gctx.dump_batches();
    puts("");
    console::cleanup();

    llama_print_timings(gctx.ctx);
}
