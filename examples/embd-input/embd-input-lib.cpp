// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "embd-input.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static llama_context ** g_ctx;

extern "C" {

struct MyModel* create_mymodel(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return nullptr;
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed  = %d\n", __func__, params.seed);

    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return nullptr;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }
    struct MyModel * ret = new MyModel();
    ret->ctx = ctx;
    ret->params = params;
    ret->n_past = 0;
    // printf("ctx: %d\n", ret->ctx);
    return ret;
}

void free_mymodel(struct MyModel * mymodel) {
    llama_context * ctx = mymodel->ctx;
    llama_print_timings(ctx);
    llama_free(ctx);
    delete mymodel;
}


bool eval_float(void * model, float * input, int N){
    MyModel * mymodel = (MyModel*)model;
    llama_context * ctx = mymodel->ctx;
    gpt_params params = mymodel->params;
    int n_emb = llama_n_embd(ctx);
    int n_past = mymodel->n_past;
    int n_batch = N; // params.n_batch;

    for (int i = 0; i < (int) N; i += n_batch) {
        int n_eval = (int) N - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_eval_embd(ctx, (input+i*n_emb), n_eval, n_past, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        n_past += n_eval;
    }
    mymodel->n_past = n_past;
    return true;
}

bool eval_tokens(void * model, std::vector<llama_token> tokens) {
    MyModel * mymodel = (MyModel* )model;
    llama_context * ctx;
    ctx = mymodel->ctx;
    gpt_params params = mymodel->params;
    int n_past = mymodel->n_past;
    for (int i = 0; i < (int) tokens.size(); i += params.n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }
        if (llama_eval(ctx, &tokens[i], n_eval, n_past, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        n_past += n_eval;
    }
    mymodel->n_past = n_past;
    return true;
}

bool eval_id(struct MyModel* mymodel, int id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(mymodel, tokens);
}

bool eval_string(struct MyModel * mymodel,const char* str){
    llama_context * ctx = mymodel->ctx;
    std::string str2 = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx, str2, true);
    eval_tokens(mymodel, embd_inp);
    return true;
}

llama_token sampling_id(struct MyModel* mymodel) {
    llama_context* ctx = mymodel->ctx;
    gpt_params params = mymodel->params;
    // int n_ctx = llama_n_ctx(ctx);

    // out of user input, sample next token
    const float   temp            = params.temp;
    const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
    const float   top_p           = params.top_p;
    const float   tfs_z           = params.tfs_z;
    const float   typical_p       = params.typical_p;
    // const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
    // const float   repeat_penalty  = params.repeat_penalty;
    // const float   alpha_presence  = params.presence_penalty;
    // const float   alpha_frequency = params.frequency_penalty;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    // const bool    penalize_nl     = params.penalize_nl;

    llama_token id = 0;
    {
        auto logits  = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        // Apply params.logit_bias map
        for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
            logits[it->first] += it->second;
        }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // TODO: Apply penalties
        // float nl_logit = logits[llama_token_nl()];
        // auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        // llama_sample_repetition_penalty(ctx, &candidates_p,
        //      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
        //      last_n_repeat, repeat_penalty);
        // llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
        // last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
        // last_n_repeat, alpha_frequency, alpha_presence);
        // if (!penalize_nl) {
        //     logits[llama_token_nl()] = nl_logit;
        // }

        if (temp <= 0) {
            // Greedy sampling
            id = llama_sample_token_greedy(ctx, &candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token(ctx, &candidates_p);
            }
        }
    }

    return id;
}

const char * sampling(struct MyModel * mymodel) {
    llama_context * ctx = mymodel->ctx;
    int id = sampling_id(mymodel);
    static std::string ret;
    if (id == llama_token_eos()) {
        ret = "</s>";
    } else {
        ret = llama_token_to_str(ctx, id);
    }
    eval_id(mymodel, id);
    return ret.c_str();
}

}
