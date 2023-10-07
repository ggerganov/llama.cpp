#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "clip.h"
#include "common.h"
#include "llama.h"


static bool eval_image_embd(llama_context * ctx_llama, float * embd, int N, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));
    
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = N - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, (embd+i*n_embd), nullptr, nullptr, nullptr, *n_past, 1, 0, };
        if (llama_decode(ctx_llama, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int N, int * n_past) {
    int n_batch = N;
    for (int i = 0; i < (int) tokens.size(); i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int N, int * n_past){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, true);
    eval_tokens(ctx_llama, embd_inp, N, n_past);
    return true;
}

static llama_token sample_id(llama_context * ctx_llama, gpt_params & params) {
      // out of user input, sample next token
    const float   temp      = params.temp;
    const int32_t top_k     = params.top_k <= 0 ? llama_n_vocab(llama_get_model(ctx_llama)) : params.top_k;
    const float   top_p     = params.top_p;
    const float   tfs_z     = params.tfs_z;
    const float   typical_p = params.typical_p;
      // const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
      // const float   repeat_penalty  = params.repeat_penalty;
      // const float   alpha_presence  = params.presence_penalty;
      // const float   alpha_frequency = params.frequency_penalty;
    const int     mirostat     = params.mirostat;
    const float   mirostat_tau = params.mirostat_tau;
    const float   mirostat_eta = params.mirostat_eta;
      // const bool    penalize_nl     = params.penalize_nl;

    llama_token id = 0;
    {
        auto logits  = llama_get_logits(ctx_llama);
        auto n_vocab = llama_n_vocab(llama_get_model(ctx_llama));

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
          // float nl_logit = logits[llama_token_nl(ctx)];
          // auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
          // llama_sample_repetition_penalty(ctx, &candidates_p,
          //      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
          //      last_n_repeat, repeat_penalty);
          // llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
          // last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
          // last_n_repeat, alpha_frequency, alpha_presence);
          // if (!penalize_nl) {
          //     logits[llama_token_nl(ctx)] = nl_logit;
          // }

        if (temp <= 0) {
              // Greedy sampling
            id = llama_sample_token_greedy(ctx_llama, &candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const  int mirostat_m    = 100;
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token_mirostat(ctx_llama, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token_mirostat_v2(ctx_llama, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                  // Temperature sampling
                llama_sample_top_k(ctx_llama, &candidates_p, top_k, 1);
                llama_sample_tail_free(ctx_llama, &candidates_p, tfs_z, 1);
                llama_sample_typical(ctx_llama, &candidates_p, typical_p, 1);
                llama_sample_top_p(ctx_llama, &candidates_p, top_p, 1);
                llama_sample_temp(ctx_llama, &candidates_p, temp);
                id = llama_sample_token(ctx_llama, &candidates_p);
            }
        }
    }

    return id;
}

const char * sample(struct llama_context * ctx_llama, gpt_params & params, int * n_past) {
    int id = sample_id(ctx_llama, params);
    static std::string ret;
    if (id == llama_token_eos(ctx_llama)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc < 3) {
        printf("usage: %s <path/to/llava-rlhf-qe_k.gguf> <path/to/llava-encoder-f16.gguf> [path/to/an/image.jpg] [a text prompt]\n", argv[0]);
    }

          params.model     = argv[1];
    const char * clip_path = argv[2];
    const char * img_path;
    if (argc >= 4) {
        img_path = argv[3];
    }

    if (argc >= 5) {
        params.prompt = argv[4];
    }

    if (params.prompt.empty()) {
        params.prompt = "describe the image in detail.";
    }
    
    
    auto ctx_clip = clip_model_load(clip_path, 3);
    clip_image_u8 img;
    clip_image_f32 img_res;
    clip_image_load_from_file(img_path, &img);
    clip_image_preprocess(ctx_clip, &img, &img_res);
    float * vec = (float *)malloc(4096 * 576 * sizeof(float));
    clip_image_encode(ctx_clip, params.n_threads, &img_res, vec, false);
    clip_free(ctx_clip);
    
    llama_backend_init(params.numa);

    llama_model_params model_params = llama_model_default_params();
      // model_params.n_gpu_layers = 99; // offload all layers to the GPU
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }
    
    llama_context_params ctx_params                 = llama_context_default_params();
                         ctx_params.seed            = 1234;
                         ctx_params.n_ctx           = 2048;
                         ctx_params.n_threads       = params.n_threads;
                         ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    llama_context        * ctx_llama                = llama_new_context_with_model(model, ctx_params);
    
    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }
    
    int n_past      = 0;
    int max_tgt_len = 256;
    eval_string(ctx_llama, "user: ", params.n_batch, &n_past);
    eval_image_embd(ctx_llama, vec, 576, params.n_batch, &n_past);
    eval_string(ctx_llama, params.prompt.c_str(), params.n_batch, &n_past);
eval_string(ctx_llama, "\nassistant:", params.n_batch, &n_past);
printf("n_past = %d\n", n_past);
    
    const char* tmp;
    for (int i=0; i<max_tgt_len; i++) {
        tmp = sample(ctx_llama, params, &n_past);
        if (strcmp(tmp, "</s>")==0) break;
        printf("%s", tmp);
        fflush(stdout);
    }
    printf("\n");

    llama_print_timings(ctx_llama);

    llama_free(ctx_llama);
    llama_free_model(model);
    llama_backend_free();
    free(vec);

    return 0;
}
