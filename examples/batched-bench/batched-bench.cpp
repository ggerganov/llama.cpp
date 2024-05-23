#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// mutates the input string
static std::vector<int> parse_list(char * p) {
    std::vector<int> ret;

    char * q = p;

    while (*p) {
        if (*p == ',') {
            *p = '\0';
            ret.push_back(std::atoi(q));
            q = p + 1;
        }

        ++p;
    }

    ret.push_back(std::atoi(q));

    return ret;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [N_KV_MAX] [N_BATCH] [N_UBATCH] [FATTN] [IS_PP_SHARED] [NGL] <PP> <TG> <PL>\n" , argv[0]);
        printf("  <PP>, <TG> and PL are comma-separated lists of numbers without spaces\n\n");
        printf("  example: %s ggml-model-f16.gguf 2048 2048 512 0 999 128,256,512 128,256 1,2,4,8,16,32\n\n", argv[0]);
        return 1 ;
    }

    int n_kv_max     = 2048;
    int n_batch      = 2048;
    int n_ubatch     = 512;
    bool flash_attn  = false;
    int is_pp_shared = 0;
    int n_gpu_layers = 0;

    std::vector<int> n_pp = { 128, 256, 512, 1024, 2048, 3584, 7680, };
    std::vector<int> n_tg = { 128, 256, };
    std::vector<int> n_pl = { 1, 2, 4, 8, 16, 32, };
    //std::vector<int> n_pl = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, };

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        n_kv_max = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        n_batch = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        n_ubatch = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        flash_attn = std::atoi(argv[5]);
    }

    if (argc >= 7) {
        is_pp_shared = std::atoi(argv[6]);
    }

    if (argc >= 8) {
        n_gpu_layers = std::atoi(argv[7]);
    }

    if (argc >= 9) {
        n_pp = parse_list(argv[8]);
    }

    if (argc >= 10) {
        n_tg = parse_list(argv[9]);
    }

    if (argc >= 11) {
        n_pl = parse_list(argv[10]);
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    const std::vector<float> t_split(llama_max_devices(), 0.0f);

    model_params.n_gpu_layers = n_gpu_layers;
    model_params.tensor_split = t_split.data();

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed       = 1234;
    ctx_params.n_ctx      = n_kv_max;
    ctx_params.n_batch    = n_batch;
    ctx_params.n_ubatch   = n_ubatch;
    ctx_params.flash_attn = flash_attn;

    ctx_params.n_threads       = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    // ensure enough sequences are available
    ctx_params.n_seq_max = *std::max_element(n_pl.begin(), n_pl.end());

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch) {
        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            llama_synchronize(ctx);
        }

        return true;
    };

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            llama_batch_add(batch, 0, i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    LOG_TEE("\n");
    LOG_TEE("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, n_batch, n_ubatch, flash_attn, is_pp_shared, n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
    LOG_TEE("\n");

    LOG_TEE("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n", "PP",     "TG",     "B",    "N_KV",     "T_PP s",   "S_PP t/s", "T_TG s",   "S_TG t/s", "T s",      "S t/s");
    LOG_TEE("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                const int pp = n_pp[i_pp];
                const int tg = n_tg[i_tg];
                const int pl = n_pl[i_pl];

                const int n_ctx_req = is_pp_shared ? pp + pl*tg : pl*(pp + tg);

                if (n_ctx_req > n_kv_max) {
                    continue;
                }

                llama_batch_clear(batch);

                for (int i = 0; i < pp; ++i) {
                    for (int j = 0; j < (is_pp_shared ? 1 : pl); ++j) {
                        llama_batch_add(batch, 0, i, { j }, false);
                    }
                }
                batch.logits[batch.n_tokens - 1] = true;

                const auto t_pp_start = ggml_time_us();

                llama_kv_cache_clear(ctx);

                if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                    LOG_TEE("%s: llama_decode() failed\n", __func__);
                    return 1;
                }

                if (is_pp_shared) {
                    for (int32_t i = 1; i < pl; ++i) {
                        llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
                    }
                }

                const auto t_pp_end = ggml_time_us();

                const auto t_tg_start = ggml_time_us();

                for (int i = 0; i < tg; ++i) {
                    llama_batch_clear(batch);

                    for (int j = 0; j < pl; ++j) {
                        llama_batch_add(batch, 0, pp + i, { j }, true);
                    }

                    if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                        LOG_TEE("%s: llama_decode() failed\n", __func__);
                        return 1;
                    }
                }

                const auto t_tg_end = ggml_time_us();

                const int32_t n_kv = n_ctx_req;

                const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                const float t    = t_pp + t_tg;

                const float speed_pp = is_pp_shared ? pp / t_pp : pl*pp / t_pp;
                const float speed_tg = pl*tg / t_tg;
                const float speed    = n_kv / t;

                LOG_TEE("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, pl, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
            }
        }
    }

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
