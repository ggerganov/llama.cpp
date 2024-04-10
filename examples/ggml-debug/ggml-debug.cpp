#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <mutex>
#include <vector>

struct callback_data {
    std::mutex m_mutex;
    std::vector<float> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(const float * data, const int64_t * ne) {
    int i, j, k;
    printf("                                     [\n");
    for (i = 0; i < ne[2] && i < 3; i++) {
        printf("                                      [\n");
        for (j = 0; j < ne[1] && j < 3; j++) {
            printf("                                       [");
            for (k = 0; k < ne[0] && k < 3; k++) {
                printf("%8.4f", data[k * ne[1] * ne[2] + j * ne[2] + i]);
                if (k < ne[0] - 1 && k < 2) printf(", ");
            }
            if (ne[0] > 3) printf(", ...");
            printf("],\n");
        }
        if (ne[1] > 3) printf("                                       ...\n");
        printf("                                      ],\n");
    }
    if (ne[2] > 3) printf("                                     ...\n");
    printf("                                     ]\n");
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    std::lock_guard<std::mutex> lock(cb_data->m_mutex);

    char src1_str[128] = {0};
    if (src1) {
        sprintf(src1_str, "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = %10s(%s{%s}, %s}) = {%s} \n", __func__,
           t->name, ggml_op_name(t->op),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes / sizeof(float));
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    const float * data = is_host ? (const float *) t->data : cb_data->data.data();
    ggml_print_tensor(data, t->ne);

    return true;
}

static bool run(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {

    callback_data cb_data;

    gpt_params params;
    params.n_batch = 512;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    print_build_info();

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = llama_model_params_from_gpt_params(params);

    llama_model * model = nullptr;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(), mparams);
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(), mparams);
    } else {
        model = llama_load_model_from_file(params.model.c_str(), mparams);
    }

    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    auto cparams = llama_context_params_from_gpt_params(params);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    cparams.cb_eval = ggml_debug;
    cparams.cb_eval_user_data = &cb_data;

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", get_system_info(params).c_str());
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
