#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>

struct callback_data {
    std::vector<uint8_t> data;
    int n_tokens = 0;
    int n_embd = 0;
    bool is_eval_pos = true;
    std::vector<float *> v_pos;
    std::vector<float *> v_neg;
    std::vector<float *> v_diff;
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

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    static const char * l_out_name = "l_out";
    const bool is_l_out = strncmp(t->name, l_out_name, strlen(l_out_name)) == 0;
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return is_l_out;
    }

    if (!is_l_out || t->ne[1] != cb_data->n_tokens) {
        return true;
    }

    char src1_str[128] = {0};
    if (src1) {
        sprintf(src1_str, "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    printf("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
           t->name, ggml_type_name(t->type), ggml_op_desc(t),
           src0->name, ggml_ne_string(src0).c_str(),
           src1 ? src1_str : "",
           ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (t->type == GGML_TYPE_F32) {
        float * data = (float *) (is_host ? t->data : cb_data->data.data());
        float * dest = (float *) malloc(ggml_nbytes(t));
        memcpy(dest, data, ggml_nbytes(t));
        if (cb_data->is_eval_pos) {
            cb_data->v_pos.push_back(dest);
        } else {
            cb_data->v_neg.push_back(dest);
        }
    }

    return true;
}

static bool get_hidden_layers(llama_context * ctx, std::vector<llama_token> & tokens) {
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }
    return true;
}

static void padding_seq(llama_context * ctx, std::vector<llama_token> & tokens, size_t len) {
    // TODO: customize padding token
    std::vector<llama_token> pad_tokens = ::llama_tokenize(ctx, " ", false);
    llama_token pad_tok = pad_tokens.back();
    while (tokens.size() < len) {
        tokens.push_back(pad_tok);
    }
}

static void calc_diff(callback_data & cb_data) {
    // TODO: assert cb_data.v_pos.size() == cb_data.v_neg.size()
    const size_t n_elems = cb_data.n_embd * cb_data.n_tokens;
    for (size_t il = 0; il < cb_data.v_pos.size(); il++) {
        auto & inp_pos = cb_data.v_pos[il];
        auto & inp_neg = cb_data.v_neg[il];
        float * dest = (float *) malloc(n_elems * sizeof(float *));
        for (size_t i = 0; i < n_elems; i++) {
            dest[i] = inp_pos[i] - inp_neg[i];
        }
        cb_data.v_diff.push_back(dest);
    }
}

int main(int argc, char ** argv) {
    callback_data cb_data;
    std::string prompt_pos = "happy";
    std::string prompt_neg = "sad";

    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    print_build_info();
    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    llama_model * model;
    llama_context * ctx;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    std::vector<llama_token> tokens_pos = ::llama_tokenize(ctx, prompt_pos, add_bos);
    std::vector<llama_token> tokens_neg = ::llama_tokenize(ctx, prompt_neg, add_bos);
    size_t max_seq_len = std::max(tokens_pos.size(), tokens_neg.size());
    padding_seq(ctx, tokens_pos, max_seq_len);
    padding_seq(ctx, tokens_neg, max_seq_len);
    cb_data.n_tokens = max_seq_len;
    cb_data.n_embd = llama_n_embd(model);

    cb_data.is_eval_pos = true;
    get_hidden_layers(ctx, tokens_pos);
    cb_data.is_eval_pos = false;
    get_hidden_layers(ctx, tokens_neg);

    printf("%f %f \n", cb_data.v_pos[0][4096], cb_data.v_pos[0][4096]);
    printf("%f %f \n", cb_data.v_neg[0][4096], cb_data.v_neg[0][4096]);

    calc_diff(cb_data);
    printf("%f %f \n", cb_data.v_diff[0][4096], cb_data.v_diff[0][4096]);

    //llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
