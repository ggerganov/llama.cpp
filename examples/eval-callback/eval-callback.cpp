#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <random>
#include <string>
#include <tuple>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
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

static std::string ggml_nb_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string((t->nb[i]/ggml_element_size(t)));
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

void print_tensor(const ggml_tensor * src0) {
    float sum = 0;

    const int64_t * ne = src0->ne;
    int64_t n = 3;
    ggml_type type = src0->type;
    void * data = src0->data;


    char *buf = static_cast<char *>(malloc(sizeof(char)*ne[0]*8));

    char *buf2 = buf;

    for (int64_t i = 0; i < 1; i++) {
        if (i == n) {
            buf2 += sprintf(buf2, "..., ");
        }
        int64_t offset = i;
        float v;
        if (type == GGML_TYPE_F16) {
            v = ggml_fp16_to_fp32(*(ggml_fp16_t *) data + offset);
        } else if (type == GGML_TYPE_F32) {
            v = *((float *) data + offset);
        } else if (type == GGML_TYPE_I32) {
            v = (float) *((int32_t *) data + offset);
        } else if (type == GGML_TYPE_I16) {
            v = (float) *(int16_t *) data + offset;
        } else if (type == GGML_TYPE_I8) {
            v = (float) *(int8_t *) data + offset;
        } else {
            GGML_ASSERT(false);
        }
        if (i < n) {
            buf2 += sprintf(buf2, "%12.4f", v);
        }
        sum += v;
    }
    int max_name_length = 15;
    int max_dim_length = 15;
    int max_str_length = 15;
    printf("%-*.15s [0]=%.15g dim={%-*.15s} str={%-*.15s} [addr]=%lu\n",
        max_name_length, src0->name,
        sum,
        max_dim_length, ggml_ne_string(src0).c_str(),
        max_str_length, ggml_nb_string(src0).c_str(),
        src0->data);
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
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
    char src1_str[128] = {0};
    if (src0) {
        print_tensor(src0);
    }
    if (src1) {
        print_tensor(src1);
    }
    printf("%s ==\n", ggml_op_desc(t));
    if (t) {
        print_tensor(t);
    }
    printf("\n\n");

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
    }

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
    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    print_build_info();

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
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
