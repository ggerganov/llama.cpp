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


static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;

    for (int64_t i0 = 0; i0 < 3; i0++) {
        if (i0 == n && ne[0] > 2*n) {
            printf("..., ");
            i0 = ne[0] - n;
        }
        size_t i = i0;//i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
        float v;
        if (type == GGML_TYPE_F16) {
            v = ggml_fp16_to_fp32(*(ggml_fp16_t *) data + i);
        } else if (type == GGML_TYPE_F32) {
            v = *(float *) data + i;
        } else if (type == GGML_TYPE_I32) {
            v = (float) *((int32_t *) data + i);
        } else if (type == GGML_TYPE_I16) {
            v = (float) *(int16_t *) data + i;
        } else if (type == GGML_TYPE_I8) {
            v = (float) *(int8_t *) data + i;
        } else {
            GGML_ASSERT(false);
        }
        printf("%12.4f", v);
        sum += v;
    }
    printf("\n");




    // for (int64_t i3 = 0; i3 < ne[3]; i3++) {
    //     printf("                                     [\n");
    //     for (int64_t i2 = 0; i2 < ne[2]; i2++) {
    //         if (i2 == n && ne[2] > 2*n) {
    //             printf("                                      ..., \n");
    //             i2 = ne[2] - n;
    //         }
    //         printf("                                      [\n");
    //         for (int64_t i1 = 0; i1 < ne[1]; i1++) {
    //             if (i1 == n && ne[1] > 2*n) {
    //                 printf("                                       ..., \n");
    //                 i1 = ne[1] - n;
    //             }
    //             printf("                                       [");
    //             for (int64_t i0 = 0; i0 < ne[0]; i0++) {
    //                 if (i0 == n && ne[0] > 2*n) {
    //                     printf("..., ");
    //                     i0 = ne[0] - n;
    //                 }
    //                 size_t i = i0;//i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    //                 float v;
    //                 if (type == GGML_TYPE_F16) {
    //                     v = ggml_fp16_to_fp32(*(ggml_fp16_t *) data + i);
    //                 } else if (type == GGML_TYPE_F32) {
    //                     v = *(float *) data + i;
    //                 } else if (type == GGML_TYPE_I32) {
    //                     v = (float) *((int32_t *) data + i);
    //                 } else if (type == GGML_TYPE_I16) {
    //                     v = (float) *(int16_t *) data + i;
    //                 } else if (type == GGML_TYPE_I8) {
    //                     v = (float) *(int8_t *) data + i;
    //                 } else {
    //                     GGML_ASSERT(false);
    //                 }
    //                 printf("%12.4f", v);
    //                 sum += v;
    //                 if (i0 < ne[0] - 1) printf(", ");
    //             }
    //             printf("],\n");
    //         }
    //         printf("                                      ],\n");
    //     }
    //     printf("                                     ]\n");
    //     printf("                                     sum = %f\n", sum);
    // }
}

float Sum(float *arr, int64_t N){
    float s = 0.0;
    for (int i = 0; i < N; i++){
        s += arr[i];
    }
    return s;
}
float PrintArr(const char * name, float * arr, int64_t N){
    float sum = 0.0;
    if (arr != NULL){
        sum =  Sum(arr, N);
        printf("%s %d %10f \n",name, N, sum);
    } else {
        printf("%s %d %10f \n",name, 0, 0.0);
    }
    return sum;
}

size_t get_nth_element(const int64_t *ne, const size_t *nb, int64_t nth) {
    size_t offset = 0;
    size_t divisor = 1;
    for (int i = 3; i >= 0; --i) {
        size_t index = size_t(floor(nth / divisor)) % ne[i];
        offset += index * nb[i]/4;
        divisor *= ne[i];
    }
    return offset;
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
        // int64_t offset = get_nth_element(src0->ne, src0->nb, i);
        // offset *=  ggml_element_size(src0);
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

        int i = 0;
        while (i < ggml_nbytes(src0)/4){
            float val = (((float *) src0->data)[i]);
            float diff = abs(val - 0.0022226818837225437164306640625);
            if (diff < 0.000001 ){
                printf("found %s: %d  =  %f\n", src0->name, i, val);
            }
            i += 1;
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



    // printf("%s\n", buf);
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
    // if (src1) {
    //     sprintf(src1_str, "%s{%s}\n", src1->name, ggml_ne_string(src1).c_str());
    // }

    if (src0) {
        print_tensor(src0);
        // printf("%s{%s} n=%d %f\n", src0->name, ggml_ne_string(src0).c_str(),src0->ne[0], Sum(static_cast<float *>(src0->data), src0->ne[0]));
        // printf("%s{%s}", src0->name, ggml_ne_string(src0).c_str());
        // enum ggml_type type = src0->name == "inp_tokens" ? GGML_TYPE_I32:src0->type;
        // ggml_print_tensor(static_cast<uint8_t *>(src0->data), src0->type, src0->ne, src0->nb, 3);
        // PrintArr(src0->name, static_cast<float *>(src0->data), src0->ne[0]);
    }
    if (src1) {
        print_tensor(src1);
        // printf("%s{%s} n=%d %f\n", src1->name, ggml_ne_string(src1).c_str(),src0->ne[0],  Sum(static_cast<float *>(src1->data), src1->ne[0]));
        // enum ggml_type type = src1->name == "inp_tokens" ? GGML_TYPE_I32:src1->type;
        // ggml_print_tensor(static_cast<uint8_t *>(src1->data), type, src1->ne, src1->nb, 3);
        // ggml_print_tensor(static_cast<uint8_t *>(src1->data), src1->type, src1->ne, src1->nb, 3);
        // PrintArr(src1->name, static_cast<float *>(src1->data), src1->ne[0]);
    }
    printf("%s ==\n", ggml_op_desc(t));
    if (t) {
        print_tensor(t);
        // printf("%s{%s} n=%d %f\n", t->name, ggml_ne_string(t).c_str(),src0->ne[0], Sum(static_cast<float *>(t->data), t->ne[0]));
        // printf("%s{%s}", t->name, ggml_ne_string(t).c_str());
        // PrintArr(t->name, static_cast<float *>(t->data), t->ne[0]);
        // ggml_print_tensor(static_cast<uint8_t *>(t->data), t->type, t->ne, t->nb, 3);
        // printf("\n == \n");
    }
    printf("\n\n");




    // printf("%24s = (%s) %10s(%s{%s}, %s}) = {%s}\n",
    //
    //
    //        t->name, ggml_op_desc(t), src0->name, ggml_ne_string(src0).c_str(),
    //        src1 ? src1_str : "",
    //        ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        // ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
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
