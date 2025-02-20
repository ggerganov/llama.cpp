#include "llama-model.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-graph.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_27B:           return "27B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                int n_expert_used = hparams.n_expert_used;
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                int n_embd_head = hparams.n_embd_head_v;
                int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                // FIXME
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 12345, w->ne[1], 6789);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // FIXME
                const int64_t d_state      = w->ne[0];
                const int64_t d_inner      = w->ne[1];
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 1;
                ggml_tensor * s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, d_inner, n_seqs);
                ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * B = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                ggml_tensor * C = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_state, n_seq_tokens, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd = hparams.n_embd;
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t & buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

// CPU: ACCEL -> CPU extra -> GPU host -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<ggml_backend_dev_t> & devices) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add extra buffer types
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (ggml_backend_dev_get_extra_bufts_fn) {
        ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(cpu_dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    for (auto * dev : devices) {
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        if (buft) {
            buft_list.emplace_back(dev, buft);
            break;
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, enum llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    return buft_list;
}

struct llama_model::impl {
    impl() {}
    ~impl() {}

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;
};

llama_model::llama_model(const struct llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
}

llama_model::~llama_model() {}

void llama_model::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

void llama_model::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.meta.get();

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH, hparams.n_embd_features);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the array hparams
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // for differentiating model types
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    // arch-specific KVs
    switch (arch) {
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 8) {
                    switch (hparams.n_layer) {
                        case 32: type = LLM_TYPE_8x7B; break;
                        case 56: type = LLM_TYPE_8x22B; break;
                        default: type = LLM_TYPE_UNKNOWN;
                    }
                } else {
                    switch (hparams.n_layer) {
                        case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
                        case 22: type = LLM_TYPE_1B; break;
                        case 26: type = LLM_TYPE_3B; break;
                        case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
                        // granite uses a vocab with len 49152
                        case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
                        case 36: type = LLM_TYPE_8B; break; // granite
                        case 40: type = LLM_TYPE_13B; break;
                        case 48: type = LLM_TYPE_34B; break;
                        case 60: type = LLM_TYPE_30B; break;
                        case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
                        default: type = LLM_TYPE_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_DECI:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 80: type = LLM_TYPE_70B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);

                switch (hparams.n_layer) {
                    case 52: type = LLM_TYPE_1B; break;
                    case 40: type = LLM_TYPE_2B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);

                switch (hparams.n_layer) {
                    case 62: type = LLM_TYPE_4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GROK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 64: type = LLM_TYPE_314B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 60: type = LLM_TYPE_40B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                if (type == LLM_TYPE_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 36: type = LLM_TYPE_3B; break;
                    case 42: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_15B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_1B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        type = LLM_TYPE_17M; break; // bge-micro
                    case 6:
                        type = LLM_TYPE_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: type = LLM_TYPE_33M; break; // MiniLM-L12, bge-small
                            case 768: type = LLM_TYPE_109M; break; // bge-base
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        type = LLM_TYPE_335M; break; // bge-large
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4:  type = LLM_TYPE_33M;  break; // jina-embeddings-small
                    case 12: type = LLM_TYPE_137M; break; // jina-embeddings-base
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    type = LLM_TYPE_137M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            case 4096: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_MPT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 48: type = LLM_TYPE_30B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_3B; break;
                    case 40: type = LLM_TYPE_12B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
                    case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 36: type = LLM_TYPE_3B; break;
                    case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
                    case 48: type = LLM_TYPE_14B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    case 80: type = LLM_TYPE_70B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_A2_7B; break;
                    case 28: type = LLM_TYPE_57B_A14B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_3B; break;
                    case 40: type = LLM_TYPE_14B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }

                // for backward compatibility ; see: https://github.com/ggerganov/llama.cpp/pull/8931
                if ((hparams.n_layer == 32 || hparams.n_layer == 40) && hparams.n_ctx_train == 4096) {
                    // default value for Phi-3-mini-4k-instruct and Phi-3-medium-4k-instruct
                    hparams.n_swa = 2047;
                } else if (hparams.n_layer == 32 && hparams.n_head_kv(0) == 32 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-mini-128k-instruct
                    hparams.n_swa = 262144;
                } else if (hparams.n_layer == 40 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-medium-128k-instruct
                    hparams.n_swa = 131072;
                }
                bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (!found_swa && hparams.n_swa == 0) {
                    throw std::runtime_error("invalid value for sliding_window");
                }
            } break;
        case LLM_ARCH_PHIMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_16x3_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: type = LLM_TYPE_SMALL; break;
                    case 24: type = LLM_TYPE_MEDIUM; break;
                    case 36: type = LLM_TYPE_LARGE; break;
                    case 48: type = LLM_TYPE_XL; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: type = LLM_TYPE_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_14B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 48: type = LLM_TYPE_20B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: type = LLM_TYPE_2B; break;
                    case 28: type = LLM_TYPE_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA2:
            {
                hparams.n_swa = 4096; // default value of gemma 2
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,      hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);
                hparams.attn_soft_cap = true;

                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_2B; break;
                    case 42: type = LLM_TYPE_9B; break;
                    case 46: type = LLM_TYPE_27B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: type = LLM_TYPE_3B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_15B; break;
                    case 52: type = LLM_TYPE_20B; break; // granite
                    case 88: type = LLM_TYPE_34B; break; // granite
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_DT_B_C_RMS,     hparams.ssm_dt_b_c_rms, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: type = LLM_TYPE_SMALL; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: type = LLM_TYPE_MEDIUM; break;
                            case 1536: type = LLM_TYPE_LARGE; break;
                            case 2048: type = LLM_TYPE_XL; break;
                            default:   type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_XVERSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    case 80: type = LLM_TYPE_65B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                ml.get_key(LLM_KV_LOGIT_SCALE,             hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 40: type = LLM_TYPE_35B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2:
            {
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_LOGIT_SCALE,              hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DBRX:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
            ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv);

            switch (hparams.n_layer) {
                case 40: type = LLM_TYPE_16x12B; break;
                default: type = LLM_TYPE_UNKNOWN;
            }
        } break;
        case LLM_ARCH_OLMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

                switch (hparams.n_layer) {
                    case 22: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 80: type = LLM_TYPE_70B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OLMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_1B; break;
                    case 32: type = LLM_TYPE_7B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OLMOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 16: type = LLM_TYPE_A1_7B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                case 16: type = LLM_TYPE_270M; break;
                case 20: type = LLM_TYPE_450M; break;
                case 28: type = LLM_TYPE_1B; break;
                case 36: type = LLM_TYPE_3B; break;
                default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL,   hparams.use_par_res);
                switch (hparams.n_layer) {
                    case 6:
                        switch (hparams.n_ff()) {
                            case 512:  type = LLM_TYPE_14M; break;
                            case 2048: type = LLM_TYPE_70M; break;
                            default:   type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: type = LLM_TYPE_160M; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 16:
                        switch (hparams.n_ff()) {
                            case 8192: type = LLM_TYPE_1B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096: type = LLM_TYPE_410M; break;
                            case 8192: type = LLM_TYPE_1_4B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_ff()) {
                            case 10240: type = LLM_TYPE_2_8B; break;
                            case 16384: type = LLM_TYPE_6_9B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 36:
                        switch (hparams.n_ff()) {
                            case 20480: type = LLM_TYPE_12B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 44:
                        switch (hparams.n_ff()) {
                            case 24576: type = LLM_TYPE_20B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ARCTIC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 128) {
                    switch (hparams.n_layer) {
                        case 35: type = LLM_TYPE_10B_128x3_66B; break;
                        default: type = LLM_TYPE_UNKNOWN;
                    }
                } else {
                    type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);

                switch (hparams.n_layer) {
                    case 28: type = LLM_TYPE_20B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                bool is_lite = (hparams.n_layer == 27);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // for compatibility with existing DeepSeek V2 and V2.5 GGUFs
                    // that have no expert_gating_func model parameter set
                    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul);

                switch (hparams.n_layer) {
                    case 27: type = LLM_TYPE_16B; break;
                    case 60: type = LLM_TYPE_236B; break;
                    case 61: type = LLM_TYPE_671B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHATGLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: {
                        if (hparams.n_head(0) == 16) {
                            type = LLM_TYPE_1_5B;
                        } else {
                            type = LLM_TYPE_6B;
                        }
                    } break;
                    case 40: {
                        if (hparams.n_head(0) == 24) {
                            type = LLM_TYPE_4B;
                        } else {
                            type = LLM_TYPE_9B;
                        }
                    } break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: type = LLM_TYPE_3B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_T5:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

                uint32_t dec_start_token_id;
                if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
                    hparams.dec_start_token_id = dec_start_token_id;
                }

                switch (hparams.n_layer) {
                    case 6:  type = LLM_TYPE_60M;  break; // t5-small
                    case 8:  type = LLM_TYPE_80M;  break; // flan-t5-small
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: type = LLM_TYPE_220M; break; // t5-base
                            case 2048: type = LLM_TYPE_250M; break; // flan-t5-base
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096:  type = LLM_TYPE_770M; break; // t5-large
                            case 2816:  type = LLM_TYPE_780M; break; // flan-t5-large
                            case 16384: type = LLM_TYPE_3B;   break; // t5-3b
                            case 5120:  type = LLM_TYPE_3B;   break; // flan-t5-xl
                            case 65536: type = LLM_TYPE_11B;  break; // t5-11b
                            case 10240: type = LLM_TYPE_11B;  break; // flan-t5-xxl
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
                type = LLM_TYPE_UNKNOWN;
            } break;
        case LLM_ARCH_JAIS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1_3B; break;
                    case 40: type = LLM_TYPE_13B; break;
                    /* TODO: add variants */
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_NEMOTRON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_EXAONE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_8B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
                ml.get_key(LLM_KV_WKV_HEAD_SIZE,               hparams.wkv_head_size);
                ml.get_key(LLM_KV_TIME_MIX_EXTRA_DIM,          hparams.time_mix_extra_dim);
                ml.get_key(LLM_KV_TIME_DECAY_EXTRA_DIM,        hparams.time_decay_extra_dim);
                ml.get_key(LLM_KV_RESCALE_EVERY_N_LAYERS,      hparams.rescale_every_n_layers, false);
                ml.get_key(LLM_KV_TOKEN_SHIFT_COUNT,           hparams.token_shift_count, false);

                switch (hparams.n_layer) {
                    case 24: type = LLM_TYPE_1_6B; break;
                    case 32:
                        switch (hparams.n_embd) {
                            case 2560: type = LLM_TYPE_3B; break;
                            case 4096: type = LLM_TYPE_7B; break;
                            default: type = LLM_TYPE_UNKNOWN;
                        } break;
                    case 61: type = LLM_TYPE_14B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE,             hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE,             hparams.f_attention_scale);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_3B; break;
                    case 40: type = LLM_TYPE_3B; break;
                    // Add additional layer/vocab/etc checks here for other model sizes
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHAMELEON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.f_norm_eps = 1e-5;  // eps for qk-norm, torch default
                ml.get_key(LLM_KV_SWIN_NORM, hparams.swin_norm);

                switch (hparams.n_layer) {
                    case 32: type = LLM_TYPE_7B; break;
                    case 48: type = LLM_TYPE_34B; break;
                    default: type = LLM_TYPE_UNKNOWN;
               }
            } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_EPS,    hparams.f_norm_group_eps);
                ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
            } break;
        default: throw std::runtime_error("unsupported model architecture");
    }

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & n_gpu_layers = params.n_gpu_layers;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer = hparams.n_layer;

    const bool use_mmap_buffer = true;

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s)\n", __func__, ml.use_mmap ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices);
    for (auto * dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i];
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const int i_gpu_start = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, (int)n_layer + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s\n", il, ggml_backend_dev_name(cpu_dev));
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu);
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s\n", il, ggml_backend_dev_name(dev));
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    // one ggml context per buffer type
    int max_n_tensors = ml.n_tensors;
    max_n_tensors += 1;         // duplicated output tensor
    max_n_tensors += n_layer*2; // duplicated rope freq tensors
    const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map[buft] = ctx;
            pimpl->ctxs.emplace_back(ctx);

            return ctx;
        }
        return it->second;
    };

    const auto TENSOR_DUPLICATED   = llama_model_loader::TENSOR_DUPLICATED;
    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        const int64_t n_head        = hparams.n_head();
        const int64_t n_head_kv     = hparams.n_head_kv();
        const int64_t n_embd        = hparams.n_embd;
        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        const int64_t n_embd_head_v = hparams.n_embd_head_v;
        const int64_t n_ff          = hparams.n_ff();
        const int64_t n_embd_gqa    = n_embd_v_gqa;
        const int64_t n_vocab       = vocab.n_tokens();
        const int64_t n_token_types = vocab.n_token_types();
        const int64_t n_rot         = hparams.n_rot;
        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;
        const int64_t n_ctx_train   = hparams.n_ctx_train;

        if (n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        int n_moved_tensors = 0;
        ggml_tensor * first_moved_tensor = nullptr;
        ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
        ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

        auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
            ggml_tensor * t_meta = ml.get_tensor_meta(tn.str().c_str());

            if (!t_meta) {
                if (flags & TENSOR_NOT_REQUIRED) {
                    return nullptr;
                }
                throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
            }

            // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
            // the tensor is duplicated
            // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
            llm_tensor tn_tensor = tn.tensor;
            if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && flags & TENSOR_DUPLICATED) {
                tn_tensor = LLM_TENSOR_OUTPUT;
            }

            llm_tensor_info info;
            try {
                info = llm_tensor_info_for(tn_tensor);
            } catch (const std::out_of_range & e) {
                throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
            }

            // tensors with "bias" suffix are always used with GGML_OP_ADD
            ggml_op op;
            bool bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
            if (bias) {
                op = GGML_OP_ADD;
            } else {
                op = info.op;
            }

            // sanity checks
            if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
                if (tn.bid != -1) {
                    GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
                }
            } else {
                if (tn.bid == -1) {
                    GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
                }
            }

            // select the buffer type for this tensor
            buft_list_t * buft_list;
            switch (info.layer) {
                case LLM_TENSOR_LAYER_INPUT:
                    buft_list = pimpl->dev_input.buft_list;
                    break;
                case LLM_TENSOR_LAYER_OUTPUT:
                    buft_list = pimpl->dev_output.buft_list;
                    break;
                case LLM_TENSOR_LAYER_REPEATING:
                    buft_list = pimpl->dev_layer.at(tn.bid).buft_list;
                    break;
                default:
                    GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
            }

            ggml_backend_buffer_type_t buft = select_weight_buft(hparams, t_meta, op, *buft_list);
            if (!buft) {
                throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
            }

            // avoid using a host buffer when using mmap
            auto * buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                buft = ggml_backend_dev_buffer_type(cpu_dev);
            }

            if (buft != buft_list->front().second) {
                n_moved_tensors++;
                if (!first_moved_tensor) {
                    first_moved_tensor = t_meta;
                    first_moved_from_buft = buft_list->front().second;
                    first_moved_to_buft   = buft;
                }
            }

            ggml_context * ctx = ctx_for_buft(buft);

            // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
            if (flags & TENSOR_DUPLICATED) {
                ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
                if (t) {
                    return t;
                }
            }
            return ml.create_tensor(ctx, tn, ne, flags);
        };

        layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);
        switch (arch) {
            case LLM_ARCH_LLAMA:
            case LLM_ARCH_REFACT:
            case LLM_ARCH_MINICPM:
            case LLM_ARCH_GRANITE:
            case LLM_ARCH_GRANITE_MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }
                        else {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }

                        if (n_expert == 0) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                            // optional MLP bias
                            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                        } else {
                            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_DECI:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];
                        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
                        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);
                        const int64_t n_embd_gqa    = hparams.n_embd_v_gqa(i);
                        const int64_t n_ff          = hparams.n_ff(i);
                        const int64_t n_head        = hparams.n_head(i);
                        const int64_t n_head_kv     = hparams.n_head_kv(i);

                        if (n_head_kv == 0 && n_head > 0) {
                            // linear attention for DeciLMCausalModel
                            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        }
                        else if (n_head_kv > 0) {
                            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
                        }

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }
                        else {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        }

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        // optional MLP bias
                        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_MINICPM3:
                {
                    const int64_t n_embd_head_qk_rope = hparams.n_rot;
                    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

                    const int64_t q_lora_rank  = hparams.n_lora_q;
                    const int64_t kv_lora_rank = hparams.n_lora_kv;
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);

                        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

                        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);

                        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + (n_embd_head_qk_rope)}, 0);
                        layer.wkv_b     = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,     "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)}, 0);
                        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {              n_head * (                      n_embd_head_v), n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_embd_head_qk_rope/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head_qk_rope/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                    }
                } break;
            case LLM_ARCH_GROK:
                {
                    if (n_expert == 0) {
                        throw std::runtime_error("Grok model cannot have zero experts");
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff, n_expert}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

                        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_DBRX:
                {
                    if (n_expert == 0) {
                        throw std::runtime_error("DBRX model cannot have zero experts");
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_BAICHUAN:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    {
                        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_FALCON:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    {
                        output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

                        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        if (!output) {
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // needs to be on GPU
                        }
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_STARCODER:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, 0);

                    // output
                    {
                        output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                        output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                        if (!output) {
                            // needs to be on GPU
                            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                        }

                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff}, 0);
                        layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_BERT:
            case LLM_ARCH_NOMIC_BERT:
                {
                    tok_embd     = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
                    type_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0);

                    if (arch == LLM_ARCH_BERT) {
                        pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,    "weight"), {n_embd, n_ctx_train}, 0);

                        cls   = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                        cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"),   {n_embd},         TENSOR_NOT_REQUIRED);

                        cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, 1}, TENSOR_NOT_REQUIRED);
                        cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {1},         TENSOR_NOT_REQUIRED);
                    }

                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        if (arch == LLM_ARCH_BERT) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i),   {n_embd}, 0);

                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i),   {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i),   {n_embd_gqa}, 0);
                        } else {
                            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        }

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_embd, n_embd}, 0);

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);

                        if (arch == LLM_ARCH_BERT) {
                            layer.bo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);
                            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, 0);
                            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, 0);
                        } else {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                        }

                        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_JINA_BERT_V2:
                {
                    tok_embd  = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0); // word_embeddings
                    type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_token_types}, 0); // token_type_embeddings

                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0); // LayerNorm
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0); //LayerNorm bias

                    cls   = create_tensor(tn(LLM_TENSOR_CLS, "weight"), {n_embd, 1}, TENSOR_NOT_REQUIRED);
                    cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"),   {1},         TENSOR_NOT_REQUIRED);
                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i]; // JinaBertLayer

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd}, 0);

                        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa}, 0);

                        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0); //output_dens
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0); //output_dens

                        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0); //output_norm
                        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

                        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
                        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_BLOOM:
                {
                    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);
                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   i), {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias",   i), {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_MPT:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, TENSOR_NOT_REQUIRED);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    if (!output) {
                        output    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // needs to be on GPU
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, TENSOR_NOT_REQUIRED);

                        layer.attn_q_norm   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.attn_k_norm   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

                        // AWQ ScaleActivation layer
                        layer.ffn_act = create_tensor(tn(LLM_TENSOR_FFN_ACT, "scales", i), {n_ff}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_STABLELM:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm =   create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors, present in Stable LM 2 1.6B
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        // optional q and k layernorms, present in StableLM 2 12B
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head},    TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, TENSOR_NOT_REQUIRED);

                        // optional FFN norm, not present in StableLM 2 12B which uses parallel residual
                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, TENSOR_NOT_REQUIRED);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd*3}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd*3}, 0);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff/2}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff/2, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff/2}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN2:
            case LLM_ARCH_QWEN2VL:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_QWEN2MOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        if (n_expert == 0) {
                            throw std::runtime_error("n_expert must be > 0 for QWEN2MOE");
                        }
                        if (n_expert_used == 0) {
                            throw std::runtime_error("n_expert_used must be > 0 for QWEN2MOE");
                        }

                        // MoE branch
                        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                        // Shared expert branch
                        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

                        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp}, 0);
                        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd}, 0);
                        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp}, 0);
                    }
                } break;
            case LLM_ARCH_PHI2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
                    output_b      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, TENSOR_NOT_REQUIRED);

                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd}, 0);

                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i),   {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i),   {n_embd_gqa}, 0);
                        }

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_PHI3:
                {
                    const int64_t n_embd_head = n_embd / n_head;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, TENSOR_NOT_REQUIRED);
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff }, 0);

                        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_embd_head/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                    }
                } break;
            case LLM_ARCH_PHIMOE:
                {
                    const int64_t n_embd_head = n_embd / n_head;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, 0);
                    output_b      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   { n_vocab }, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias",   i), { n_embd }, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias",   i), {n_embd}, 0);

                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa}, 0);

                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa}, 0);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa}, 0);
                        }
                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), { n_embd }, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), { n_embd }, 0);

                        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert},         0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);

                        layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), { n_embd_head/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head/2 }, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                     }
                } break;
            case LLM_ARCH_PLAMO:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_GPT2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
                    pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_CODESHELL:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_ORION:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_INTERNLM2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        // layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_GEMMA:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_GEMMA2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_STARCODER2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, 0);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, 0);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        // optional bias tensors
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP ,  "bias", i), {  n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_MAMBA:
                {
                    const int64_t d_conv  = hparams.ssm_d_conv;
                    const int64_t d_inner = hparams.ssm_d_inner;
                    const int64_t d_state = hparams.ssm_d_state;
                    const int64_t dt_rank = hparams.ssm_dt_rank;

                    // only an expansion factor of 2 is supported for now
                    if (2 * n_embd != d_inner) {
                        throw std::runtime_error("only an expansion factor of 2 is supported for now");
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed, duplicated to allow offloading
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        // norm
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, 2*d_inner}, 0);

                        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner}, 0);
                        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner}, 0);

                        layer.ssm_x = create_tensor(tn(LLM_TENSOR_SSM_X, "weight", i), {d_inner, dt_rank + 2*d_state}, 0);

                        layer.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_rank, d_inner}, 0);
                        layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

                        // no "weight" suffix for these
                        layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {d_state, d_inner}, 0);
                        layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {d_inner}, 0);

                        // out_proj
                        layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_XVERSE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_COMMAND_R:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    // init output from the input tok embed
                    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        if (n_layer >= 64){
                            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head}, 0);
                            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, 0);
                        }

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_COHERE2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                    // init output from the input tok embed
                    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab },
                                                      TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                    }
                }
                break;
            case LLM_ARCH_OLMO:  // adapted from LLM_ARCH_LLAMA with norm params removed
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_OLMO2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_OLMOE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                        if (n_expert == 0) {
                            throw std::runtime_error("n_expert must be > 0");
                        }
                        if (n_expert_used == 0) {
                            throw std::runtime_error("n_expert_used must be > 0");
                        }

                        // MoE branch
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert}, 0);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_OPENELM:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    // init output from the input tok embed
                    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

                    for (int i = 0; i < n_layer; ++i) {
                        const int64_t n_head      =   hparams.n_head(i);
                        const int64_t n_head_qkv  = 2*hparams.n_head_kv(i) + n_head;
                        const int64_t n_ff        =   hparams.n_ff(i);

                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_head_qkv*n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*n_embd_head_k, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_GPTNEOX:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_ARCTIC:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                        layer.ffn_norm_exps = create_tensor(tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, false);
                        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
                        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);
                    }
                } break;
            case LLM_ARCH_DEEPSEEK:
                {

                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (i < (int) hparams.n_layer_dense_lead) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        } else {
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

                            if (n_expert == 0) {
                                throw std::runtime_error("n_expert must be > 0");
                            }
                            if (n_expert_used == 0) {
                                throw std::runtime_error("n_expert_used must be > 0");
                            }

                            // MoE branch
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                            // Shared expert branch
                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_DEEPSEEK2:
                {
                    const bool is_lite = (hparams.n_layer == 27);

                    const int64_t n_embd_head_qk_rope = hparams.n_rot;
                    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

                    const int64_t q_lora_rank  = hparams.n_lora_q;
                    const int64_t kv_lora_rank = hparams.n_lora_kv;

                    const int64_t n_ff_exp        = hparams.n_ff_exp;
                    const int64_t n_expert_shared = hparams.n_expert_shared;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        if (!is_lite) {
                            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
                        }

                        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

                        if (!is_lite) {
                            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);
                        } else {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        }

                        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + (n_embd_head_qk_rope)}, 0);
                        layer.wkv_b     = create_tensor(tn(LLM_TENSOR_ATTN_KV_B,     "weight", i), {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)}, 0);
                        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {              n_head * (                      n_embd_head_v), n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        if (i < (int) hparams.n_layer_dense_lead) {
                            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                        } else {
                            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
                            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

                            if (n_expert == 0) {
                                throw std::runtime_error("n_expert must be > 0");
                            }
                            if (n_expert_used == 0) {
                                throw std::runtime_error("n_expert_used must be > 0");
                            }

                            // MoE branch
                            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
                            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
                            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

                            // Shared expert branch
                            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
                            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
                        }
                    }
                } break;
            case LLM_ARCH_BITNET:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
                        layer.attn_sub_norm = create_tensor(tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd}, 0);

                        layer.wq       = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wq_scale = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.wk       = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wk_scale = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.wv       = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv_scale = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.wo_scale = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale",  i), {1}, TENSOR_NOT_REQUIRED);

                        layer.ffn_norm     = create_tensor(tn(LLM_TENSOR_FFN_NORM,     "weight", i), {n_embd}, 0);
                        layer.ffn_sub_norm = create_tensor(tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff}, 0);

                        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_gate_scale = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down       = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_scale = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up         = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_scale   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_T5:
                {
                    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm_enc  = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        layer.attn_rel_b = create_tensor(tn(LLM_TENSOR_DEC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.attn_norm_cross  = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        // this tensor seems to be unused in HF transformers implementation
                        layer.attn_rel_b_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_T5ENCODER:
                {
                    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm_enc  = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd}, 0);
                        layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, TENSOR_NOT_REQUIRED);

                        layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd}, 0);

                        layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, TENSOR_NOT_REQUIRED);
                        layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up_enc   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_JAIS:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd}, 0);

                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, 0);

                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

                        layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE,   "bias", i),   {n_ff}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_CHATGLM:
                {
                    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

                        if (layer.wqkv == nullptr) {
                            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                            layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
                            layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
                            layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        }

                        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2}, 0);

                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
                    }
                } break;
            case LLM_ARCH_NEMOTRON:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
                    output        = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        // optional bias tensors
                        layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);
                        layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, TENSOR_NOT_REQUIRED);
                        layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     TENSOR_NOT_REQUIRED);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), {n_embd}, 0);

                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

                        // optional MLP bias
                        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
                    }
                } break;
            case LLM_ARCH_EXAONE:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM,   "weight", i), {n_embd}, 0);
                        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
                        layer.ffn_gate   = create_tensor(tn(LLM_TENSOR_FFN_GATE,   "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN,   "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,     "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_RWKV6:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // Block 0, LN0
                    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), {n_embd}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, 0);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    const int time_mix_extra_dim = hparams.time_mix_extra_dim;
                    const int time_decay_extra_dim = hparams.time_decay_extra_dim;
                    const int head_size = hparams.wkv_head_size;
                    const int attn_hidden_size = n_embd;
                    const int ffn_size = hparams.n_ff_arr[0];

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

                        layer.attn_norm_2   = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, 0);
                        layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, 0);

                        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, time_mix_extra_dim * 5}, 0);
                        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {time_mix_extra_dim, n_embd, 5}, 0);

                        layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), {n_embd, 1, 1}, 0);
                        layer.time_mix_lerp_w = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_W, "weight", i), {n_embd, 1, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_v = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_V, "weight", i), {n_embd, 1, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_r = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_R, "weight", i), {n_embd, 1, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_g = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_G, "weight", i), {n_embd, 1, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        GGML_ASSERT(!(layer.time_mix_lerp_fused == NULL && layer.time_mix_lerp_w == NULL));

                        layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), {head_size, n_embd / head_size}, 0);
                        layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), {n_embd}, 0);
                        layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), {n_embd, time_decay_extra_dim}, 0);
                        layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), {time_decay_extra_dim, attn_hidden_size}, 0);
                        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), {attn_hidden_size, n_embd}, 0);

                        layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), {n_embd}, 0);
                        layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), {n_embd}, 0);
                        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

                        layer.channel_mix_lerp_k = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), {n_embd, 1, 1}, 0);
                        layer.channel_mix_lerp_r = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_R, "weight", i), {n_embd, 1, 1}, 0);

                        layer.channel_mix_key = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_KEY, "weight", i), {n_embd, ffn_size}, 0);
                        layer.channel_mix_value = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_VALUE, "weight", i), {ffn_size, n_embd}, 0);
                        layer.channel_mix_receptance = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_RECEPTANCE, "weight", i), {n_embd, n_embd}, 0);
                    }

                } break;
            case LLM_ARCH_RWKV6QWEN2:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

                    const int time_mix_extra_dim = hparams.time_mix_extra_dim;
                    const int time_decay_extra_dim = hparams.time_decay_extra_dim;
                    const int head_size = hparams.wkv_head_size;
                    const int attn_hidden_size = n_embd;
                    const int n_head_kv = hparams.n_head_kv();
                    int attn_key_value_size;
                    if (n_head_kv == 0 || attn_hidden_size / head_size == n_head_kv) {
                        attn_key_value_size = attn_hidden_size;
                    } else {
                        attn_key_value_size = n_head_kv * head_size;
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), {n_embd, time_mix_extra_dim * 5}, 0);
                        layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), {time_mix_extra_dim, n_embd, 5}, 0);

                        layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), {n_embd, 1, 1}, 0);
                        layer.time_mix_lerp_fused = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_FUSED, "weight", i), {n_embd, 1, 1, 5}, 0);

                        layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), {head_size, n_embd / head_size}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), {n_embd}, 0);
                        layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), {n_embd, time_decay_extra_dim}, 0);
                        layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), {time_decay_extra_dim, attn_hidden_size}, 0);
                        layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), {n_embd, attn_key_value_size}, 0);
                        layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), {n_embd, attn_key_value_size}, 0);
                        layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), {attn_hidden_size, n_embd}, 0);
                        // optional bias tensors
                        layer.time_mix_key_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "bias", i), {attn_key_value_size}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_value_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "bias", i), {attn_key_value_size}, llama_model_loader::TENSOR_NOT_REQUIRED);
                        layer.time_mix_receptance_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "bias", i), {attn_hidden_size}, llama_model_loader::TENSOR_NOT_REQUIRED);

                        layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), {n_embd, attn_hidden_size}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_CHAMELEON:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head}, 0);
                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, 0);
                        layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias", i),  {n_embd_head_k, n_head}, TENSOR_NOT_REQUIRED);
                        layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias", i),  {n_embd_head_k, n_head_kv}, TENSOR_NOT_REQUIRED);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            case LLM_ARCH_WAVTOKENIZER_DEC:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hparams.n_embd_features, n_vocab}, 0);

                    conv1d   = create_tensor(tn(LLM_TENSOR_CONV1D, "weight"), {7, hparams.n_embd_features, hparams.posnet.n_embd}, 0);
                    conv1d_b = create_tensor(tn(LLM_TENSOR_CONV1D, "bias"),   {1, hparams.posnet.n_embd}, 0);

                    // posnet
                    {
                        const int64_t n_embd = hparams.posnet.n_embd;

                        for (uint32_t i = 0; i < hparams.posnet.n_layer; ++i) {
                            auto & layer = layers[i].posnet;

                            // posnet:
                            //
                            //  - resnet
                            //  - resnet
                            //  - attn
                            //  - resnet
                            //  - resnet
                            //  - norm
                            //
                            switch (i) {
                                case 0:
                                case 1:
                                case 3:
                                case 4:
                                    {
                                        layer.norm1   = create_tensor(tn(LLM_TENSOR_POS_NET_NORM1, "weight", i), {1, n_embd}, 0);
                                        layer.norm1_b = create_tensor(tn(LLM_TENSOR_POS_NET_NORM1, "bias",   i), {1, n_embd}, 0);

                                        layer.conv1   = create_tensor(tn(LLM_TENSOR_POS_NET_CONV1, "weight", i), {3, n_embd, n_embd}, 0);
                                        layer.conv1_b = create_tensor(tn(LLM_TENSOR_POS_NET_CONV1, "bias",   i), {1, n_embd}, 0);

                                        layer.norm2   = create_tensor(tn(LLM_TENSOR_POS_NET_NORM2, "weight", i), {1, n_embd}, 0);
                                        layer.norm2_b = create_tensor(tn(LLM_TENSOR_POS_NET_NORM2, "bias",   i), {1, n_embd}, 0);

                                        layer.conv2   = create_tensor(tn(LLM_TENSOR_POS_NET_CONV2, "weight", i), {3, n_embd, n_embd}, 0);
                                        layer.conv2_b = create_tensor(tn(LLM_TENSOR_POS_NET_CONV2, "bias",   i), {1, n_embd}, 0);
                                    } break;
                                case 2:
                                    {
                                        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "weight", i), {1, n_embd}, 0);
                                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "bias",   i), {1, n_embd}, 0);

                                        layer.attn_q      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_Q,    "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_q_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_Q,    "bias",   i), {1, n_embd}, 0);

                                        layer.attn_k      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_K,    "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_k_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_K,    "bias",   i), {1, n_embd}, 0);

                                        layer.attn_v      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_V,    "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_v_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_V,    "bias",   i), {1, n_embd}, 0);

                                        layer.attn_o      = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_OUT,  "weight", i), {1, n_embd, n_embd}, 0);
                                        layer.attn_o_b    = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_OUT,  "bias",   i), {1, n_embd}, 0);
                                    } break;
                                case 5:
                                    {
                                        layer.norm   = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "weight", i), {1, n_embd}, 0);
                                        layer.norm_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "bias",   i), {1, n_embd}, 0);
                                    } break;
                                default: GGML_ABORT("unknown posnet layer");
                            };
                        }
                    }

                    GGML_ASSERT(hparams.posnet.n_embd == hparams.convnext.n_embd);

                    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {hparams.posnet.n_embd}, 0);
                    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {hparams.posnet.n_embd}, 0);

                    // convnext
                    {
                        const int64_t n_embd = hparams.convnext.n_embd;

                        for (uint32_t i = 0; i < hparams.convnext.n_layer; ++i) {
                            auto & layer = layers[i].convnext;

                            layer.dw     = create_tensor(tn(LLM_TENSOR_CONVNEXT_DW,    "weight", i), {7, 1, n_embd}, 0);
                            layer.dw_b   = create_tensor(tn(LLM_TENSOR_CONVNEXT_DW,    "bias",   i), {1, n_embd}, 0);

                            layer.norm   = create_tensor(tn(LLM_TENSOR_CONVNEXT_NORM,  "weight", i), {n_embd}, 0);
                            layer.norm_b = create_tensor(tn(LLM_TENSOR_CONVNEXT_NORM,  "bias",   i), {n_embd}, 0);

                            layer.pw1    = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW1,   "weight", i), {n_embd, n_ff}, 0);
                            layer.pw1_b  = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW1,   "bias",   i), {n_ff}, 0);

                            layer.pw2    = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW2,   "weight", i), {n_ff, n_embd}, 0);
                            layer.pw2_b  = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW2,   "bias",   i), {n_embd}, 0);

                            layer.gamma  = create_tensor(tn(LLM_TENSOR_CONVNEXT_GAMMA, "weight", i), {n_embd}, 0);
                        }

                        // output
                        output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                        output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
                    }

                    output   = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {hparams.convnext.n_embd, n_embd}, 0);
                    output_b = create_tensor(tn(LLM_TENSOR_OUTPUT, "bias"),   {n_embd}, 0);
                } break;
            default:
                throw std::runtime_error("unknown architecture");
        }

        if (n_moved_tensors > 0) {
            LLAMA_LOG_DEBUG("%s: tensor '%s' (%s) (and %d others) cannot be used with preferred buffer type %s, using %s instead\n",
                __func__, first_moved_tensor->name, ggml_type_name(first_moved_tensor->type), n_moved_tensors - 1,
                ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
        }
    }

    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_bufs;
    ctx_bufs.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    pimpl->bufs.reserve(n_max_backend_buffer);

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx              = it.second;

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                pimpl->bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        }
        else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            pimpl->bufs.emplace_back(buf);
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        if (pimpl->bufs.empty()) {
            throw std::runtime_error("failed to allocate buffer");
        }

        for (auto & buf : buf_map) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_bufs.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & buf : pimpl->bufs) {
        LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (auto & ctx : pimpl->ctxs) {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur != NULL; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto & it : ctx_bufs) {
        ggml_context * ctx = it.first;
        auto & bufs = it.second;
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const char * rope_scaling_type = LLAMA_ROPE_SCALING_TYPES.at(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch             = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only       = %d\n",     __func__, hparams.vocab_only);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer          = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head           = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv        = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot            = %u\n",     __func__, hparams.n_rot);
        LLAMA_LOG_INFO("%s: n_swa            = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: n_embd_head_k    = %u\n",     __func__, hparams.n_embd_head_k);
        LLAMA_LOG_INFO("%s: n_embd_head_v    = %u\n",     __func__, hparams.n_embd_head_v);
        LLAMA_LOG_INFO("%s: n_gqa            = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa     = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa     = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale    = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: n_ff             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert         = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used    = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n",     __func__, rope_scaling_type);
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        LLAMA_LOG_INFO("%s: ssm_d_conv       = %u\n",     __func__, hparams.ssm_d_conv);
        LLAMA_LOG_INFO("%s: ssm_d_inner      = %u\n",     __func__, hparams.ssm_d_inner);
        LLAMA_LOG_INFO("%s: ssm_d_state      = %u\n",     __func__, hparams.ssm_d_state);
        LLAMA_LOG_INFO("%s: ssm_dt_rank      = %u\n",     __func__, hparams.ssm_dt_rank);
        LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms   = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
    }

    LLAMA_LOG_INFO("%s: model type       = %s\n",     __func__, type_name().c_str());
    if (pimpl->n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.2f T\n", __func__, pimpl->n_elements*1e-12);
    } else if (pimpl->n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, pimpl->n_elements*1e-9);
    } else if (pimpl->n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.2f M\n", __func__, pimpl->n_elements*1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params     = %.2f K\n", __func__, pimpl->n_elements*1e-3);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n",    __func__, name.c_str());

    if (arch == LLM_ARCH_DEEPSEEK) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
    }

    if (arch == LLM_ARCH_DEEPSEEK2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_lora_q             = %d\n",     __func__, hparams.n_lora_q);
        LLAMA_LOG_INFO("%s: n_lora_kv            = %d\n",     __func__, hparams.n_lora_kv);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((enum llama_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul    = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
    }

    if (arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n",     __func__, hparams.n_ff_shexp);
    }

    if (arch == LLM_ARCH_MINICPM || arch == LLM_ARCH_GRANITE || arch == LLM_ARCH_GRANITE_MOE) {
        LLAMA_LOG_INFO("%s: f_embedding_scale = %f\n", __func__, hparams.f_embedding_scale);
        LLAMA_LOG_INFO("%s: f_residual_scale  = %f\n", __func__, hparams.f_residual_scale);
        LLAMA_LOG_INFO("%s: f_attention_scale = %f\n", __func__, hparams.f_attention_scale);
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

const struct ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, struct ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

//
// llm_build
//

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

struct llm_build_context {
    const llama_model   & model;
    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_ctx_orig;

    const bool worst_case;
    const bool flash_attn;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_context * ctx0 = nullptr;
    llama_graph_i * lgf = nullptr;

    llama_graph_result res;

    // TODO: consider making the entire interface noexcept
    llm_build_context(
            ggml_context  * ctx,
            llama_graph_i * lgf,
      const llama_model   & model,
      const llama_cparams & cparams,
      const llama_ubatch  & ubatch,
            bool            worst_case) :
        model            (model),
        hparams          (model.hparams),
        cparams          (cparams),
        ubatch           (ubatch),
        n_embd           (hparams.n_embd),
        n_layer          (hparams.n_layer),
        n_rot            (hparams.n_rot),
        n_ctx            (cparams.n_ctx),
        n_head           (hparams.n_head()),
        n_head_kv        (hparams.n_head_kv()),
        n_embd_head_k    (hparams.n_embd_head_k),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v),
        n_embd_v_gqa     (hparams.n_embd_v_gqa()),
        n_expert         (hparams.n_expert),
        n_expert_used    (hparams.n_expert_used),
        freq_base        (cparams.rope_freq_base),
        freq_scale       (cparams.rope_freq_scale),
        ext_factor       (cparams.yarn_ext_factor),
        attn_factor      (cparams.yarn_attn_factor),
        beta_fast        (cparams.yarn_beta_fast),
        beta_slow        (cparams.yarn_beta_slow),
        norm_eps         (hparams.f_norm_eps),
        norm_rms_eps     (hparams.f_norm_rms_eps),
        n_tokens         (ubatch.n_tokens),
        n_ctx_orig       (cparams.n_ctx_orig_yarn),
        worst_case       (worst_case),
        flash_attn       (cparams.flash_attn),
        pooling_type     (cparams.pooling_type),
        rope_type        (hparams.rope_type),
        ctx0             (ctx),
        lgf              (lgf) {
        }

    // TODO: tmp
    void cb(struct ggml_tensor * cur, const char * name, int il) {
        lgf->build_cb(cur, name, ubatch, il);
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_embd(struct ggml_tensor * tok_embd) {
        struct ggml_tensor * inpL = lgf->build_inp_embd(ctx0, tok_embd, ubatch);
        cb(inpL, "inp_embd", -1);

        return inpL;
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_pos() {
        ggml_tensor * cur = lgf->build_inp_pos(ctx0, n_tokens);
        cb(cur, "inp_pos", -1);

        return cur;
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_out_ids() {
        ggml_tensor * cur = lgf->build_inp_out_ids(ctx0, n_tokens, worst_case);
        cb(cur, "inp_out_ids", -1);

        return cur;
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_mean() {
        ggml_tensor * cur = lgf->build_inp_mean(ctx0, n_tokens);
        cb(cur, "inp_mean", -1);

        return cur;
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_cls() {
        ggml_tensor * cur = lgf->build_inp_cls(ctx0, n_tokens);
        cb(cur, "inp_cls", -1);

        return cur;
    }

    // TODO: tmp
    struct ggml_tensor * build_lora_mm(
              struct ggml_tensor * w,
              struct ggml_tensor * cur) {
        return lgf->build_lora_mm(ctx0, w, cur);
    }

    // TODO: tmp
    struct ggml_tensor * build_lora_mm_id(
              struct ggml_tensor * w,   // struct ggml_tensor * as
              struct ggml_tensor * cur, // struct ggml_tensor * b
              struct ggml_tensor * ids) {
        return lgf->build_lora_mm_id(ctx0, w, cur, ids);
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_embd_enc() {
        ggml_tensor * cur = lgf->build_inp_embd_enc(ctx0, n_tokens, worst_case);
        cb(cur, "embd_enc", -1);

        return cur;
    }

    // TODO: tmp
    struct ggml_tensor * build_inp_kq_mask_cross() {
        ggml_tensor * cur = lgf->build_inp_kq_mask_cross(ctx0, n_tokens, worst_case);
        cb(cur, "KQ_mask_cross", -1);

        return cur;
    }

    struct ggml_tensor * build_norm(
             struct ggml_tensor * cur,
             struct ggml_tensor * mw,
             struct ggml_tensor * mb,
                  llm_norm_type   type,
                            int   il) {
        switch (type) {
            case LLM_NORM:       cur = ggml_norm      (ctx0, cur, hparams.f_norm_eps);     break;
            case LLM_NORM_RMS:   cur = ggml_rms_norm  (ctx0, cur, hparams.f_norm_rms_eps); break;
            case LLM_NORM_GROUP:
                {
                    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                    cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
                } break;
        }

        if (mw || mb) {
            cb(cur, "norm", il);
        }

        if (mw) {
            cur = ggml_mul(ctx0, cur, mw);
            if (mb) {
                cb(cur, "norm_w", il);
            }
        }

        if (mb) {
            cur = ggml_add(ctx0, cur, mb);
        }

        return cur;
    }

    struct ggml_tensor * build_ffn(
             struct ggml_tensor * cur,
             struct ggml_tensor * up,
             struct ggml_tensor * up_b,
             struct ggml_tensor * up_s,
             struct ggml_tensor * gate,
             struct ggml_tensor * gate_b,
             struct ggml_tensor * gate_s,
             struct ggml_tensor * down,
             struct ggml_tensor * down_b,
             struct ggml_tensor * down_s,
             struct ggml_tensor * act_scales,
                llm_ffn_op_type   type_op,
              llm_ffn_gate_type   type_gate,
                            int   il) {
        struct ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
        cb(tmp, "ffn_up", il);

        if (up_b) {
            tmp = ggml_add(ctx0, tmp, up_b);
            cb(tmp, "ffn_up_b", il);
        }

        if (up_s) {
            tmp = ggml_mul(ctx0, tmp, up_s);
            cb(tmp, "ffn_up_s", il);
        }

        if (gate) {
            switch (type_gate) {
                case LLM_FFN_SEQ:
                    {
                        cur = build_lora_mm(gate, tmp);
                        cb(cur, "ffn_gate", il);
                    } break;
                case LLM_FFN_PAR:
                    {
                        cur = build_lora_mm(gate, cur);
                        cb(cur, "ffn_gate", il);
                    } break;
            }

            if (gate_b) {
                cur = ggml_add(ctx0, cur, gate_b);
                cb(cur, "ffn_gate_b", il);
            }

            if (gate_s) {
                cur = ggml_mul(ctx0, cur, gate_s);
                cb(cur, "ffn_gate_s", il);
            }

        } else {
            cur = tmp;
        }

        switch (type_op) {
            case LLM_FFN_SILU:
                {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_silu", il);
                } break;
            case LLM_FFN_GELU:
                {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_gelu", il);
                    if (act_scales != NULL) {
                        cur = ggml_div(ctx0, cur, act_scales);
                        cb(cur, "ffn_act", il);
                    }
                } break;
            case LLM_FFN_RELU:
                {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_relu", il);
                } break;
            case LLM_FFN_RELU_SQR:
                {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_relu", il);

                    cur = ggml_sqr(ctx0, cur);
                    cb(cur, "ffn_sqr(relu)", il);
                } break;
            case LLM_FFN_SWIGLU:
                {
                    // Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
                    int64_t split_point = cur->ne[0] / 2;
                    struct ggml_tensor * x0 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], 0));
                    struct ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, split_point, cur->ne[1], cur->nb[1], split_point * ggml_element_size(cur)));

                    x0 = ggml_silu(ctx0, x0);
                    cb(cur, "ffn_silu", il);

                    cur = ggml_mul(ctx0, x0, x1);
                    cb(cur, "ffn_mul", il);
                } break;
        }

        if (type_gate == LLM_FFN_PAR) {
            cur = ggml_mul(ctx0, cur, tmp);
            cb(cur, "ffn_gate_par", il);
        }

        if (down) {
            cur = build_lora_mm(down, cur);
        }

        if (down_b) {
            cb(cur, "ffn_down", il);
        }

        if (down_b) {
            cur = ggml_add(ctx0, cur, down_b);
        }

        if (down_s) {
            cur = ggml_mul(ctx0, cur, down_s);
            cb(cur, "ffn_down_s", il);
        }

        return cur;
    }

    struct ggml_tensor * build_moe_ffn(
             struct ggml_tensor * cur,
             struct ggml_tensor * gate_inp,
             struct ggml_tensor * up_exps,
             struct ggml_tensor * gate_exps,
             struct ggml_tensor * down_exps,
             struct ggml_tensor * exp_probs_b,
                        int64_t   n_expert,
                        int64_t   n_expert_used,
                llm_ffn_op_type   type_op,
                           bool   norm_w,
                           bool   scale_w,
                          float   w_scale,
  llama_expert_gating_func_type   gating_op,
                            int   il) {
        int64_t n_embd = cur->ne[0];
        int64_t n_tokens = cur->ne[1];

        ggml_tensor * logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * probs = nullptr;
        switch (gating_op) {
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
                {
                    probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
                } break;
            case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
                {
                    probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
                } break;
            default:
                GGML_ABORT("fatal error");
        }
        cb(probs, "ffn_moe_probs", il);

        // add experts selection bias - introduced in DeepSeek V3
        // leave probs unbiased as it's later used to get expert weights
        ggml_tensor * selection_probs = probs;
        if (exp_probs_b != nullptr) {
            selection_probs = ggml_add(ctx0, probs, exp_probs_b);
            cb(selection_probs, "ffn_moe_probs_biased", il);
        }

        // select experts
        ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
        cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights", il);

        if (norm_w) {
            weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

            ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
            cb(weights_sum, "ffn_moe_weights_sum", il);

            weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
            cb(weights, "ffn_moe_weights_norm", il);

            weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        }
        if (scale_w) {
            weights = ggml_scale(ctx0, weights, w_scale);
            cb(weights, "ffn_moe_weights_scaled", il);
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        ggml_tensor * gate = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(gate, "ffn_moe_gate", il);

        switch (type_op) {
            case LLM_FFN_SILU:
                {
                    gate = ggml_silu(ctx0, gate);
                    cb(gate, "ffn_moe_silu", il);
                } break;
            case LLM_FFN_GELU:
                {
                    gate = ggml_gelu(ctx0, gate);
                    cb(gate, "ffn_moe_gelu", il);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        ggml_tensor * par = ggml_mul(ctx0, up, gate); // [n_ff, n_expert_used, n_tokens]
        cb(par, "ffn_moe_gate_par", il);

        ggml_tensor * experts = build_lora_mm_id(down_exps, par, selected_experts); // [n_embd, n_expert_used, n_tokens]
        cb(experts, "ffn_moe_down", il);

        experts = ggml_mul(ctx0, experts, weights);

        // aggregate experts
        ggml_tensor * moe_out = nullptr;
        for (int i = 0; i < n_expert_used; ++i) {
            ggml_tensor * cur_expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                    experts->nb[2], i*experts->nb[1]);

            if (i == 0) {
                moe_out = cur_expert;
            } else {
                moe_out = ggml_add(ctx0, moe_out, cur_expert);
            }
        }

        if (n_expert_used == 1) {
            // avoid returning a non-contiguous tensor
            moe_out = ggml_cont(ctx0, moe_out);
        }

        return moe_out;
    }

    struct ggml_tensor * build_attn(
             struct ggml_cgraph * gf,
             struct ggml_tensor * wo,
             struct ggml_tensor * wo_b,
             struct ggml_tensor * q_cur,
             struct ggml_tensor * k_cur,
             struct ggml_tensor * v_cur,
                        int32_t   n_tokens,
                        float     kq_scale,
                        int       il) {
        // these nodes are added to the graph together so that they are not reordered
        // by doing so, the number of splits in the graph is reduced
        ggml_build_forward_expand(gf, q_cur);
        ggml_build_forward_expand(gf, k_cur);
        ggml_build_forward_expand(gf, v_cur);

        ggml_tensor * cur = lgf->build_attn(ctx0, gf, wo, wo_b, q_cur, k_cur, v_cur, n_tokens, kq_scale, il, worst_case);
        cb(cur, "kqv_out", il);

        return cur;
    }

    struct ggml_tensor * build_rwkv_channel_mix(
        const struct llama_layer * layer,
        struct ggml_tensor * cur,
        struct ggml_tensor * x_prev,
        const llm_arch arch) {
        struct ggml_tensor * sx = ggml_sub(ctx0, x_prev, cur);
        switch (arch) {
            case LLM_ARCH_RWKV6:
            {
                struct ggml_tensor * xk = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
                struct ggml_tensor * xr = ggml_add(ctx0, ggml_mul(ctx0, sx, layer->channel_mix_lerp_r), cur);

                struct ggml_tensor * r = ggml_sigmoid(ctx0, build_lora_mm(layer->channel_mix_receptance, xr));
                struct ggml_tensor * k = ggml_sqr(
                        ctx0,
                        ggml_relu(
                            ctx0,
                            build_lora_mm(layer->channel_mix_key, xk)
                            )
                    );
                cur = ggml_mul(ctx0, r, build_lora_mm(layer->channel_mix_value, k));
            } break;
            default:
                GGML_ABORT("fatal error");
        }

        return cur;
    }

    void append_pooling(struct ggml_cgraph * gf) {
        struct ggml_tensor * inp = res.t_embd;

        //// find result_norm tensor for input
        //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
        //    inp = ggml_graph_node(gf, i);
        //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
        //        break;
        //    }

        //    inp = nullptr;
        //}

        GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

        struct ggml_tensor * cur;

        switch (pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    cur = inp;
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
                {
                    struct ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } break;
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    struct ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    struct ggml_tensor * inp_cls = build_inp_cls();
                    inp = ggml_get_rows(ctx0, inp, inp_cls);

                    // classification head
                    // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                    GGML_ASSERT(model.cls       != nullptr);
                    GGML_ASSERT(model.cls_b     != nullptr);

                    cur = ggml_add (ctx0, ggml_mul_mat(ctx0, model.cls, inp), model.cls_b);
                    cur = ggml_tanh(ctx0, cur);

                    // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                    // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                    if (model.cls_out) {
                        GGML_ASSERT(model.cls_out_b != nullptr);

                        cur = ggml_add (ctx0, ggml_mul_mat(ctx0, model.cls_out, cur), model.cls_out_b);
                    }
                } break;
            default:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }

        cb(cur, "result_embd_pooled", -1);
        res.t_embd_pooled = cur;

        ggml_build_forward_expand(gf, cur);
    }

    //struct ggml_tensor * build_pos_bucket(bool causal) {
    //    if (causal) {
    //        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv,     n_tokens);
    //    } else {
    //        lctx.inp_pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    //    }

    //    ggml_set_input(lctx.inp_pos_bucket);
    //    cb(lctx.inp_pos_bucket, "pos_bucket", -1);

    //    return lctx.inp_pos_bucket;
    //}

    //struct ggml_tensor * build_pos_bias(struct ggml_tensor * pos_bucket, struct ggml_tensor * attn_rel_b) {
    //    struct ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    //    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    //    struct ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);
    //    cb(pos_bias, "pos_bias", -1);

    //    pos_bias = ggml_view_3d(ctx0, pos_bias, pos_bias->ne[0], lctx.inp_pos_bucket->ne[0], lctx.inp_pos_bucket->ne[1], ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * lctx.inp_pos_bucket->ne[0],  0);
    //    cb(pos_bias, "pos_bias", -1);

    //    pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
    //    cb(pos_bias, "pos_bias", -1);

    //    pos_bias = ggml_cont(ctx0, pos_bias);
    //    cb(pos_bias, "pos_bias", -1);

    //    return pos_bias;
    //}

    void build_llama(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, kq_scale, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {

                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                cb(cur, "ffn_moe_out", il);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // For Granite architecture
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_deci(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head    = hparams.n_head(il);

            if (n_head == 0) {
                // attention-free layer of Llama-3_1-Nemotron-51B
                cur = inpL;
            } else {
                // norm
                cur = build_norm(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            if (n_head > 0 && n_head_kv == 0) {
                // "linear attention" of Llama-3_1-Nemotron-51B
                cur = build_lora_mm(model.layers[il].wo, cur);
                cb(cur, "wo", il);
            } else if (n_head > 0) {
                // self-attention
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, kq_scale, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            // modified to support attention-free layer of Llama-3_1-Nemotron-51B
            struct ggml_tensor * ffn_inp = cur;
            if (n_head > 0) {
                ffn_inp = ggml_add(ctx0, cur, inpSA);
                cb(ffn_inp, "ffn_inp", il);
            }

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // For Granite architecture
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_baichuan(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = model.type == LLM_TYPE_7B ? build_inp_pos() : nullptr;

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                switch (model.type) {
                    case LLM_TYPE_7B:
                        Qcur = ggml_rope_ext(
                            ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                        );
                        Kcur = ggml_rope_ext(
                            ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow
                        );
                        break;
                    case LLM_TYPE_13B:
                        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd/n_head, n_head, n_tokens);
                        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd/n_head, n_head, n_tokens);
                        break;
                    default:
                        GGML_ABORT("fatal error");
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_xverse(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_falcon(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * attn_norm;

            attn_norm = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                if (model.layers[il].attn_norm_2) {
                    // Falcon-40B
                    cur = build_norm(inpL,
                            model.layers[il].attn_norm_2,
                            model.layers[il].attn_norm_2_b,
                            LLM_NORM, il);
                    cb(cur, "attn_norm_2", il);
                } else {
                    cur = attn_norm;
                }

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur       = ggml_get_rows(ctx0,       cur, inp_out_ids);
                inpL      = ggml_get_rows(ctx0,      inpL, inp_out_ids);
                attn_norm = ggml_get_rows(ctx0, attn_norm, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                cur = build_ffn(attn_norm, // !! use the attn norm, not the result
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = ggml_add(ctx0, cur, inpL);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_grok(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // multiply by embedding_multiplier_scale of 78.38367176906169
        inpL = ggml_scale(ctx0, inpL, 78.38367176906169f);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);


            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Grok
            // if attn_out_norm is present then apply it before adding the input
            if (model.layers[il].attn_out_norm) {
                cur = build_norm(cur,
                        model.layers[il].attn_out_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_out_norm", il);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            // Grok
            // if layer_out_norm is present then apply it before adding the input
            // Idea: maybe ffn_out_norm is a better name
            if (model.layers[il].layer_out_norm) {
                cur = build_norm(cur,
                        model.layers[il].layer_out_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "layer_out_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // Grok
        // multiply logits by output_multiplier_scale of 0.5773502691896257

        cur = ggml_scale(ctx0, cur, 0.5773502691896257f);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_dbrx(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(cur, "wqkv_clamped", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].attn_out_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_out_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_starcoder(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        struct ggml_tensor * pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_refact(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                cb(Qcur, "Qcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_bert(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;
        struct ggml_tensor * inp_pos = nullptr;

        if (model.arch != LLM_ARCH_JINA_BERT_V2) {
            inp_pos = build_inp_pos();
        }

        // construct input embeddings (token, type, position)
        inpL = build_inp_embd(model.tok_embd);

        // token types are hardcoded to zero ("Sentence A")
        struct ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
        inpL = ggml_add(ctx0, inpL, type_row0);
        if (model.arch == LLM_ARCH_BERT) {
            inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
        }
        cb(inpL, "inp_embd", -1);

        // embed layer norm
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
        cb(inpL, "inp_norm", -1);

        lgf->build_attn_inp(ctx0, n_tokens, false, false, worst_case);

        // iterate layers
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * cur = inpL;

            struct ggml_tensor * Qcur;
            struct ggml_tensor * Kcur;
            struct ggml_tensor * Vcur;

            // self-attention
            if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_JINA_BERT_V2) {
                Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                }

                Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);

                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                }

                Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            } else {
                // compute Q and K and RoPE them
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            if (il == n_layer - 1 && pooling_type == LLAMA_POOLING_TYPE_NONE) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention layer norm
            cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);

            if (model.layers[il].attn_norm_2 != nullptr) {
                cur = ggml_add(ctx0, cur, inpL); // re-add the layer input
                cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
            }

            struct ggml_tensor * ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.arch == LLM_ARCH_BERT) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
            } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL,                        NULL,
                        model.layers[il].ffn_gate, NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
            }
            cb(cur, "ffn_out", il);

            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, cur, ffn_inp);

            // output layer norm
            cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cb(cur, "result_embd", -1);
        res.t_embd = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_bloom(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        inpL = build_norm(inpL,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, -1);
        cb(inpL, "inp_norm", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_mpt(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * pos;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        if (model.pos_embd) {
            // inp_pos - contains the positions
            struct ggml_tensor * inp_pos = build_inp_pos();
            pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
            cb(pos, "pos_embd", -1);

            inpL = ggml_add(ctx0, inpL, pos);
            cb(inpL, "inpL", -1);
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * attn_norm;

            attn_norm = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                cur = attn_norm;

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv){
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                if (hparams.f_clamp_kqv > 0.0f) {
                    cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(cur, "wqkv_clamped", il);
                }

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // Q/K Layernorm
                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            model.layers[il].attn_q_norm_b,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            model.layers[il].attn_k_norm_b,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);

                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                    cur = build_attn(gf,
                            model.layers[il].wo, model.layers[il].bo,
                            Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
                } else {
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                    cur = build_attn(gf,
                            model.layers[il].wo, model.layers[il].bo,
                            Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
                }
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed forward
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        model.layers[il].ffn_act,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_stablelm(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {


            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor * inpSA = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                cb(Qcur, "Qcur", il);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                cb(Kcur, "Kcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = build_norm(Qcur,
                            model.layers[il].attn_q_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }


                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpL  = ggml_get_rows(ctx0,  inpL, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                if (model.layers[il].ffn_norm) {
                    cur = build_norm(ffn_inp,
                            model.layers[il].ffn_norm,
                            model.layers[il].ffn_norm_b,
                            LLM_NORM, il);
                    cb(cur, "ffn_norm", il);
                } else {
                    // parallel residual
                    cur = inpSA;
                }
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_qwen(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_qwen2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_qwen2vl(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        int sections[4];
        std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_multi(
                    ctx0,
                    ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_multi(
                    ctx0,
                    ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_qwen2moe(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * moe_out =
                    build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
            cb(cur, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * cur_gate_inp = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
                cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

                // sigmoid
                ggml_tensor * cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
                cb(cur_gate, "ffn_shexp_gate", il);

                ggml_tensor * cur_ffn = build_ffn(cur,
                        model.layers[il].ffn_up_shexp,   NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur_ffn, "ffn_shexp", il);

                ggml_tensor * ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
                cb(ffn_shexp_out, "ffn_shexp_out", il);

                moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
                cb(moe_out, "ffn_out", il);

                cur = moe_out;
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_phi2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * attn_norm_output;
        struct ggml_tensor * ffn_output;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(attn_norm_output, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                // with phi2, we scale the Q to avoid precision issues
                // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
                Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur              = ggml_get_rows(ctx0,              cur, inp_out_ids);
                inpL             = ggml_get_rows(ctx0,             inpL, inp_out_ids);
                attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
            }

            // FF
            {
                ffn_output = build_ffn(attn_norm_output,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(ffn_output, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_output);
            cur = ggml_add(ctx0, cur, inpL);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output_no_bias", -1);

        cur = ggml_add(ctx0, cur, model.output_b);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_phi3(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        lgf->build_attn_inp(ctx0, n_tokens, true, true, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            auto * residual = inpL;

            // self-attention
            {
                // rope freq factors for 128k context
                struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);

                struct ggml_tensor* attn_norm_output = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM_RMS, il);
                cb(attn_norm_output, "attn_norm", il);

                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = build_lora_mm(model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0 * sizeof(float) * (n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa)));
                } else {
                    Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, residual);
            residual = cur;

            cur = build_norm(cur,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                cur = build_moe_ffn(cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                cb(cur, "ffn_moe_out", il);
            }

            cur = ggml_add(ctx0, residual, cur);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        if (model.output_b != nullptr) {
            cb(cur, "result_output_no_bias", -1);
            cur = ggml_add(ctx0, cur, model.output_b);
        }

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_plamo(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor * attention_norm = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens), inp_pos, nullptr,
                        n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens), inp_pos, nullptr,
                        n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }
            struct ggml_tensor * sa_out = cur;

            cur = attention_norm;

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur    = ggml_get_rows(ctx0,    cur, inp_out_ids);
                sa_out = ggml_get_rows(ctx0, sa_out, inp_out_ids);
                inpL   = ggml_get_rows(ctx0,   inpL, inp_out_ids);
            }

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cur = ggml_add(ctx0, cur, inpL);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_gpt2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * pos;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_codeshell(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * tmpq = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * tmpk = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(tmpq, "tmpq", il);
                cb(tmpk, "tmpk", il);
                cb(Vcur, "Vcur", il);

                struct ggml_tensor * Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_orion(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                // if (model.layers[il].bq) {
                //     Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                //     cb(Qcur, "Qcur", il);
                // }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                // if (model.layers[il].bk) {
                //     Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                //     cb(Kcur, "Kcur", il);
                // }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                // if (model.layers[il].bv) {
                //     Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                //     cb(Vcur, "Vcur", il);
                // }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_internlm2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_minicpm3(ggml_cgraph * gf) {
        //TODO: if the model varies, these parameters need to be read from the model
        const int64_t n_embd_base = 256;
        const float scale_embd  = 12.0f;
        const float scale_depth = 1.4f;
        const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // scale the input embeddings
        inpL = ggml_scale(ctx0, inpL, scale_embd);
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                struct ggml_tensor * q = NULL;
                // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q,
                        model.layers[il].attn_q_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(q, "q", il);

                // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                struct ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                        kv_pe_compresseed->nb[1],
                        0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_pe_compresseed->nb[1],
                        kv_pe_compresseed->nb[1],
                        ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                // TODO: the CUDA backend used to not support non-cont. (RMS) norm, investigate removing ggml_cont
                kv_compressed = ggml_cont(ctx0, kv_compressed);
                kv_compressed = build_norm(kv_compressed,
                        model.layers[il].attn_kv_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                    ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                    0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(ctx0, q_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(ctx0, k_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        q_states, k_states, v_states, n_tokens, kq_scale, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // scale_res - scale the hidden states for residual connection
            const float scale_res = scale_depth/sqrtf(float(n_layer));
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled", il);

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            // scale the hidden states for residual connection
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled_ffn", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head scaling
        const float scale_lmhead = float(n_embd_base)/float(n_embd);
        cur = ggml_scale(ctx0, cur, scale_lmhead);
        cb(cur, "lmhead_scaling", -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_gemma(ggml_cgraph * gf) {
        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_gemma2(ggml_cgraph * gf) {
        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, true, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head,    n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                // ref: https://github.com/google/gemma_pytorch/commit/03e657582d17cb5a8617ebf333c1c16f3694670e
                switch (model.type) {
                    case LLM_TYPE_2B:
                    case LLM_TYPE_9B:  Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));   break;
                    case LLM_TYPE_27B: Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd / n_head))); break;
                    default: GGML_ABORT("fatal error");
                };
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_kv, n_tokens), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f, il);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = build_norm(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, sa_out);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        // final logit soft-capping
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    // TODO: move up next to build_starcoder
    void build_starcoder2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_mamba(ggml_cgraph * gf) {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        struct ggml_tensor * state_copy = lgf->build_inp_s_copy(ctx0, worst_case);
        struct ggml_tensor * state_mask = lgf->build_inp_s_mask(ctx0, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            //cur = build_mamba_layer(gf, cur, state_copy, state_mask, il);
            cur = lgf->build_mamba_layer(ctx0, gf, cur, state_copy, state_mask, ubatch, il, worst_case);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // residual
            cur = ggml_add(ctx0, cur, inpL);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        // final rmsnorm
        cur = build_norm(inpL,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_command_r(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        const float f_logit_scale = hparams.f_logit_scale;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);
            struct ggml_tensor * ffn_inp = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                                ggml_element_size(Qcur) * n_embd_head,
                                ggml_element_size(Qcur) * n_embd_head * n_head,
                                0);
                    cb(Qcur, "Qcur", il);
                    Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                                ggml_element_size(Kcur) * n_embd_head,
                                ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                                0);
                    cb(Kcur, "Kcur", il);

                    Qcur = build_norm(Qcur,
                                model.layers[il].attn_q_norm,
                                NULL,
                                LLM_NORM, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = build_norm(Kcur,
                            model.layers[il].attn_k_norm,
                            NULL,
                            LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur     = ggml_get_rows(ctx0,     cur, inp_out_ids);
                inpL    = ggml_get_rows(ctx0,    inpL, inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            struct ggml_tensor * attn_out = cur;

            // feed-forward network
            {
                cur = build_ffn(ffn_inp,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_cohere2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        const float f_logit_scale = hparams.f_logit_scale;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, true, worst_case);

        // sliding window switch pattern
        const int32_t sliding_window_pattern = 4;

        for (int il = 0; il < n_layer; ++il) {
            // three layers sliding window attention (window size 4096) and ROPE
            // fourth layer uses global attention without positional embeddings
            const bool is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);

            // norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM, il);
            cb(cur, "attn_norm", il);
            struct ggml_tensor * ffn_inp = cur;

            // self-attention
            {
                // rope freq factors for 128k context
                struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                if (is_sliding) {
                    Qcur = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                                        beta_fast, beta_slow);
                    cb(Qcur, "Qcur", il);

                    Kcur = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos,
                                        rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                        attn_factor, beta_fast, beta_slow);
                    cb(Kcur, "Kcur", il);
                } else {
                    // For non-sliding layers, just reshape without applying RoPE
                    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                    cb(Qcur, "Qcur", il);

                    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                    cb(Kcur, "Kcur", il);
                }

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur                              = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL                             = ggml_get_rows(ctx0, inpL, inp_out_ids);
                ffn_inp                          = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            struct ggml_tensor * attn_out = cur;

            // feed-forward network
            {
                cur = build_ffn(ffn_inp, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate,
                                    NULL, NULL, model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR,
                                    il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    // ref: https://allenai.org/olmo
    // based on the original build_llama() function, changes:
    //   * non-parametric layer norm
    //   * clamp qkv
    //   * removed bias
    //   * removed MoE
    void build_olmo(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    NULL, NULL,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    NULL, NULL,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                NULL, NULL,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_olmo2(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = inpL;

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            cur = build_norm(cur,
                    model.layers[il].attn_post_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_ffn(ffn_inp,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = build_norm(cur,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    // based on the build_qwen2moe() function, changes:
    //   * removed shared experts
    //   * removed bias
    //   * added q, k norm
    void build_olmoe(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_openelm(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head    = hparams.n_head(il);
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head_qkv = 2*n_head_kv + n_head;

            cur = inpL;
            struct ggml_tensor * residual = cur;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_reshape_3d(ctx0, cur, n_embd_head_k, n_head_qkv, n_tokens);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 0));
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*n_head));
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, cur->nb[1], cur->nb[2], cur->nb[1]*(n_head+n_head_kv)));
                cb(Vcur, "Vcur", il);

                Qcur = build_norm(Qcur,
                        model.layers[il].attn_q_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Qcur, "Qcur", il);

                Kcur = build_norm(Kcur,
                        model.layers[il].attn_k_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                Vcur = ggml_reshape_2d(ctx0, Vcur, n_embd_head * n_head_kv, n_tokens);
                cb(Qcur, "Vcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, residual, cur);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_gptneox(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // ffn
            if (hparams.use_par_res) {
                // attention and ffn are computed in parallel
                // x = x + attn(ln1(x)) + ffn(ln2(x))

                struct ggml_tensor * attn_out = cur;

                cur = build_norm(inpL,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, inpL);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, attn_out);

                cur = lgf->build_cvec(ctx0, cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            } else {
                // attention and ffn are computed sequentially
                // x = x + attn(ln1(x))
                // x = x + ffn(ln2(x))

                struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
                cb(ffn_inp, "ffn_inp", il);

                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_inp);

                cur = lgf->build_cvec(ctx0, cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            }
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_arctic(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            struct ggml_tensor * ffn_out = ggml_add(ctx0, cur, ffn_inp);
            cb(ffn_out, "ffn_out", il);

            // MoE
            cur = build_norm(inpSA,
                    model.layers[il].ffn_norm_exps, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm_exps", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_out);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_deepseek(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, kq_scale, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }


            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out =
                        build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            nullptr,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, false,
                            false, hparams.expert_weights_scale,
                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                            il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_deepseek2(ggml_cgraph * gf) {
        bool is_lite = (hparams.n_layer == 27);

        // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
        // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
        const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        const float kq_scale = 1.0f*mscale*mscale/sqrtf(float(hparams.n_embd_head_k));
        const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                struct ggml_tensor * q = NULL;
                if (!is_lite) {
                    // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                    cb(q, "q", il);

                    q = build_norm(q,
                            model.layers[il].attn_q_a_norm, NULL,
                            LLM_NORM_RMS, il);
                    cb(q, "q", il);

                    // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    cb(q, "q", il);
                } else {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                    cb(q, "q", il);
                }

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                        ggml_row_size(q->type, hparams.n_embd_head_k),
                        ggml_row_size(q->type, hparams.n_embd_head_k * n_head),
                        ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                struct ggml_tensor * kv_compressed = ggml_view_2d(ctx0, kv_pe_compresseed, kv_lora_rank, n_tokens,
                        kv_pe_compresseed->nb[1],
                        0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_pe_compresseed, n_embd_head_qk_rope, 1, n_tokens,
                        kv_pe_compresseed->nb[1],
                        kv_pe_compresseed->nb[1],
                        ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                // TODO: the CUDA backend used to not support non-cont. (RMS) norm, investigate removing ggml_cont
                kv_compressed = ggml_cont(ctx0, kv_compressed);
                kv_compressed = build_norm(kv_compressed,
                        model.layers[il].attn_kv_a_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                struct ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                        ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                        ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                struct ggml_tensor * v_states = ggml_view_3d(ctx0, kv, hparams.n_embd_head_v, n_head, n_tokens,
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)*n_head),
                        ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view_2d(ctx0, v_states, hparams.n_embd_head_v * n_head, n_tokens,
                    ggml_row_size(kv->type, hparams.n_embd_head_v * n_head),
                    0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(ctx0, q_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(ctx0, k_pe); // TODO: the CUDA backend used to not support non-cont. RoPE, investigate removing this
                k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                struct ggml_tensor * q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor * k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        q_states, k_states, v_states, n_tokens, kq_scale, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            } else {
                // MoE branch
                ggml_tensor * moe_out =
                        build_moe_ffn(cur,
                            model.layers[il].ffn_gate_inp,
                            model.layers[il].ffn_up_exps,
                            model.layers[il].ffn_gate_exps,
                            model.layers[il].ffn_down_exps,
                            model.layers[il].ffn_exp_probs_b,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, hparams.expert_weights_norm,
                            true, hparams.expert_weights_scale,
                            (enum llama_expert_gating_func_type) hparams.expert_gating_func,
                            il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor * ffn_shexp = build_ffn(cur,
                            model.layers[il].ffn_up_shexp,   NULL, NULL,
                            model.layers[il].ffn_gate_shexp, NULL, NULL,
                            model.layers[il].ffn_down_shexp, NULL, NULL,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_bitnet(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                if (model.layers[il].wq_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, model.layers[il].wq_scale);
                }
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                // B1.K
                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                if (model.layers[il].wk_scale) {
                    Kcur = ggml_mul(ctx0, Kcur, model.layers[il].wk_scale);
                }
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                // B1.V
                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                if (model.layers[il].wv_scale) {
                    Vcur = ggml_mul(ctx0, Vcur, model.layers[il].wv_scale);
                }
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        NULL, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);

                cur = build_norm(cur,
                        model.layers[il].attn_sub_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "attn_sub_norm", il);

                cur = build_lora_mm(model.layers[il].wo, cur);
                if (model.layers[il].wo_scale) {
                    cur = ggml_mul(ctx0, cur, model.layers[il].wo_scale);
                }
                if (model.layers[il].bo) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bo);
                }
                cb(cur, "attn_o_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_scale,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_scale,
                    NULL,                      NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_sub_out", il);

            cur = build_norm(cur,
                    model.layers[il].ffn_sub_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_sub_norm", il);

            cur = build_lora_mm(model.layers[il].ffn_down, cur);
            if (model.layers[il].ffn_down_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_down_scale);
            }
            cb(cur, "ffn_down", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        // FIXME: do not use model.tok_embd directly, duplicate as model.output
        cur = build_lora_mm(model.tok_embd, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    //void build_t5_enc(ggml_cgraph * gf) {
    //    const int64_t n_embd_head = hparams.n_embd_head_v;
    //    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    //    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    //    struct ggml_tensor * cur;
    //    struct ggml_tensor * inpL;

    //    inpL = build_inp_embd(model.tok_embd);

    //    GGML_ASSERT(lctx.is_encoding);
    //    struct ggml_tensor * pos_bucket_enc = build_pos_bucket(false);

    //    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    //    struct ggml_tensor * KQ_mask_enc = build_inp_kq_mask(false);

    //    for (int il = 0; il < n_layer; ++il) {
    //        struct ggml_tensor * inpSA = inpL;

    //        // norm
    //        cur = build_norm(inpL,
    //                model.layers[il].attn_norm_enc, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "attn_norm", il);

    //        // self-attention
    //        {
    //            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_enc, cur);
    //            cb(Qcur, "Qcur", il);

    //            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_enc, cur);
    //            cb(Kcur, "Kcur", il);

    //            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_enc, cur);
    //            cb(Vcur, "Vcur", il);

    //            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    //            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

    //            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
    //            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

    //            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    //            cb(kq, "kq", il);

    //            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
    //            struct ggml_tensor * pos_bias = build_pos_bias(pos_bucket_enc, attn_rel_b);
    //            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
    //            cb(kq_b, "kq_b", il);

    //            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_enc, 1.0f, hparams.f_max_alibi_bias);
    //            cb(kq, "kq_soft_max_ext", il);

    //            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
    //            cb(v, "v", il);

    //            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
    //            cb(kqv, "kqv", il);

    //            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    //            cb(kqv_merged, "kqv_merged", il);

    //            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
    //            cb(cur, "kqv_merged_cont", il);

    //            ggml_build_forward_expand(gf, cur);

    //            cur = build_lora_mm(model.layers[il].wo_enc, cur);
    //            cb(cur, "kqv_out", il);
    //        }

    //        if (il == n_layer - 1) {
    //            // skip computing output for unused tokens
    //            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
    //            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
    //            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    //        }

    //        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    //        cb(ffn_inp, "ffn_inp", il);

    //        // feed-forward network
    //        {
    //            cur = build_norm(ffn_inp,
    //                    model.layers[il].ffn_norm_enc, NULL,
    //                    LLM_NORM_RMS, il);
    //            cb(cur, "ffn_norm", il);

    //            // T5 uses relu, flan-T5 uses gelu-gated
    //            cur = build_ffn(cur,
    //                    model.layers[il].ffn_up_enc,   NULL, NULL,
    //                    model.layers[il].ffn_gate_enc, NULL, NULL,
    //                    model.layers[il].ffn_down_enc, NULL, NULL,
    //                    NULL,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR  : LLM_FFN_SEQ,
    //                    il);
    //            cb(cur, "ffn_out", il);
    //        }

    //        cur = ggml_add(ctx0, cur, ffn_inp);
    //        cb(cur, "ffn_out", il);

    //        ggml_tensor * layer_dir = cvec.tensor_for(il);
    //        if (layer_dir != nullptr) {
    //            cur = ggml_add(ctx0, cur, layer_dir);
    //        }
    //        cb(cur, "l_out", il);

    //        // input for next layer
    //        inpL = cur;
    //    }

    //    cur = inpL;
    //    cb(cur, "result_embd", -1);

    //    cur = build_norm(cur,
    //            model.output_norm_enc, NULL,
    //            LLM_NORM_RMS, -1);
    //
    //    cb(cur, "result_norm", -1);
    //    res.t_embd = cur;

    //    ggml_build_forward_expand(gf, cur);
    //}

    //void build_t5_dec(ggml_cgraph * gf) {
    //    const int64_t n_embd_head = hparams.n_embd_head_v;
    //    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    //    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    //    struct ggml_tensor * cur;
    //    struct ggml_tensor * inpL;

    //    inpL = build_inp_embd(model.tok_embd);

    //    GGML_ASSERT(!lctx.is_encoding);
    //    GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first");

    //    struct ggml_tensor * embd_enc       = build_inp_embd_enc();
    //    struct ggml_tensor * pos_bucket_dec = build_pos_bucket(true);

    //    struct ggml_tensor * KQ_mask_dec   = build_inp_kq_mask();
    //    struct ggml_tensor * KQ_mask_cross = build_inp_kq_mask_cross();

    //    for (int il = 0; il < n_layer; ++il) {
    //        struct ggml_tensor * inpSA = inpL;

    //        // norm
    //        cur = build_norm(inpL,
    //                model.layers[il].attn_norm, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "attn_norm", il);

    //        // self-attention
    //        {
    //            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    //            cb(Qcur, "Qcur", il);

    //            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    //            cb(Kcur, "Kcur", il);

    //            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    //            cb(Vcur, "Vcur", il);

    //            build_kv_store(gf, Kcur, Vcur, il);

    //            struct ggml_tensor * k =
    //                ggml_view_3d(ctx0, kv_self.k_l[il],
    //                        n_embd_head_k, n_kv, n_head_kv,
    //                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
    //                        ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
    //                        0);
    //            cb(k, "k", il);

    //            struct ggml_tensor * v =
    //                ggml_view_3d(ctx0, kv_self.v_l[il],
    //                        n_kv, n_embd_head_v, n_head_kv,
    //                        ggml_element_size(kv_self.v_l[il])*n_ctx,
    //                        ggml_element_size(kv_self.v_l[il])*n_ctx*n_embd_head_v,
    //                        0);
    //            cb(v, "v", il);

    //            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

    //            struct ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

    //            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    //            cb(kq, "kq", il);

    //            struct ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
    //            struct ggml_tensor * pos_bias = build_pos_bias(pos_bucket_dec, attn_rel_b);
    //            struct ggml_tensor * kq_b = ggml_add(ctx0, kq, pos_bias);
    //            cb(kq_b, "kq_b", il);

    //            kq = ggml_soft_max_ext(ctx0, kq_b, KQ_mask_dec, 1.0f, hparams.f_max_alibi_bias);
    //            cb(kq, "kq_soft_max_ext", il);

    //            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
    //            cb(kqv, "kqv", il);

    //            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    //            cb(kqv_merged, "kqv_merged", il);

    //            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
    //            cb(cur, "kqv_merged_cont", il);

    //            ggml_build_forward_expand(gf, cur);

    //            cur = build_lora_mm(model.layers[il].wo, cur);
    //            cb(cur, "kqv_out", il);
    //        }

    //        cur = ggml_add(ctx0, cur, inpSA);
    //        cb(cur, "cross_inp", il);

    //        struct ggml_tensor * inpCA = cur;

    //        // norm
    //        cur = build_norm(cur,
    //                model.layers[il].attn_norm_cross, NULL,
    //                LLM_NORM_RMS, il);
    //        cb(cur, "attn_norm_cross", il);

    //        // cross-attention
    //        {
    //            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
    //            cb(Qcur, "Qcur", il);

    //            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
    //            cb(Kcur, "Kcur", il);

    //            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
    //            cb(Vcur, "Vcur", il);

    //            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
    //            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);

    //            struct ggml_tensor * q =                 ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
    //            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

    //            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    //            cb(kq, "kq", il);

    //            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
    //            cb(kq, "kq_soft_max_ext", il);

    //            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
    //            cb(v, "v", il);

    //            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
    //            cb(kqv, "kqv", il);

    //            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    //            cb(kqv_merged, "kqv_merged", il);

    //            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
    //            cb(cur, "kqv_merged_cont", il);

    //            ggml_build_forward_expand(gf, cur);

    //            cur = build_lora_mm(model.layers[il].wo_cross, cur);
    //            cb(cur, "kqv_out", il);
    //        }

    //        if (il == n_layer - 1) {
    //            // skip computing output for unused tokens
    //            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
    //            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
    //            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    //            inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
    //        }

    //        struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpCA);
    //        cb(ffn_inp, "ffn_inp", il);

    //        // feed-forward network
    //        {
    //            cur = build_norm(ffn_inp,
    //                    model.layers[il].ffn_norm, NULL,
    //                    LLM_NORM_RMS, il);
    //            cb(cur, "ffn_norm", il);

    //            // T5 uses relu, flan-T5 uses gelu-gated
    //            cur = build_ffn(cur,
    //                    model.layers[il].ffn_up,   NULL, NULL,
    //                    model.layers[il].ffn_gate, NULL, NULL,
    //                    model.layers[il].ffn_down, NULL, NULL,
    //                    NULL,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
    //                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
    //                    il);
    //            cb(cur, "ffn_out", il);
    //        }

    //        cur = ggml_add(ctx0, cur, ffn_inp);
    //        cb(cur, "ffn_out", il);

    //        ggml_tensor * layer_dir = lctx.cvec.tensor_for(il);
    //        if (layer_dir != nullptr) {
    //            cur = ggml_add(ctx0, cur, layer_dir);
    //        }
    //        cb(cur, "l_out", il);

    //        // input for next layer
    //        inpL = cur;
    //    }

    //    cur = inpL;
    //    cb(cur, "result_embd", -1);

    //    cur = build_norm(cur,
    //            model.output_norm, NULL,
    //            LLM_NORM_RMS, -1);

    //    cb(cur, "result_norm", -1);
    //    res.t_embd = cur;

    //    // lm_head
    //    cur = build_lora_mm(model.output, cur);

    //    cb(cur, "result_output", -1);
    //    res.t_logits = cur;

    //    ggml_build_forward_expand(gf, cur);

    //    return gf;
    //}

    void build_jais(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*cur->nb[0]*(n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*cur->nb[0]*(n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/float(n_embd_head), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = build_norm(inpL,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_chatglm(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor * Qcur = nullptr;
                struct ggml_tensor * Kcur = nullptr;
                struct ggml_tensor * Vcur = nullptr;

                if (model.layers[il].wqkv == nullptr) {
                    Qcur = build_lora_mm(model.layers[il].wq, cur);
                    if (model.layers[il].bq) {
                        Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    }
                    Kcur = build_lora_mm(model.layers[il].wk, cur);
                    if (model.layers[il].bk) {
                        Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    }
                    Vcur = build_lora_mm(model.layers[il].wv, cur);
                    if (model.layers[il].bv) {
                        Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    }
                } else {
                    cur = build_lora_mm(model.layers[il].wqkv, cur);
                    cb(cur, "wqkv", il);
                    if (model.layers[il].bqkv) {
                        cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                        cb(cur, "bqkv", il);
                    }
                    Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                //printf("freq_base: %f freq_scale: %f ext_factor: %f attn_factor: %f\n", freq_base, freq_scale, ext_factor, attn_factor);
                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = build_attn(gf,
                        model.layers[il].wo, NULL,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);

            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm,
                        NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);

                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   NULL, NULL,
                        NULL,                      NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
                cb(cur, "ffn_out", il);

            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = build_norm(inpL,
                model.output_norm,
                NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_nemotron(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        //GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_b,
                    LLM_NORM, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_SEQ, il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, model.output_norm_b,
                LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_exaone(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor * rope_factors = lgf->build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_rwkv6(ggml_cgraph * gf) {
        GGML_ASSERT(hparams.token_shift_count == 2);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);
        inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);

        struct ggml_tensor * state_copy = lgf->build_inp_s_copy(ctx0, worst_case);
        struct ggml_tensor * state_mask = lgf->build_inp_s_mask(ctx0, worst_case);

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];

            struct ggml_tensor * token_shift = lgf->build_rwkv_token_shift_load(
                ctx0, gf, state_copy, state_mask, ubatch, il, worst_case
            );

            struct ggml_tensor * att_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
            struct ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], n_embd * ggml_element_size(token_shift));

            struct ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM, il);
            cb(att_norm, "attn_norm", il);

            struct ggml_tensor * x_prev = ggml_concat(
                ctx0,
                att_shift,
                ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                1
            );

            cur = lgf->build_rwkv6_time_mix(ctx0, gf, att_norm, x_prev, state_copy, state_mask, ubatch, il, worst_case);

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            struct ggml_tensor * ffn_norm = build_norm(ffn_inp, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, il);
            cb(ffn_norm, "ffn_norm", il);

            x_prev = ggml_concat(
                ctx0,
                ffn_shift,
                ggml_view_3d(ctx0, ffn_norm, n_embd, n_seq_tokens - 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], 0),
                1
            );

            cur = build_rwkv_channel_mix(layer, ffn_norm, x_prev, LLM_ARCH_RWKV6);
            cur = ggml_add(ctx0, cur, ffn_inp);

            token_shift = ggml_concat(ctx0,
                ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm)),
                ggml_view_3d(ctx0, ffn_norm, n_embd, 1, n_seqs, ffn_norm->nb[1], ffn_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(ffn_norm)),
                1
            );
            ggml_build_forward_expand(gf, lgf->build_rwkv_token_shift_store(ctx0, token_shift, ubatch, il, worst_case));

            if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
                cur = ggml_scale(ctx0, cur, 0.5F);
            }

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        struct ggml_tensor * inp_out_ids = build_inp_out_ids();

        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);

        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    // ref: https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1/blob/main/modeling_rwkv6qwen2.py
    void build_rwkv6qwen2(ggml_cgraph * gf) {
        GGML_ASSERT(n_embd == hparams.n_embd_k_s());

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        struct ggml_tensor * state_copy = lgf->build_inp_s_copy(ctx0, worst_case);
        struct ggml_tensor * state_mask = lgf->build_inp_s_mask(ctx0, worst_case);

        const auto n_embd = hparams.n_embd;
        const auto n_seq_tokens = ubatch.n_seq_tokens;
        const auto n_seqs = ubatch.n_seqs;

        inpL = build_inp_embd(model.tok_embd);

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];

            struct ggml_tensor * token_shift = lgf->build_rwkv_token_shift_load(
                ctx0, gf, state_copy, state_mask, ubatch, il, worst_case
            );

            struct ggml_tensor * att_norm = build_norm(inpL, layer->attn_norm, layer->attn_norm_b, LLM_NORM_RMS, il);
            cb(att_norm, "attn_norm", il);

            struct ggml_tensor * x_prev = ggml_concat(
                ctx0,
                token_shift,
                ggml_view_3d(ctx0, att_norm, n_embd, n_seq_tokens - 1, n_seqs, att_norm->nb[1], att_norm->nb[2], 0),
                1
            );

            cur = lgf->build_rwkv6_time_mix(ctx0, gf, att_norm, x_prev, state_copy, state_mask, ubatch, il, worst_case);

            token_shift = ggml_view_3d(ctx0, att_norm, n_embd, 1, n_seqs, att_norm->nb[1], att_norm->nb[2], (n_seq_tokens-1)*n_embd*ggml_element_size(att_norm));
            ggml_build_forward_expand(gf, lgf->build_rwkv_token_shift_store(ctx0, token_shift, ubatch, il, worst_case));

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        struct ggml_tensor * inp_out_ids = build_inp_out_ids();
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);

        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        cur = build_lora_mm(model.output, cur);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    // ref: https://github.com/facebookresearch/chameleon
    // based on the original build_llama() function, changes:
    //   * qk-norm
    //   * swin-norm
    //   * removed bias
    //   * removed MoE
    void build_chameleon(ggml_cgraph * gf) {
        const int64_t n_embd_head = hparams.n_embd_head_v;

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();

        lgf->build_attn_inp(ctx0, n_tokens, true, false, worst_case);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            if (hparams.swin_norm) {
                cur = inpL;
            } else {
                cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
                cb(cur, "attn_norm", il);
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens,
                                ggml_element_size(Qcur) * n_embd_head,
                                ggml_element_size(Qcur) * n_embd_head * n_head,
                                0);
                    cb(Qcur, "Qcur", il);

                    Qcur = build_norm(Qcur,
                                model.layers[il].attn_q_norm,
                                model.layers[il].attn_q_norm_b,
                                LLM_NORM, il);
                    cb(Qcur, "Qcur", il);
                }

                if (model.layers[il].attn_k_norm) {
                    Kcur = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens,
                                ggml_element_size(Kcur) * n_embd_head,
                                ggml_element_size(Kcur) * n_embd_head * n_head_kv,
                                0);
                    cb(Kcur, "Kcur", il);

                    Kcur = build_norm(Kcur,
                               model.layers[il].attn_k_norm,
                               model.layers[il].attn_k_norm_b,
                               LLM_NORM, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = build_attn(gf,
                        model.layers[il].wo, nullptr,
                        Qcur, Kcur, Vcur, n_tokens, 1.0f/sqrtf(float(n_embd_head)), il);

                if (hparams.swin_norm) {
                    cur = build_norm(cur,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                }
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (!hparams.swin_norm) {
                cur = build_norm(ffn_inp,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
            }

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            if (hparams.swin_norm) {
                cur = build_norm(cur,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
                cb(cur, "ffn_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lgf->build_cvec(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);

        cb(cur, "result_norm", -1);
        res.t_embd = cur;

        // lm_head
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output_with_img_logits", -1);

        // TODO: this suppresses the output of image tokens, which is required to enable text-only outputs.
        // Needs to be removed once image outputs are supported.
        int img_token_end_idx = 8196;
        int img_token_start_idx = 4;
        int num_img_tokens = img_token_end_idx - img_token_start_idx;
        // creates 1d tensor of size num_img_tokens and values -FLT_MAX,
        // which ensures that text token values are always at least larger than image token values
        struct ggml_tensor * img_logits = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, num_img_tokens);
        img_logits = ggml_clamp(ctx0, img_logits, -FLT_MAX, -FLT_MAX);
        cb(img_logits, "img_logits", -1);

        cur = ggml_set_1d(ctx0, cur, img_logits, ggml_element_size(cur) * img_token_start_idx);

        cb(cur, "result_output", -1);
        res.t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }

    void build_wavtokenizer_dec(ggml_cgraph * gf) {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = build_inp_embd(model.tok_embd);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

        cur = ggml_conv_1d_ph(ctx0, model.conv1d, cur, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_b);

        // posnet
        for (uint32_t il = 0; il < hparams.posnet.n_layer; ++il) {
            const auto & layer = model.layers[il].posnet;

            inpL = cur;

            switch (il) {
                case 0:
                case 1:
                case 3:
                case 4:
                    {
                        cur = build_norm(cur,
                                layer.norm1,
                                layer.norm1_b,
                                LLM_NORM_GROUP, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv1, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv1_b);

                        cur = build_norm(cur,
                                layer.norm2,
                                layer.norm2_b,
                                LLM_NORM_GROUP, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv2, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv2_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
                case 2:
                    {
                        cur = build_norm(cur,
                                layer.attn_norm,
                                layer.attn_norm_b,
                                LLM_NORM_GROUP, 0);

                        struct ggml_tensor * q;
                        struct ggml_tensor * k;
                        struct ggml_tensor * v;

                        q = ggml_conv_1d_ph(ctx0, layer.attn_q, cur, 1, 1);
                        k = ggml_conv_1d_ph(ctx0, layer.attn_k, cur, 1, 1);
                        v = ggml_conv_1d_ph(ctx0, layer.attn_v, cur, 1, 1);

                        q = ggml_add(ctx0, q, layer.attn_q_b);
                        k = ggml_add(ctx0, k, layer.attn_k_b);
                        v = ggml_add(ctx0, v, layer.attn_v_b);

                        q = ggml_cont(ctx0, ggml_transpose(ctx0, q));
                        k = ggml_cont(ctx0, ggml_transpose(ctx0, k));

                        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);

                        kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f/sqrtf(float(hparams.posnet.n_embd)), 0.0f);

                        cur = ggml_mul_mat(ctx0, kq, v);

                        cur = ggml_conv_1d_ph(ctx0, layer.attn_o, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.attn_o_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
                case 5:
                    {
                        cur = build_norm(cur,
                                layer.norm,
                                layer.norm_b,
                                LLM_NORM_GROUP, 0);
                    } break;
                default: GGML_ABORT("unknown posnet layer");
            };
        }

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                model.tok_norm,
                model.tok_norm_b,
                LLM_NORM, -1);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        inpL = cur;

        // convnext
        for (uint32_t il = 0; il < hparams.convnext.n_layer; ++il) {
            const auto & layer = model.layers[il].convnext;

            cur = inpL;

            cur = ggml_conv_1d_dw_ph(ctx0, layer.dw, cur, 1, 1);
            cur = ggml_add(ctx0, cur, layer.dw_b);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            cur = build_norm(cur,
                    layer.norm,
                    layer.norm_b,
                    LLM_NORM, -1);

            cur = build_ffn(cur,
                    layer.pw1, layer.pw1_b, NULL,
                    NULL,      NULL,        NULL,
                    layer.pw2, layer.pw2_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);

            cur = ggml_mul(ctx0, cur, layer.gamma);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            inpL = ggml_add(ctx0, cur, inpL);
        }

        cur = inpL;

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = build_norm(cur,
                model.output_norm,
                model.output_norm_b,
                LLM_NORM, -1);

        // lm_head
        cur = build_lora_mm(model.output, cur);

        cur = ggml_add(ctx0, cur, model.output_b);

        cb(cur, "result_embd", -1);
        res.t_embd = cur;

        ggml_build_forward_expand(gf, cur);
    }
};

llama_graph_result llama_model::build_graph(
          ggml_context * ctx,
           ggml_cgraph * gf,
         llama_graph_i * lgf,
   const llama_cparams & cparams,
   const llama_ubatch  & ubatch,
                  bool   worst_case) const {
    struct llm_build_context llm(ctx, lgf, *this, cparams, ubatch, worst_case);

    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                llm.build_llama(gf);
            } break;
        case LLM_ARCH_DECI:
            {
                llm.build_deci(gf);
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                llm.build_baichuan(gf);
            } break;
        case LLM_ARCH_FALCON:
            {
                llm.build_falcon(gf);
            } break;
        case LLM_ARCH_GROK:
            {
                llm.build_grok(gf);
            } break;
        case LLM_ARCH_STARCODER:
            {
                llm.build_starcoder(gf);
            } break;
        case LLM_ARCH_REFACT:
            {
                llm.build_refact(gf);
            } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_NOMIC_BERT:
            {
                llm.build_bert(gf);
            } break;
        case LLM_ARCH_BLOOM:
            {
                llm.build_bloom(gf);
            } break;
        case LLM_ARCH_MPT:
            {
                llm.build_mpt(gf);
            } break;
         case LLM_ARCH_STABLELM:
            {
                llm.build_stablelm(gf);
            } break;
        case LLM_ARCH_QWEN:
            {
                llm.build_qwen(gf);
            } break;
        case LLM_ARCH_QWEN2:
            {
                llm.build_qwen2(gf);;
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                llm.build_qwen2vl(gf);
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                llm.build_qwen2moe(gf);
            } break;
        case LLM_ARCH_PHI2:
            {
                llm.build_phi2(gf);
            } break;
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
            {
                llm.build_phi3(gf);
            } break;
        case LLM_ARCH_PLAMO:
            {
                llm.build_plamo(gf);
            } break;
        case LLM_ARCH_GPT2:
            {
                llm.build_gpt2(gf);
            } break;
        case LLM_ARCH_CODESHELL:
            {
                llm.build_codeshell(gf);
            } break;
        case LLM_ARCH_ORION:
            {
                llm.build_orion(gf);
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                llm.build_internlm2(gf);
            } break;
        case LLM_ARCH_MINICPM3:
            {
                llm.build_minicpm3(gf);
            } break;
        case LLM_ARCH_GEMMA:
            {
                llm.build_gemma(gf);
            } break;
        case LLM_ARCH_GEMMA2:
            {
                llm.build_gemma2(gf);
            } break;
        case LLM_ARCH_STARCODER2:
            {
                llm.build_starcoder2(gf);
            } break;
        case LLM_ARCH_MAMBA:
            {
                llm.build_mamba(gf);
            } break;
        case LLM_ARCH_XVERSE:
            {
                llm.build_xverse(gf);
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                llm.build_command_r(gf);
            } break;
        case LLM_ARCH_COHERE2:
            {
                llm.build_cohere2(gf);
            } break;
        case LLM_ARCH_DBRX:
            {
                llm.build_dbrx(gf);
            } break;
        case LLM_ARCH_OLMO:
            {
                llm.build_olmo(gf);
            } break;
        case LLM_ARCH_OLMO2:
            {
                llm.build_olmo2(gf);
            } break;
        case LLM_ARCH_OLMOE:
            {
                llm.build_olmoe(gf);
            } break;
        case LLM_ARCH_OPENELM:
            {
                llm.build_openelm(gf);
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                llm.build_gptneox(gf);
            } break;
        case LLM_ARCH_ARCTIC:
            {
                llm.build_arctic(gf);
            } break;
        case LLM_ARCH_DEEPSEEK:
            {
                llm.build_deepseek(gf);
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                llm.build_deepseek2(gf);
            } break;
        case LLM_ARCH_CHATGLM:
            {
                llm.build_chatglm(gf);
            } break;
        case LLM_ARCH_BITNET:
            {
                llm.build_bitnet(gf);
            } break;
        //case LLM_ARCH_T5:
        //    {
        //        if (lctx.is_encoding) {
        //            llm.build_t5_enc(gf);
        //        } else {
        //            llm.build_t5_dec(gf);
        //        }
        //    } break;
        //case LLM_ARCH_T5ENCODER:
        //    {
        //        llm.build_t5_enc(gf);
        //    } break;
        case LLM_ARCH_JAIS:
            {
                llm.build_jais(gf);
            } break;
        case LLM_ARCH_NEMOTRON:
            {
                llm.build_nemotron(gf);
            } break;
        case LLM_ARCH_EXAONE:
            {
                llm.build_exaone(gf);
            } break;
        case LLM_ARCH_RWKV6:
            {
                llm.build_rwkv6(gf);
            } break;
        case LLM_ARCH_RWKV6QWEN2:
            {
                llm.build_rwkv6qwen2(gf);
            } break;
        case LLM_ARCH_CHAMELEON:
            {
                llm.build_chameleon(gf);
            } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
            {
                llm.build_wavtokenizer_dec(gf);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // add on pooling layer
    if (cparams.embeddings) {
        llm.append_pooling(gf);
    }

    return llm.res;
}

//
// interface implementation
//

struct llama_model_params llama_model_default_params() {
    struct llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.n_gpu_layers                =*/ 0,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
    };

#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    result.n_gpu_layers = 999;
#endif

    return result;
}

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model) {
    return &model->vocab;
}

void llama_free_model(struct llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(struct llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const struct llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const struct llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_layer(const struct llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_model_n_head(const struct llama_model * model) {
    return model->hparams.n_head();
}

// deprecated
int32_t llama_n_ctx_train(const struct llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const struct llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const struct llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const struct llama_model * model) {
    return llama_model_n_head(model);
}

enum llama_rope_type llama_model_rope_type(const struct llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_ORION:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_CHAMELEON:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_MINICPM3:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const struct llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const struct llama_model * model) {
    return (int)model->gguf_kv.size();
}

int32_t llama_model_meta_key_by_index(const struct llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t llama_model_size(const struct llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const struct llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE_N)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const struct llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const struct llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_MAMBA:  return true;
        case LLM_ARCH_RWKV6:  return true;
        case LLM_ARCH_RWKV6QWEN2: return true;
        default:              return false;
    }
}
