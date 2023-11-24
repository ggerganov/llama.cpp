#include <set>

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_PERSIMMON,
    LLM_ARCH_REFACT,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_UNKNOWN,
};

enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,
    LLM_KV_ROPE_SCALING_TYPE,
    LLM_KV_ROPE_SCALING_FACTOR,
    LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
    LLM_KV_ROPE_SCALING_FINETUNED,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_ADD_BOS,
    LLM_KV_TOKENIZER_ADD_EOS,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
};

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_1B,
    MODEL_3B,
    MODEL_7B,
    MODEL_8B,
    MODEL_13B,
    MODEL_15B,
    MODEL_30B,
    MODEL_34B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
};

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

struct LLM_KV {
  LLM_KV(llm_arch arch) : arch(arch) {}
  
  llm_arch arch;

  std::string operator()(llm_kv kv) const; // moved to llama.cpp file

};

enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_TOKEN_EMBD_NORM,
    LLM_TENSOR_POS_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ROPE_FREQS,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_QKV,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_NORM_2,
    LLM_TENSOR_ATTN_ROT_EMBD,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
};


struct llama_cparams {
    uint32_t n_ctx;       // context size used during inference
    uint32_t n_batch;
    uint32_t n_threads;       // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

    float    rope_freq_base;
    float    rope_freq_scale;

    uint32_t n_yarn_orig_ctx;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool mul_mat_q;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_norm_b;
    struct ggml_tensor * attn_norm_2;
    struct ggml_tensor * attn_norm_2_b;
    struct ggml_tensor * attn_q_norm;
    struct ggml_tensor * attn_q_norm_b;
    struct ggml_tensor * attn_k_norm;
    struct ggml_tensor * attn_k_norm_b;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;
    struct ggml_tensor * wqkv;

    // attention bias
    struct ggml_tensor * bo;
    struct ggml_tensor * bqkv;

    // normalization
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_norm_b;

    // ff
    struct ggml_tensor * ffn_gate; // w1
    struct ggml_tensor * ffn_down; // w2
    struct ggml_tensor * ffn_up;   // w3

    // ff bias
    struct ggml_tensor * ffn_down_b; // b2
    struct ggml_tensor * ffn_up_b;   // b3
};

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
	return seq_id.find(id) != seq_id.end();
    }
};

struct llama_buffer {
    void * data = NULL;
    size_t size = 0;

    // fallback to malloc / free
    // useful in cases where CUDA can try to allocate PINNED memory
    bool fallback = false;

  void resize(size_t n) ;	


  ~llama_buffer();

};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<llama_kv_cell> cells;

    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_buffer buf;

    ~llama_kv_cache() {
	if (ctx) {
	    ggml_free(ctx);
	}

#ifdef GGML_USE_CUBLAS
	if (ggml_cublas_loaded()) {
	    ggml_cuda_free_data(k);
	    ggml_cuda_free_data(v);
	}
#endif
    }
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
	token text;
	float score;
	ttype type;
    };

    enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::unordered_map<token, id> special_tokens_cache;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1;
    id special_pad_id = -1;

    int special_add_bos = -1; // -1 unknown, 1 add, 0 don't add.
    int special_add_eos = -1; // -1 unknown, 1 add, 0 don't add.

    id linefeed_id       = 13;
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;

    int find_bpe_rank(std::string token_left, std::string token_right) const {
	GGML_ASSERT(token_left.find(" ") == std::string::npos);
	GGML_ASSERT(token_left.find("\n") == std::string::npos);
	GGML_ASSERT(token_right.find(" ") == std::string::npos);
	GGML_ASSERT(token_right.find("\n") == std::string::npos);

	auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
	if (it == bpe_ranks.end()) {
	    return -1;
	}

	return it->second;
    }
};

struct llama_mmap {
  void * addr;
  size_t size;
  
  llama_mmap(const llama_mmap &) = delete;
  
  llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false);
  ~llama_mmap();

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;
#else
    static constexpr bool SUPPORTED = false;
#endif
};


struct llama_hparams {
    bool     vocab_only;
    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    float    rope_freq_base_train;
    float    rope_freq_scale_train;
    uint32_t n_yarn_orig_ctx;
    int8_t   rope_scaling_type_train : 3;
    bool     rope_finetuned : 1;

    float f_clamp_kqv;
    float f_max_alibi_bias;

  bool operator!=(const llama_hparams & other) const;
    uint32_t n_gqa() const {
	return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
	return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
	return n_embd/n_gqa();
    }
};

struct llama_mlock {
  void * addr = NULL;
  size_t size = 0;
  bool failed_already = false;
  llama_mlock() ;

  llama_mlock(const llama_mlock &) = delete;
  ~llama_mlock();
  void init(void * ptr);
  void grow_to(size_t target_size);
#ifdef _POSIX_MEMLOCK_RANGE
  static constexpr bool SUPPORTED = true;
  static size_t lock_granularity();
#ifdef __APPLE__
#define MLOCK_SUGGESTION						\
  "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
  "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION						\
  "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
#endif
  bool raw_lock(const void * addr, size_t size) const ;
#undef MLOCK_SUGGESTION
  static void raw_unlock(void * addr, size_t size);
#elif defined(_WIN32)
  static constexpr bool SUPPORTED = true; 
  static size_t lock_granularity();	
  bool raw_lock(void * ptr, size_t len) const ;
  static void raw_unlock(void * ptr, size_t len);
#else
    static constexpr bool SUPPORTED = false;
  static size_t lock_granularity();
  bool raw_lock(const void * addr, size_t len) const;
  static void raw_unlock(const void * addr, size_t len);
#endif
};


struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llm_arch    arch  = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embd;
    struct ggml_tensor * pos_embd;
    struct ggml_tensor * tok_norm;
    struct ggml_tensor * tok_norm_b;

    struct ggml_tensor * output_norm;
    struct ggml_tensor * output_norm_b;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    int n_gpu_layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_mlock mlock_buf;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    ~llama_model() {
	if (ctx) {
	    ggml_free(ctx);
	}

#ifdef GGML_USE_CUBLAS
	if (ggml_cublas_loaded()) {
	    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
		ggml_cuda_free_data(tensors_by_name[i].second);
	    }
	    ggml_cuda_free_scratch();
	}
#endif

#if defined(GGML_USE_CLBLAST)
	for (size_t i = 0; i < tensors_by_name.size(); ++i) {
	    ggml_cl_free_data(tensors_by_name[i].second);
	}
#endif
    }
};

struct llama_context {
    llama_context(const llama_model & model) : model(model), t_start_us(model.t_start_us), t_load_us(model.t_load_us) {}
  ~llama_context();

    llama_cparams cparams;

    const llama_model & model;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_start_us;
    int64_t t_load_us;
    int64_t t_sample_us = 0;
    int64_t t_p_eval_us = 0;
    int64_t t_eval_us   = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    int32_t n_eval   = 0; // number of eval calls

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    llama_buffer buf_compute;

    llama_buffer buf_alloc;
    ggml_allocr * alloc = NULL;

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = NULL;
#endif

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif
};


struct LLM_TN {
  LLM_TN(llm_arch arch) ;

  llm_arch arch;

  std::string operator()(llm_tensor tensor) const;

  std::string operator()(llm_tensor tensor, const std::string & suffix) const ;

  std::string operator()(llm_tensor tensor, int bid) const ;

  std::string operator()(llm_tensor tensor, const std::string & suffix, int bid) const ;

};
