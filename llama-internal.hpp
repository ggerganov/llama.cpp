#include <set>
#include <queue>
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
    LLM_ARCH_QWEN,
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

#include "llama-layer.hpp"

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
  uint32_t used = 0; // used cells (i.e. at least one seq_id);

    // computed before each graph build
    uint32_t n = 0;

    std::vector<llama_kv_cell> cells;

    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_buffer buf;

  ~llama_kv_cache();
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

  int find_bpe_rank(std::string token_left, std::string token_right) const;
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

  ~llama_model() ;

};

struct llama_context {
  llama_context(const llama_model & model);
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


struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

  llama_file(const char * fname, const char * mode) ;
  size_t tell() const;
  void seek(size_t offset, int whence) const;
  void read_raw(void * ptr, size_t len) const;
  uint32_t read_u32() const;
  void write_raw(const void * ptr, size_t len) const ;
  void write_u32(std::uint32_t val) const;
  ~llama_file();

};


struct llama_state {
  llama_state();
    // We save the log callback globally
    ggml_log_callback log_callback;
    void * log_callback_user_data = nullptr;
  bool operator!=(const llama_hparams & other) const;
  static llama_state g_state;
};



struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t  n_bytes    = 0;

    bool use_mmap = false;

    llama_file  file;
    llama_ftype ftype;
    llama_fver  fver;

    std::unique_ptr<llama_mmap> mapping;

    struct gguf_context * ctx_gguf = NULL;
    struct ggml_context * ctx_meta = NULL;

  llama_model_loader(const std::string & fname, bool use_mmap) ;

  ~llama_model_loader();

  std::string get_arch_name() const;

  enum llm_arch get_arch() const ;
  const char * get_tensor_name(int i) const;

  struct ggml_tensor * get_tensor_meta(int i) const;

  void calc_sizes(size_t & ctx_size_p, size_t & mmapped_size_p) const;

  struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, struct ggml_tensor * meta, ggml_backend_type backend) ;

  struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, ggml_backend_type backend, bool required = true) ;

  void done_getting_tensors() const;

  size_t file_offset(const char * name) const;


  void load_data_for(struct ggml_tensor * cur) const ;
  void load_all_data(struct ggml_context * ctx, llama_progress_callback progress_callback, void * progress_callback_user_data, llama_mlock * lmlock) ;
};

struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t * ptr;
    size_t size_written = 0;
  llama_data_buffer_context(uint8_t * p) ;
  void write(const void * src, size_t size) override ;
  size_t get_size_written() override ;
};

struct llama_data_file_context : llama_data_context {
    llama_file * file;
    size_t size_written = 0;
  llama_data_file_context(llama_file * f);
  size_t get_size_written() override ;
  void write(const void * src, size_t size);
};


struct llama_beam {
  std::vector<llama_token> tokens;
  float p;  // Cumulative beam probability (renormalized relative to all beams)
  bool eob; // Initialize end-of-beam to false. Callback sets this to true.
  // Sort beams by probability. In case of ties, prefer beams at eob.
  bool operator<(const llama_beam & rhs) const ;
  void shift_tokens(const size_t n) ;
  llama_beam_view view() const;
};

// A struct for calculating logit-related info.
struct llama_logit_info {
    const float * const logits;
    const int n_vocab;
    const float max_l;
    const float normalizer;
    struct sum_exp {
	float max_l;
	float operator()(float sum, float l) const { return sum + std::exp(l - max_l); }
    };
  llama_logit_info(llama_context * ctx);
  llama_token_data get_token_data(const llama_token token_id) const ;
  std::vector<llama_token_data> top_k(size_t k) ;
  float probability_from_logit(float logit) const ;
};


struct llama_beam_search_data {
  llama_context * ctx;
  size_t n_beams;
  int n_past;
  int n_predict;
  std::vector<llama_beam> beams;
  std::vector<llama_beam> next_beams;
  size_t common_prefix_length;
  std::vector<llama_beam_view> beam_views;
  llama_beam_search_data(llama_context * ctx, size_t n_beams, int n_past, int n_predict);
  void collapse_beams(const size_t beam_idx) ;
  void fill_next_beams_by_top_probabilities(llama_beam & beam) ;
  size_t find_common_prefix_length() ;
  llama_beams_state get_beams_state(const bool last_call) ;
  void loop(const llama_beam_search_callback_fn_t callback, void * const callback_data);
  static void renormalize_beam_probabilities(std::vector<llama_beam> & beams) ;
  size_t top_beam_index();
  void update_beams_from_beam_views();
};

using llm_build_cb = std::function<void(struct ggml_tensor * cur, const char * name, int nl)>;

enum llm_rope_type {
    LLM_ROPE,
    LLM_ROPE_NEOX,
    LLM_ROPE_GLM,
};

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llm_build_context {
    const llama_model    & model;
    const llama_hparams  & hparams;
    const llama_cparams  & cparams;
    const llama_batch    & batch;
    const llama_kv_cache & kv_self;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head;
    const int64_t n_embd_gqa;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_kv;     // size of KV cache to consider (n_kv <= n_ctx)
    const int32_t kv_head;  // index of where we store new KV data in the cache
    const int32_t n_orig_ctx;

    const bool do_rope_shift;

    const llm_build_cb & cb;

    llama_buffer & buf_compute;

    struct ggml_context * ctx0 = nullptr;

    // TODO: consider making the entire interface noexcept
    llm_build_context(
	llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
	bool   worst_case);

  void init() ;
  void free() ;
  struct ggml_cgraph * build_llama() ;
  struct ggml_cgraph * build_baichuan() ;
  struct ggml_cgraph * build_falcon() ;
  struct ggml_cgraph * build_starcoder() ;
  struct ggml_cgraph * build_persimmon() ;
  struct ggml_cgraph * build_refact() ;
  struct ggml_cgraph * build_bloom() ;
  struct ggml_cgraph * build_mpt() ;
  struct ggml_cgraph * build_stablelm();
  struct ggml_cgraph * build_qwen();
};


enum llm_offload_func_e {
    OFFLOAD_FUNC_NOP,
    OFFLOAD_FUNC,
    OFFLOAD_FUNC_KQ,
    OFFLOAD_FUNC_V,
    OFFLOAD_FUNC_NR,
    OFFLOAD_FUNC_EMB,
    OFFLOAD_FUNC_OUT,
};

struct llm_offload_trie {
  struct node {
    ~node() ;
    node * children[256] = { nullptr };
    llm_offload_func_e func = OFFLOAD_FUNC_NOP;
  };
  node * root = nullptr;
  llm_offload_trie();
  llm_offload_trie(const std::unordered_map<const char *, llm_offload_func_e> & map) ;
  ~llm_offload_trie();
  void add(const char * name, llm_offload_func_e func);
  llm_offload_func_e find(const char * name) const;
  
};

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};


struct llm_bigram_spm {
    struct comparator {
      bool operator()(llm_bigram_spm & l, llm_bigram_spm & r);
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm {
  llm_tokenizer_spm(const llama_vocab & vocab);
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output);


private:
  void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) ;
  void try_add_bigram(int left, int right) ;
  const llama_vocab & vocab;

  std::vector<llm_symbol> symbols;
  llm_bigram_spm::queue work_queue;

    std::map<std::string, std::pair<int, int>> rev_merge;
};

// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

struct llm_bigram_bpe {
    struct comparator {
      bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const ;
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe {
  llm_tokenizer_bpe(const llama_vocab & vocab);

  void tokenize(const std::string & text, std::vector<llama_vocab::id> & output);

private:
  void add_new_bigram(int left, int right) ;

  std::vector<std::string> bpe_gpt2_preprocess(const std::string & text) ;

  const llama_vocab & vocab;

  std::vector<llm_symbol> symbols;
  std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
};

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE{
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant{
  fragment_buffer_variant(llama_vocab::id _token);
  fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length);
  const FRAGMENT_BUFFER_VARIANT_TYPE type;
  const llama_vocab::id token;
  const std::string _dummy;
  const std::string & raw_text;
  const uint64_t offset;
  const uint64_t length;
};

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>>   rules;
    std::vector<std::vector<const llama_grammar_element *>> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8                                      partial_utf8;
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
};

struct quantize_state_internal {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv    = 0;
    int n_feed_forward_w2 = 0;
    int i_attention_wv    = 0;
    int i_feed_forward_w2 = 0;

    int n_k_quantized     = 0;
    int n_fallback        = 0;

    quantize_state_internal(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};
