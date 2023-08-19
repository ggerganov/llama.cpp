#include "ggml.h"
#include "cmpnct_gpt2bpe.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// default hparams
struct falcon_hparams {
    size_t n_merges = 0;
    size_t n_vocab  = 0;
    uint32_t n_ctx    = 0;
    uint32_t n_embd   = 0;
    uint32_t n_head   = 0;
    uint32_t n_head_kv = 1; // Needs to be 1 for 7B model
    uint32_t n_ff = 0;
    uint32_t n_block  = 0;
    float norm_eps = 1e-5;
};
struct falcon_block {
    // normalization
    struct ggml_tensor* input_layernorm;
    struct ggml_tensor* input_layernorm_b;
    struct ggml_tensor* attention_norm;    // Falcon-40B only
    struct ggml_tensor* attention_norm_b;  // Falcon-40B only

    // attention
    struct ggml_tensor* query_key_value;
    struct ggml_tensor* wo;

    // ff
    struct ggml_tensor* ffn_up;
    struct ggml_tensor* ffn_down;
};

struct falcon_model {
    falcon_hparams hparams;

    struct ggml_tensor* tok_embeddings;
    struct ggml_tensor* output_norm;
    struct ggml_tensor* output_norm_b;
    struct ggml_tensor* lm_head;

    std::vector<falcon_block> blocks;

    // key + value memory
    struct ggml_tensor* memory_k;
    struct ggml_tensor* memory_v;

    struct gguf_context * ggufctx;
    struct ggml_context * ctx;
    struct ggml_context * kvctx;

    std::map<std::string, struct ggml_tensor*> tensors;
};

struct gpt_params {
    int32_t seed      = -1;  // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    uint32_t n_predict = 200; // new tokens to predict
    uint32_t n_batch   = 512;   // batch size for prompt processing

    // sampling parameters
    int32_t top_k          = 40;
    float top_p            = 1.0f;
    float temp             = 0.8f;
    int32_t repeat_last_n  = 64;
    float repeat_penalty   = 1.02f;

    std::string model      = ""; // model path
    std::string prompt     = "";

    std::string token_test = "";
    bool    interactive      = false;
    int32_t interactive_port = -1;
    int32_t n_gpu_layers     = 0;
};

void gpt_print_usage(int /*argc*/, char ** argv, const gpt_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -ngl N, --gpu-layers N  number of layers to offload to GPU on supported models (default: %d)\n", params.n_gpu_layers);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME\n");
    fprintf(stderr, "                        load prompt from a file\n");
    fprintf(stderr, "  -tt TOKEN_TEST, --token_test TOKEN_TEST\n");
    fprintf(stderr, "                        test tokenization\n");
    fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n", params.n_predict);
    fprintf(stderr, "  --top_k N             top-k sampling, 0 = n_vocab (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)\n", (double)params.repeat_penalty);
    fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

// Function to check if the next argument exists
std::string get_next_arg(int& i, int argc, char** argv, const std::string& flag, gpt_params& params) {
    if (i + 1 < argc && argv[i + 1][0] != '-') {
        return argv[++i];
    } else {
        fprintf(stderr, "error: %s requires one argument.\n", flag.c_str());
        gpt_print_usage(argc, argv, params);
        exit(0);
    }
}

bool gpt_params_parse(int argc, char ** argv, gpt_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
            params.n_gpu_layers = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = get_next_arg(i, argc, argv, arg, params);
        } else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--top_k") {
            params.top_k = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--top_p") {
            params.top_p = std::stof(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--temp") {
            params.temp = std::stof(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--repeat-last-n") {
            params.repeat_last_n = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "--repeat-penalty") {
            params.repeat_penalty = std::stof(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch= std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-m" || arg == "--model") {
            params.model = get_next_arg(i, argc, argv, arg, params);
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "-ip" || arg == "--interactive-port") {
            params.interactive = true;
            params.interactive_port = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, params);
            exit(0);
        } else if (arg == "-f" || arg == "--file") {
            get_next_arg(i, argc, argv, arg, params);
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-tt" || arg == "--token_test") {
            params.token_test = get_next_arg(i, argc, argv, arg, params);
        }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            gpt_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

gpt2bpe_vocab::id sample_top_k_top_p_repeat(
        const gpt2bpe_vocab & vocab,
        const float * logits,
        const int32_t * last_n_tokens_data,
        size_t last_n_tokens_data_size,
        int    top_k,
        double top_p,
        double temp,
        int repeat_last_n,
        float repeat_penalty,
        std::mt19937 & rng) {

    int n_logits = vocab.id_to_token.size();

    const auto * plogits = logits;

    const auto last_n_tokens = std::vector<int32_t>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_data_size);

    if (temp <= 0) {
        // select the token with the highest logit directly
        float max_logit = plogits[0];
        gpt2bpe_vocab::id max_id = 0;

        for (int i = 1; i < n_logits; ++i) {
            if (plogits[i] > max_logit) {
                max_logit = plogits[i];
                max_id = i;
            }
        }
        return max_id;
    }


    std::vector<std::pair<double, gpt2bpe_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (repeat_last_n > 0 && std::find(last_n_tokens.end()-repeat_last_n, last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
        }
    }

    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, gpt2bpe_vocab::id> & a, const std::pair<double, gpt2bpe_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                top_k = i + 1;
                probs.resize(top_k);
                logits_id.resize(top_k);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

//    printf("\n");
//    for (int i = 0; i < (int) probs.size(); i++) {
//    for (int i = 0; i < 10; i++) {
//        printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
//    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;

}

struct ggml_tensor * get_tensor_ex( struct ggml_context * ctx, std::string name){

    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if( cur == NULL ) {
        fprintf(stdout, "%s: tensor '%s' not found!\n", __func__, name.c_str());
    } else {
//        fprintf(stdout, "%s: n_dims = %d, name = '%s'\n", __func__, cur->n_dims, cur->name);
    }

    return cur;
}

// load the model's weights from a file
bool falcon_model_load(const std::string & fname, falcon_model & model, gpt2bpe_vocab & vocab) {
    printf("%s: loading model from '%s'..\n", __func__, fname.c_str());

    model.ctx = NULL;

    struct gguf_init_params ggufparams = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &model.ctx,
    };

    auto & ggufctx = model.ggufctx;

    ggufctx  = gguf_init_from_file(fname.c_str(), ggufparams);

    if (!ggufctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    fprintf(stdout, "%s: gguf version     = %d\n", __func__, gguf_get_version(ggufctx));
    fprintf(stdout, "%s: gguf alignment   = %zu\n", __func__, gguf_get_alignment(ggufctx));
    fprintf(stdout, "%s: gguf data offset = %zu\n", __func__, gguf_get_data_offset(ggufctx));

    // print all kv
    #if 0
    {
        const int n_kv = gguf_get_n_kv(ggufctx);

        fprintf(stdout, "%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ggufctx, i);

            fprintf(stdout, "%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }
    #endif

    // print some standard metadata
    {
        int keyidx;

        keyidx = gguf_find_key(ggufctx, "general.name");
        if (keyidx != -1) { fprintf(stdout, "%s: model name           = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.description");
        if (keyidx != -1) { fprintf(stdout, "%s: model description    = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.author");
        if (keyidx != -1) { fprintf(stdout, "%s: model author         = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.license");
        if (keyidx != -1) { fprintf(stdout, "%s: model license        = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.architecture");
        if (keyidx != -1) { fprintf(stdout, "%s: model architecture   = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.file_type");
        if (keyidx != -1) { fprintf(stdout, "%s: model file type      = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "gptneox.tensor_data_layout");
        if (keyidx != -1) { fprintf(stdout, "%s: model data layout    = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
        keyidx = gguf_find_key(ggufctx, "general.source.hugginface.repository");
        if (keyidx != -1) { fprintf(stdout, "%s: model source HF repo = %s\n", __func__, gguf_get_val_str(ggufctx, keyidx)); }
    }

    // check required metadata
    {
        int keyidx;

        // check model architecture kv
        keyidx = gguf_find_key(ggufctx, "general.architecture");
        if (keyidx != -1) {
            if ( strcmp(gguf_get_val_str(ggufctx, keyidx), "falcon") != 0) {
                fprintf(stdout, "%s: model architecture not supported!\n", __func__);
                return false;
            }
        } else {
            fprintf(stdout, "%s: gguf model architecture not found!\n", __func__);
            return false;
        }

        // check model tensor data layout kv
        keyidx = gguf_find_key(ggufctx, "falcon.tensor_data_layout");
        if (keyidx != -1) {
            if ( strcmp(gguf_get_val_str(ggufctx, keyidx), "jploski") != 0) {
                fprintf(stdout, "%s: model tensor data layout not supported!\n", __func__);
                return false;
            }
        } else {
            fprintf(stdout, "%s: gguf model tensor data layout not found!\n", __func__);
            return false;
        }

    }

    // load hparams
    {
        auto & hparams = model.hparams;

        bool ok = true;
        int keyidx;

        if (ok) { keyidx = gguf_find_key(ggufctx, "falcon.context_length");
                  if (keyidx != -1) { hparams.n_ctx = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "falcon.embedding_length");
                  if (keyidx != -1) { hparams.n_embd = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "falcon.attention.head_count");
                  if (keyidx != -1) { hparams.n_head = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "falcon.feed_forward_length");
                  if (keyidx != -1) { hparams.n_ff = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "falcon.block_count");
                  if (keyidx != -1) { hparams.n_block = gguf_get_val_u32(ggufctx, keyidx); } else { ok = false; }  }

        if (ok) { keyidx = gguf_find_key(ggufctx, "falcon.attention.layer_norm_epsilon");
                  if (keyidx != -1) { hparams.norm_eps= gguf_get_val_f32(ggufctx, keyidx); } else { ok = false; }  }

        if (!ok) {
            fprintf(stderr, "%s: required hparam missing!\n", __func__);
            return false;
        }

        keyidx = gguf_find_key(ggufctx, "falcon.attention.head_count_kv");
        if (keyidx != -1) { hparams.n_head_kv = gguf_get_val_u32(ggufctx, keyidx); }


        printf("%s: n_ctx      = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd     = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head     = %d\n", __func__, hparams.n_head);
        printf("%s: n_head_kv  = %d\n", __func__, hparams.n_head_kv);
        printf("%s: n_block    = %d\n", __func__, hparams.n_block);
        printf("%s: norm_eps   = %g\n", __func__, hparams.norm_eps);

    }

    // load vocab
    {
        auto & hparams = model.hparams;

        int keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.model");

        if (keyidx != -1) {
            if ( strcmp(gguf_get_val_str(ggufctx, keyidx), "gpt2") != 0) {
                fprintf(stdout, "%s: tokenizer model not supported!\n", __func__);
                return false;
            }
        } else {
            fprintf(stdout, "%s: tokenizer model not found!\n", __func__);
            return false;
        }


        int tokens_keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.tokens");

        if (tokens_keyidx == -1) {
            fprintf(stdout, "%s: gpt2 tokenizer vocab not found!\n", __func__);
            return false;
        }

        int merges_keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.merges");

        if (merges_keyidx == -1) {
            fprintf(stdout, "%s: gpt2 tokenizer merges not found!\n", __func__);
            return false;
        }

        hparams.n_vocab = gguf_get_arr_n(ggufctx,tokens_keyidx);
        hparams.n_merges = gguf_get_arr_n(ggufctx,merges_keyidx);

        fprintf(stdout, "%s: gpt2 tokenizer vocab  = %zu\n", __func__, hparams.n_vocab);
        fprintf(stdout, "%s: gpt2 tokenizer merges = %zu\n", __func__, hparams.n_merges);

        for (size_t i = 0; i < hparams.n_vocab; i++) {
            std::string word = gguf_get_arr_str(ggufctx, tokens_keyidx, i);

//            printf("token %d = '%s'\n",i,word.c_str() );

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            if( vocab.id_to_token[i] == "\n" ) {
                vocab.linefeed_id = i;
            }
        }

        std::vector<std::pair<std::string, std::string>> bpe_merges;

        for (size_t i = 0; i < hparams.n_merges; i++) {

            std::string word = gguf_get_arr_str(ggufctx, merges_keyidx, i);

            // Split the merges
            std::string first, second;
            size_t pos = word.find(' ', 1); // Start the search from the second character
            if (pos != std::string::npos) {
                first = word.substr(0, pos);
                second = word.substr(pos + 1);
            }

            bpe_merges.push_back(std::make_pair(first, second));
        }

        vocab.populate_bpe_ranks(bpe_merges);


        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.bos_token_id"); if( keyidx != -1 ) {       vocab.special_bos_id = (int32_t)gguf_get_val_u32(ggufctx, keyidx); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.eos_token_id"); if( keyidx != -1 ) {       vocab.special_eos_id = (int32_t)gguf_get_val_u32(ggufctx, keyidx); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.unknown_token_id"); if( keyidx != -1 ) {   vocab.special_unk_id = (int32_t)gguf_get_val_u32(ggufctx, keyidx); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.separator_token_id"); if( keyidx != -1 ) { vocab.special_sep_id = (int32_t)gguf_get_val_u32(ggufctx, keyidx); }
        keyidx = gguf_find_key(ggufctx, "tokenizer.ggml.padding_token_id"); if( keyidx != -1 ) {   vocab.special_pad_id = (int32_t)gguf_get_val_u32(ggufctx, keyidx); }

        if( vocab.special_bos_id != -1 ) { fprintf(stdout, "%s: BOS token = %d '%s'\n", __func__, vocab.special_bos_id, vocab.id_to_token[vocab.special_bos_id].c_str() ); }
        if( vocab.special_eos_id != -1 ) { fprintf(stdout, "%s: EOS token = %d '%s'\n", __func__, vocab.special_eos_id, vocab.id_to_token[vocab.special_eos_id].c_str() ); }
        if( vocab.special_unk_id != -1 ) { fprintf(stdout, "%s: UNK token = %d '%s'\n", __func__, vocab.special_unk_id, vocab.id_to_token[vocab.special_unk_id].c_str() ); }
        if( vocab.special_sep_id != -1 ) { fprintf(stdout, "%s: SEP token = %d '%s'\n", __func__, vocab.special_sep_id, vocab.id_to_token[vocab.special_sep_id].c_str() ); }
        if( vocab.special_pad_id != -1 ) { fprintf(stdout, "%s: PAD token = %d '%s'\n", __func__, vocab.special_pad_id, vocab.id_to_token[vocab.special_pad_id].c_str() ); }
        if( vocab.linefeed_id    != -1 ) { fprintf(stdout, "%s: LF token  = %d\n",      __func__, vocab.linefeed_id ); }

    }


    auto & ctx = model.ctx;
    size_t ctx_size = ggml_get_mem_size(ctx);

    printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));

    // print tensor info
    #if 0
    {
        const int n_tensors = gguf_get_n_tensors(ggufctx);

        fprintf(stdout, "%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ggufctx, i);
            const size_t offset = gguf_get_tensor_offset(ggufctx, i);

            fprintf(stdout, "%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }
    #endif

    // prepare memory for the weights
    {

        auto & hparams = model.hparams;

        const int n_block = hparams.n_block;

        model.blocks.resize(n_block);

        model.tok_embeddings = ggml_get_tensor(ctx, "token_embd.weight");

        model.output_norm = ggml_get_tensor(ctx, "output_norm.weight");
        model.output_norm_b = ggml_get_tensor(ctx, "output_norm.bias");
        model.lm_head = ggml_get_tensor(ctx, "output.weight");

        // map by name
        model.tensors["token_embd.weight"] = model.tok_embeddings;
        model.tensors["output_norm.weight"] = model.output_norm;
        model.tensors["output_norm.bias"] = model.output_norm_b;
        model.tensors["output.weight"] = model.lm_head;

        for (int i = 0; i < n_block; ++i) {

            auto& block = model.blocks[i];
            std::string blocknamestart = "blk." + std::to_string(i) + ".";

            block.input_layernorm   =  get_tensor_ex(ctx, blocknamestart + "attn_norm.weight" );
            block.input_layernorm_b =  get_tensor_ex(ctx, blocknamestart + "attn_norm.bias" );

            if ( hparams.n_head_kv == 8 ) { // Falcon-40B
                block.attention_norm   =  get_tensor_ex(ctx, blocknamestart + "attn_norm_2.weight" );
                block.attention_norm_b =  get_tensor_ex(ctx, blocknamestart + "attn_norm_2.bias" );
            }

            // query_key_value shape for config.multi_query == True:
            block.query_key_value = get_tensor_ex(ctx, blocknamestart + "attn_qkv.weight" );
            block.wo = get_tensor_ex(ctx, blocknamestart + "attn_output.weight" );

            block.ffn_up = get_tensor_ex(ctx, blocknamestart + "ffn_up.weight" );
            block.ffn_down = get_tensor_ex(ctx, blocknamestart + "ffn_down.weight" );

            // map by name
            if ( hparams.n_head_kv == 8 ) { // Falcon-40B
                // Falcon-40B:
                model.tensors[blocknamestart + "attn_norm.weight"] = block.input_layernorm;
                model.tensors[blocknamestart + "attn_norm.bias"] = block.input_layernorm_b;
                model.tensors[blocknamestart + "attn_norm_2.weight"] = block.attention_norm;
                model.tensors[blocknamestart + "attn_norm_2.bias"] = block.attention_norm_b;
            } else {
                // Falcon-7B:
                model.tensors[blocknamestart + "attn_norm.weight"] = block.input_layernorm;
                model.tensors[blocknamestart + "attn_norm.bias"] = block.input_layernorm_b;
            }

            model.tensors[blocknamestart + "attn_qkv.weight"] = block.query_key_value;
            model.tensors[blocknamestart + "attn_output.weight"] = block.wo;

            model.tensors[blocknamestart + "ffn_up.weight"] = block.ffn_up;
            model.tensors[blocknamestart + "ffn_down.weight"] = block.ffn_down;
        }
    }

    // key + value memory
    {
        const auto & kvctx = model.kvctx;
        const auto & hparams = model.hparams;

        const int n_block = hparams.n_block;
        const int n_ctx   = hparams.n_ctx;
        const int n_embd = hparams.n_embd;

        const int64_t n_mem      = n_block*n_ctx;
        const int64_t n_elements = n_embd*n_mem;

        // create the ggml context
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ size_t(n_elements*4+ggml_tensor_overhead()*2),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ false,
            };

            model.kvctx = ggml_init(params);
            if (!model.kvctx) {
                fprintf(stderr, "%s: kv ggml_init() failed\n", __func__);
                return false;
            }

        }


        model.memory_k = ggml_new_tensor_1d(kvctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(kvctx, GGML_TYPE_F16, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %" PRId64 "\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    return true;
}


// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool falcon_eval(
        const falcon_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt2bpe_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {


    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_block = hparams.n_block;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_head_kv = hparams.n_head_kv;
    const int n_vocab = hparams.n_vocab;
    const size_t head_dim = n_embd / n_head;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    // use 2 scratch buffers
    // TODO: very hacky solution - reimplement in a more elegant way
    static size_t scr0_size = 256u*1024*1024;
    static void * scr0 = malloc(scr0_size);

    static size_t scr1_size = 256u*1024*1024;
    static void * scr1 = malloc(scr1_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
//    gf.n_threads = n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);
//    struct ggml_tensor* repeat_dummy = ggml_new_tensor_3d(ctx0, inpL->type, head_dim, N + n_past, n_head);

    ggml_type wtype = GGML_TYPE_F32;
    const int sizeof_wtype = ggml_type_sizef(wtype);

    for (int il = 0; il < n_block; ++il) {
        struct ggml_tensor * cur;
        struct ggml_tensor * layernorm_output;

        ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

        // self-attention
        {
            layernorm_output = ggml_norm(ctx0, inpL);

            layernorm_output = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.blocks[il].input_layernorm, layernorm_output),
                        layernorm_output),
                    ggml_repeat(ctx0, model.blocks[il].input_layernorm_b, layernorm_output));

            if ( hparams.n_head_kv == 8 ) { // Falcon-40B
                cur = ggml_norm(ctx0, inpL);

                cur = ggml_add(ctx0,
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, model.blocks[il].attention_norm, cur),
                            cur),
                        ggml_repeat(ctx0, model.blocks[il].attention_norm_b, cur));
            }
            else { // Falcon 7B
                cur = layernorm_output;
            }

            // compute QKV

            cur = ggml_mul_mat(ctx0, model.blocks[il].query_key_value, cur);

            // Note that the strides for Kcur, Vcur are set up so that the
            // resulting views are misaligned with the tensor's storage
            // (by applying the K/V offset we shift the tensor's original
            // view to stick out behind the viewed QKV tensor's allocated
            // memory, so to say). This is ok because no actual accesses
            // happen to that out-of-range memory, but it can require some
            // trickery when trying to accurately dump these views for
            // debugging.

            struct ggml_tensor * Qcur = ggml_view_3d(
                ctx0, cur, head_dim, n_head, N,
                head_dim * sizeof_wtype,
                head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
                0);

            struct ggml_tensor * Kcur = ggml_view_3d(
                ctx0, cur, head_dim, n_head_kv, N,
                head_dim * sizeof_wtype,
                head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
                head_dim * n_head * sizeof_wtype);

            struct ggml_tensor * Vcur = ggml_view_3d(
                ctx0, cur, head_dim, n_head_kv, N,
                head_dim * sizeof_wtype,
                head_dim * (n_head + 2 * n_head_kv) * sizeof_wtype,
                head_dim * (n_head + n_head_kv) * sizeof_wtype);

            // using mode = 2 for neox mode
            Qcur = ggml_rope_inplace(ctx0, Qcur, n_past, head_dim, 2, 0);
            Kcur = ggml_rope_inplace(ctx0, Kcur, n_past, head_dim, 2, 0);

            // store key and value to memory
            {
                struct ggml_tensor* k = ggml_view_1d(
                    ctx0, model.memory_k, N * n_head_kv * head_dim,
                    (ggml_element_size(model.memory_k) * n_head_kv * head_dim) *
                        (il * n_ctx + n_past));
                struct ggml_tensor* v = ggml_view_1d(
                    ctx0, model.memory_v, N * n_head_kv * head_dim,
                    (ggml_element_size(model.memory_v) * n_head_kv * head_dim) *
                        (il * n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * K = ggml_permute(
                ctx0,
                ggml_reshape_3d(
                    ctx0,
                    ggml_view_1d(ctx0, model.memory_k, (n_past + N) * n_head_kv * head_dim,
                                 il * n_ctx *
                                     ggml_element_size(model.memory_k) *
                                     n_head_kv *
                                     head_dim),
                    head_dim, n_head_kv, n_past + N),
                0, 2, 1, 3);

            // K * Q

//            K = ggml_cont(ctx0, ggml_repeat2(ctx0, K, repeat_dummy));

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale_inplace(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(head_dim)))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor* V = ggml_permute(
                ctx0,
                ggml_reshape_3d(
                    ctx0,
                    ggml_view_1d(ctx0, model.memory_v, (n_past + N) * n_head_kv * head_dim,
                                 il * n_ctx *
                                     ggml_element_size(model.memory_v) *
                                     n_head_kv *
                                     head_dim),
                    head_dim, n_head_kv, n_past + N),
                0, 2, 1, 3);

//            V = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_repeat2(ctx0, V, repeat_dummy)));
            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection
            {
                cur = ggml_mul_mat(ctx0,
                        model.blocks[il].wo,
                        cur);
            }
        }

        ggml_set_scratch(ctx0, { 0, scr1_size, scr1, });

        struct ggml_tensor* inpFF = layernorm_output;
        struct ggml_tensor* attn_out = ggml_cpy(
            ctx0, cur, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

        {
            cur = ggml_mul_mat(ctx0, model.blocks[il].ffn_up, inpFF);
            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.blocks[il].ffn_down, cur);
        }

        cur = ggml_add(ctx0, cur, attn_out);
        cur = ggml_add(ctx0, cur, inpL);
        // input for next layer
        inpL = cur;
    }

    ggml_set_scratch(ctx0, { 0, scr0_size, scr0, });

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.output_norm, inpL),
                    inpL),
                ggml_repeat(ctx0, model.output_norm_b, inpL));
    }

    ggml_set_scratch(ctx0, { 0, 0, nullptr, });

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lm_head, inpL);

        //inpL = ggml_add(ctx0,
        //        ggml_repeat(ctx0, model.lmh_b, inpL),
        //        inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max_inplace(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
//    ggml_graph_compute       (ctx0, &gf);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *)ggml_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    gpt2bpe_vocab vocab;
    falcon_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!falcon_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    if (params.top_k == 0) {
        params.top_k = model.hparams.n_vocab;
    }

    printf("%s: seed           = %d\n",   __func__, params.seed);
    printf("%s: temp           = %.3f\n", __func__, params.temp);
    printf("%s: top_k          = %d\n",   __func__, params.top_k);
    printf("%s: top_p          = %.3f\n", __func__, params.top_p);
    printf("%s: repeat_last_n  = %d\n",   __func__, params.repeat_last_n);
    printf("%s: repeat_penalty = %.3f\n", __func__, params.repeat_penalty);

    std::mt19937 rng(params.seed);

    if (params.prompt.empty()) {
        params.prompt = "Once upon";
    }

    std::vector<int32_t> last_n_tokens(model.hparams.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt2bpe_vocab::id> embd_inp = gpt2bpe_tokenize(vocab, params.prompt,false, false);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
//    for (size_t i = 0; i < embd_inp.size(); i++) {
//        printf("%s: token[%zu] = %6d, %s\n", __func__, i, embd_inp[i], vocab.id_to_token[embd_inp[i]].c_str());
//    }

    if( model.hparams.n_ctx < params.n_predict+embd_inp.size() ) {
        params.n_predict = model.hparams.n_ctx-embd_inp.size();
    }

    printf("%s: n_predict = %d\n", __func__, params.n_predict);
    printf("\n");

    std::vector<gpt2bpe_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    falcon_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!falcon_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const int repeat_last_n = params.repeat_last_n;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt2bpe_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = sample_top_k_top_p_repeat(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_last_n, repeat_penalty, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (size_t k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str()  );
        }
        fflush(stdout);

        // end of text token
        if (vocab.special_eos_id != -1 && embd.back() == vocab.special_eos_id) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
