#include "llama.h"

#include "ggml.h"

#include <cinttypes>
#include <fstream>
#include <random>
#include <unordered_map>
#include <queue>
#include <regex>
#include <cassert>
#include <cstring>

// determine number of model parts based on the dimension
static const std::unordered_map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_model {
    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::unordered_map<std::string, struct ggml_tensor *> tensors;
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct llama_context {
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls

    llama_model model;
    llama_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;
};

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.n_ctx      =*/ 512,
        /*.n_parts    =*/ -1,
        /*.seed       =*/ 0,
        /*.f16_kv     =*/ false,
        /*.logits_all =*/ false,
        /*.vocab_only =*/ false,
        /*.use_mlock  =*/ false,
        /*.embedding  =*/ false,
    };

    return result;
}

//
// model loading
//

static bool llama_model_load(
        const std::string & fname,
        llama_context & lctx,
        int n_ctx,
        int n_parts,
        ggml_type memory_type,
        bool vocab_only) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    const int64_t t_start_us = ggml_time_us();

    lctx.t_start_us = t_start_us;

    std::vector<char> f_buf(1024*1024);

    auto & model = lctx.model;
    auto & vocab = lctx.vocab;

    auto fin = std::ifstream(fname, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files!)\n",
                    __func__, fname.c_str());
            return false;
        }
        if (magic != LLAMA_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }

        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != LLAMA_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, fname.c_str(), format_version, LLAMA_FILE_VERSION);
            return false;
        }
    }

    int n_ff = 0;

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

        if (n_parts < 1) {
            n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
        }

        // temp warning to tell the user to use "--n_parts"
        if (hparams.f16 == 4 && n_parts != 1) {
            fprintf(stderr, "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
            fprintf(stderr, "%s: use '--n_parts 1' if necessary\n", __func__);
        }

        fprintf(stderr, "%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        fprintf(stderr, "%s: n_embd  = %d\n", __func__, hparams.n_embd);
        fprintf(stderr, "%s: n_mult  = %d\n", __func__, hparams.n_mult);
        fprintf(stderr, "%s: n_head  = %d\n", __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer = %d\n", __func__, hparams.n_layer);
        fprintf(stderr, "%s: n_rot   = %d\n", __func__, hparams.n_rot);
        fprintf(stderr, "%s: f16     = %d\n", __func__, hparams.f16);
        fprintf(stderr, "%s: n_ff    = %d\n", __func__, n_ff);
        fprintf(stderr, "%s: n_parts = %d\n", __func__, n_parts);
    }

    // load vocab
    {
        std::string word;
        vocab.id_to_token.resize(model.hparams.n_vocab);
        std::vector<char> tmp(64);

        for (int i = 0; i < model.hparams.n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                word.assign(tmp.data(), len);
            } else {
                word.clear();
            }

            float score;
            fin.read((char *) &score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto &tok_score = vocab.id_to_token[i];
            tok_score.tok = word;
            tok_score.score = score;
        }
    }

    if (vocab_only) {
        return true;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // wtype is for per-layer weights, while vtype is for other weights
    ggml_type wtype, vtype;
    switch (model.hparams.f16) {
        case 0: wtype = vtype = GGML_TYPE_F32;  break;
        case 1: wtype = vtype = GGML_TYPE_F16;  break;
        case 2: wtype = vtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = vtype = GGML_TYPE_Q4_1; break;
        case 4: wtype = GGML_TYPE_Q4_1; vtype = GGML_TYPE_F16; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*n_vocab*ggml_type_sizef(vtype); // tok_embeddings

        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm

        ctx_size += n_embd*n_vocab*ggml_type_sizef(vtype); // output

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attention_norm

        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wq
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wk
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wv
        ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wo

        ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm

        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w1
        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w2
        ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w3

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(memory_type); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(memory_type); // memory_v

        ctx_size += (5 + 10*n_layer)*256; // object overhead

        fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ggml_new_tensor_2d(ctx, vtype, n_embd, n_vocab);

        model.norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output = ggml_new_tensor_2d(ctx, vtype,         n_embd, n_vocab);

        // map by name
        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;

        model.tensors["norm.weight"]   = model.norm;
        model.tensors["output.weight"] = model.output;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);

            // map by name
            model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;

            model.tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
            model.tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
            model.tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;

            model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;

            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, memory_type, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, memory_type, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        fprintf(stderr, "%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    std::vector<uint8_t> tmp;

    for (int i = 0; i < n_parts; ++i) {
        const int part_id = i;
        //const int part_id = n_parts - i - 1;

        std::string fname_part = fname;
        if (i > 0) {
            fname_part += "." + std::to_string(i);
        }

        fprintf(stderr, "%s: loading model part %d/%d from '%s'\n", __func__, i+1, n_parts, fname_part.c_str());

        fin = std::ifstream(fname_part, std::ios::binary);
        fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
        fin.seekg(file_offset);

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            fprintf(stderr, "%s: ", __func__);

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ftype;

                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

                if (fin.eof()) {
                    break;
                }

                int32_t nelements = 1;
                int32_t ne[2] = { 1, 1 };
                for (int i = 0; i < n_dims; ++i) {
                    fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                std::string name(length, 0);
                fin.read(&name[0], length);

                if (model.tensors.find(name.data()) == model.tensors.end()) {
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    return false;
                }

                // split_type = 0: split by columns
                // split_type = 1: split by rows
                int split_type = 0;

                // split_type = 0:
                // regex:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = 1:
                // regex:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                if (name.find("tok_embeddings") != std::string::npos) {
                    split_type = 0;
                } else if (name.find("layers") != std::string::npos) {
                    if (name.find("attention.wo.weight") != std::string::npos) {
                        split_type = 0;
                    } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                        split_type = 0;
                    } else {
                        split_type = 1;
                    }
                } else if (name.find("output") != std::string::npos) {
                    split_type = 1;
                }

                auto tensor = model.tensors[name.data()];

                if (n_dims == 1) {
                    if (ggml_nelements(tensor) != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                } else {
                    if (ggml_nelements(tensor)/n_parts != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                }

                if (n_dims == 1) {
                    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                        fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                        return false;
                    }
                } else {
                    if (split_type == 0) {
                        if (tensor->ne[0]/n_parts != ne[0] || tensor->ne[1] != ne[1]) {
                            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0]/n_parts, tensor->ne[1], ne[0], ne[1]);
                            return false;
                        }
                    } else {
                        if (tensor->ne[0] != ne[0] || tensor->ne[1]/n_parts != ne[1]) {
                            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0], tensor->ne[1]/n_parts, ne[0], ne[1]);
                            return false;
                        }
                    }
                }

                if (0) {
                    static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                    fprintf(stderr, "%24s - [%5d, %5d], type = %6s, split = %d\n", name.data(), ne[0], ne[1], ftype_str[ftype], split_type);
                }

                size_t bpe = 0;

                switch (ftype) {
                    case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                    case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                    case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                    case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                    default:
                            {
                                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                                return false;
                            }
                };

                if (n_dims == 1 || n_parts == 1) {
                    if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                        return false;
                    }

                    if (part_id == 0) {
                        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
                    } else {
                        fin.seekg(ggml_nbytes(tensor), std::ios::cur);
                    }

                    total_size += ggml_nbytes(tensor);
                } else {
                    if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)/n_parts) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ggml_nbytes(tensor)/n_parts, nelements*bpe);
                        return false;
                    }

                    if (split_type == 0) {
                        const int np0 = ne[0];

                        const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                        assert(row_size == tensor->nb[1]);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = i1*row_size;
                            const size_t offset = offset_row + ((part_id*np0)/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
                        }
                    } else {
                        const int np1 = ne[1];

                        const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = (i1 + part_id*np1)*row_size;
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
                        }
                    }

                    total_size += ggml_nbytes(tensor)/n_parts;
                }

                //fprintf(stderr, "%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
                if (++n_tensors % 8 == 0) {
                    fprintf(stderr, ".");
                    fflush(stderr);
                }
            }

            fprintf(stderr, " done\n");

            fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
        }

        fin.close();
    }

    lctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_eval_internal(
        llama_context & lctx,
    const llama_token * tokens,
            const int   n_tokens,
            const int   n_past,
            const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_embd/hparams.n_head;

    auto & mem_per_token = lctx.mem_per_token;

    // TODO: fix this hardcoded size
    static size_t buf_size = 2048u*1024*1024; // TMP !!!
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.3*(mem_per_token*N); // add 30% to account for ggml object overhead
        //fprintf(stderr, "\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

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
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, tokens, N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                    ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);


            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
        }

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // used at the end to optionally extract the embeddings
    struct ggml_tensor * embeddings = NULL;

    // norm
    {
        inpL = ggml_rms_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.norm, inpL),
                    inpL);

        embeddings = inpL;
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.output, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (lctx.embedding.size()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //fprintf(stderr, "used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }

    return true;
}

//
// tokenizer
//

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};

static std::vector<llama_vocab::id> llama_tokenize(const llama_vocab & vocab, const std::string & text, bool bos) {
    llama_tokenizer tokenizer(vocab);
    std::vector<llama_vocab::id> output;

    if (text.size() == 0) {
        return output;
    }

    if (bos) {
        output.push_back(1);
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// sampling
//

static void sample_top_k(std::vector<std::pair<double, llama_vocab::id>> & logits_id, int top_k) {
    // find the top k tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, llama_vocab::id> & a, const std::pair<double, llama_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

static llama_vocab::id llama_sample_top_p_top_k(
        llama_context & lctx,
        const std::vector<llama_vocab::id> & last_n_tokens,
        int top_k,
        double top_p,
        double temp,
        double repeat_penalty) {
    auto & rng = lctx.rng;

    const auto & vocab = lctx.vocab;
    const auto & logits = lctx.logits;

    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, llama_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (logits[i] < 0.0) {
                    logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(logits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    double maxl = -std::numeric_limits<double>::infinity();
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top k tokens
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
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

//
// quantization
//

// TODO: reuse code from the llama_model_load() somehow
bool llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, int itype, int qk) {
    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
        case 2: type = GGML_TYPE_Q4_0; break;
        case 3: type = GGML_TYPE_Q4_1; break;
        default: fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype); return 1;
    };

    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1) {
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, type);
        return false;
    }

    llama_vocab vocab;

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files!)\n",
                    __func__, fname_inp.c_str());
            return false;
        }
        if (magic != LLAMA_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *) &magic, sizeof(magic));

        uint32_t format_version;
        finp.read((char *) &format_version, sizeof(format_version));

        if (format_version != LLAMA_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, fname_inp.c_str(), format_version, LLAMA_FILE_VERSION);
            return false;
        }

        fout.write((char *) &format_version, sizeof(format_version));
    }

    llama_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //finp.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        finp.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        finp.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        finp.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        finp.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        finp.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        finp.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_mult  = %d\n", __func__, hparams.n_mult);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);

        fout.write((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fout.write((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fout.write((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fout.write((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fout.write((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fout.write((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fout.write((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fout.write((char *) &itype,           sizeof(hparams.f16));
    }

    // load vocab
    {
        const int32_t n_vocab = hparams.n_vocab;

        if (n_vocab != hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
            return false;
        }

        std::string word;
        vocab.id_to_token.resize(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word.resize(len);
            finp.read ((char *) word.data(), len);
            fout.write((char *) word.data(), len);

            float score;
            finp.read ((char *) &score, sizeof(score));
            fout.write((char *) &score, sizeof(score));

            vocab.token_to_id[word] = i;

            auto &tok_score = vocab.id_to_token[i];
            tok_score.tok = word;
            tok_score.score = score;
        }
    }

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t>     data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float>       data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (finp.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read (&name[0], length);

            {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                ".*weight",
            };

            bool quantize = false;
            for (const auto & s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = true;
                    break;
                }
            }

            // quantize only 2D tensors
            quantize &= (n_dims == 2);

            if (quantize) {
                if (ftype != 0 && ftype != 1) {
                    fprintf(stderr, "%s: unsupported ftype %d for integer quantization\n", __func__, ftype);
                    return false;
                }

                if (ftype == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = itype;
            } else {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements*bpe);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            if (quantize) {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (type) {
                    case GGML_TYPE_Q4_0:
                        {
                            cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], qk, hist_cur.data());
                        } break;
                    case GGML_TYPE_Q4_1:
                        {
                            cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], qk, hist_cur.data());
                        } break;
                    default:
                        {
                            fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, type);
                            return false;
                        }
                }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
                for (int i = 0; i < (int) hist_cur.size(); ++i) {
                    hist_all[i] += hist_cur[i];
                }

                for (int i = 0; i < (int) hist_cur.size(); ++i) {
                    printf("%5.3f ", hist_cur[i] / (float)nelements);
                }
                printf("\n");
            } else {
                printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

        {
            int64_t sum_all = 0;
            for (int i = 0; i < (int) hist_all.size(); ++i) {
                sum_all += hist_all[i];
            }

            printf("%s: hist: ", __func__);
            for (int i = 0; i < (int) hist_all.size(); ++i) {
                printf("%5.3f ", hist_all[i] / (float)sum_all);
            }
            printf("\n");
        }
    }

    finp.close();
    fout.close();

    return true;
}

//
// interface implementation
//

struct llama_context * llama_init_from_file(
                             const char * path_model,
            struct llama_context_params   params) {
    ggml_time_init();

    llama_context * ctx = new llama_context;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type type_memory = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!llama_model_load(path_model, *ctx, params.n_ctx, params.n_parts, type_memory,
                          params.vocab_only)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        llama_free(ctx);
        return nullptr;
    }

    if (params.use_mlock) {
        char *err;
        if (!ggml_mlock(ctx->model.ctx, &err)) {
            fprintf(stderr, "%s\n", err);
            free(err);
            llama_free(ctx);
            return nullptr;
        }
    }

    // reserve memory for context buffers
    {
        const auto & hparams = ctx->model.hparams;
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_ctx);
        }

        if (params.embedding){
            ctx->embedding.reserve(hparams.n_embd);
        }
    }

    return ctx;
}

void llama_free(struct llama_context * ctx) {
    if (ctx->model.ctx) {
        ggml_free(ctx->model.ctx);
    }

    delete ctx;
}

int llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
               int   itype,
               int   qk) {
    if (!llama_model_quantize_internal(fname_inp, fname_out, itype, qk)) {
        fprintf(stderr, "%s: failed to quantize\n", __func__);
        return 1;
    }

    return 0;
}

int llama_eval(
        struct llama_context * ctx,
           const llama_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!llama_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

int llama_tokenize(
        struct llama_context * ctx,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = llama_tokenize(ctx->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_n_vocab(struct llama_context * ctx) {
    return ctx->vocab.id_to_token.size();
}

int llama_n_ctx(struct llama_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

float * llama_get_logits(struct llama_context * ctx) {
    return ctx->logits.data();
}

float * llama_get_embeddings(struct llama_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_token_to_str(struct llama_context * ctx, llama_token token) {
    if (token >= llama_n_vocab(ctx)) {
        return nullptr;
    }

    return ctx->vocab.id_to_token[token].tok.c_str();
}

llama_token llama_token_bos() {
    return 1;
}

llama_token llama_token_eos() {
    return 2;
}

llama_token llama_sample_top_p_top_k(
          llama_context * ctx,
      const llama_token * last_n_tokens_data,
                    int   last_n_tokens_size,
                    int   top_k,
                 double   top_p,
                 double   temp,
                 double   repeat_penalty) {
    const int64_t t_start_sample_us = ggml_time_us();

    llama_token result = 0;

    // TODO: avoid this ...
    const auto last_n_tokens = std::vector<llama_token>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_size);

    result = llama_sample_top_p_top_k(
            *ctx,
            last_n_tokens,
            top_k,
            top_p,
            temp,
            repeat_penalty);

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;

    return result;
}


void llama_print_timings(struct llama_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    const int32_t n_sample = std::max(1, ctx->n_sample);
    const int32_t n_eval   = std::max(1, ctx->n_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0f);
    fprintf(stderr, "%s:   sample time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->t_sample_us, n_sample, 1e-3f * ctx->t_sample_us / n_sample);
    fprintf(stderr, "%s:     eval time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->t_eval_us,   n_eval,   1e-3f * ctx->t_eval_us   / n_eval);
    fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0f);
}

void llama_reset_timings(struct llama_context * ctx) {
    ctx->t_start_us = ggml_time_us();

    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
}

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}

