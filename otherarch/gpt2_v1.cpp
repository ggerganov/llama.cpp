#include "ggml_v1.h"
#include "otherarch.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>



// load the model's weights from a file
ModelLoadResult gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, FileFormat file_format) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return ModelLoadResult::FAIL;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return ModelLoadResult::FAIL;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return ModelLoadResult::FAIL;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats
    // in order to save memory and also to speed up the computation
    const ggml_v1_type wtype = model.hparams.f16 ? GGML_V1_TYPE_F16 : GGML_V1_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32); // ln_f_b

        ctx_size += n_vocab*n_embd*ggml_v1_type_size(wtype);         // wte
        ctx_size +=   n_ctx*n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32); // wpe

        ctx_size += n_layer*(n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // ln_1_g
        ctx_size += n_layer*(n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // ln_1_b

        ctx_size += n_layer*(n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // ln_2_g
        ctx_size += n_layer*(n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // ln_2_b

        ctx_size += n_layer*(3*n_embd*n_embd*ggml_v1_type_size(wtype));         // c_attn_attn_w
        ctx_size += n_layer*(       3*n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // c_attn_attn_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_v1_type_size(wtype));           // c_attn_proj_w
        ctx_size += n_layer*(       n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32));   // c_attn_proj_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_v1_type_size(wtype));         // c_mlp_fc_w
        ctx_size += n_layer*(       4*n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_v1_type_size(wtype));         // c_mlp_proj_w
        ctx_size += n_layer*(         n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32)); // c_mlp_proj_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_v1_type_size(GGML_V1_TYPE_F32); // memory_v

        ctx_size += (6 + 12*n_layer)*256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_v1_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = NULL,
        };

        model.ctx = ggml_v1_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_v1_init() failed\n", __func__);
            return ModelLoadResult::FAIL;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.ln_f_g = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32, n_embd);
        model.ln_f_b = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32, n_embd);

        model.wte = ggml_v1_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        model.wpe = ggml_v1_new_tensor_2d(ctx, GGML_V1_TYPE_F32, n_embd, n_ctx);

        // map by name
        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wte"] = model.wte;
        model.tensors["model/wpe"] = model.wpe;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g             = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32,   n_embd);
            layer.ln_1_b             = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32,   n_embd);

            layer.ln_2_g             = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32,   n_embd);
            layer.ln_2_b             = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32,   n_embd);

            layer.c_attn_attn_w      = ggml_v1_new_tensor_2d(ctx, wtype,         3*n_embd, n_embd);
            layer.c_attn_attn_b      = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w      = ggml_v1_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b      = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w         = ggml_v1_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
            layer.c_mlp_fc_b         = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w_trans = ggml_v1_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
            layer.c_mlp_proj_b       = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32,   n_embd);

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"]        = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"]        = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"]        = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"]        = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w_trans;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b;
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

        model.memory_k = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32, n_elements);
        model.memory_v = ggml_v1_new_tensor_1d(ctx, GGML_V1_TYPE_F32, n_elements);

        const size_t memory_size = ggml_v1_nbytes(model.memory_k) + ggml_v1_nbytes(model.memory_v);

        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;

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
                return ModelLoadResult::FAIL;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_v1_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return ModelLoadResult::FAIL;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return ModelLoadResult::FAIL;
            }

            const size_t bpe = (ftype == 0) ? sizeof(float) : sizeof(ggml_v1_fp16_t);

            if (nelements*bpe != ggml_v1_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_v1_nbytes(tensor), nelements*bpe);
                return ModelLoadResult::FAIL;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_v1_nbytes(tensor));

            //printf("%24s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_v1_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_v1_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return ModelLoadResult::SUCCESS;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt2_eval(
        const gpt2_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token,
              FileFormat file_format) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

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

    struct ggml_v1_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_v1_context * ctx0 = ggml_v1_init(params);
    struct ggml_v1_cgraph gf = { .n_threads = n_threads };

    struct ggml_v1_tensor * embd = ggml_v1_new_tensor_1d(ctx0, GGML_V1_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_v1_element_size(embd));

    struct ggml_v1_tensor * position = ggml_v1_new_tensor_1d(ctx0, GGML_V1_TYPE_I32, N);
    for (int i = 0; i < N; ++i) {
        ((int32_t *) position->data)[i] = n_past + i;
    }

    // wte + wpe
    struct ggml_v1_tensor * inpL =
        ggml_v1_add(ctx0,
                ggml_v1_get_rows(ctx0, model.wte, embd),
                ggml_v1_get_rows(ctx0, model.wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_v1_tensor * cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_v1_norm(ctx0, inpL);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_v1_add(ctx0,
                    ggml_v1_mul(ctx0,
                        ggml_v1_repeat(ctx0, model.layers[il].ln_1_g, cur),
                        cur),
                    ggml_v1_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_v1_mul_mat(ctx0,
                    ggml_v1_transpose(ctx0, model.layers[il].c_attn_attn_w),
                    cur);

            cur = ggml_v1_add(ctx0,
                    ggml_v1_repeat(ctx0, model.layers[il].c_attn_attn_b, cur),
                    cur);
        }

        // self-attention
        {
            struct ggml_v1_tensor * Qcur = ggml_v1_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_v1_tensor * Kcur = ggml_v1_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_v1_tensor * Vcur = ggml_v1_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_v1_tensor * k = ggml_v1_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_v1_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_v1_tensor * v = ggml_v1_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_v1_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_v1_build_forward_expand(&gf, ggml_v1_cpy(ctx0, Kcur, k));
                ggml_v1_build_forward_expand(&gf, ggml_v1_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_v1_tensor * Q =
                ggml_v1_permute(ctx0,
                        ggml_v1_cpy(ctx0,
                            Qcur,
                            ggml_v1_new_tensor_3d(ctx0, GGML_V1_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            struct ggml_v1_tensor * K =
                ggml_v1_permute(ctx0,
                        ggml_v1_reshape_3d(ctx0,
                            ggml_v1_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_v1_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_v1_tensor * V =
            //    ggml_v1_cpy(ctx0,
            //            ggml_v1_permute(ctx0,
            //                ggml_v1_reshape_3d(ctx0,
            //                    ggml_v1_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_v1_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_v1_new_tensor_3d(ctx0, GGML_V1_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_v1_tensor * KQV = ggml_v1_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            struct ggml_v1_tensor * KQ = ggml_v1_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            struct ggml_v1_tensor * KQ_scaled =
                ggml_v1_scale(ctx0,
                        KQ,
                        ggml_v1_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            struct ggml_v1_tensor * KQ_masked = ggml_v1_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            struct ggml_v1_tensor * KQ_soft_max = ggml_v1_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            struct ggml_v1_tensor * V_trans =
                ggml_v1_permute(ctx0,
                        ggml_v1_reshape_3d(ctx0,
                            ggml_v1_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_v1_element_size(model.memory_v)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_v1_tensor * KQV = ggml_v1_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_v1_tensor * KQV_merged = ggml_v1_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_v1_cpy(ctx0,
                    KQV_merged,
                    ggml_v1_new_tensor_2d(ctx0, GGML_V1_TYPE_F32, n_embd, N));
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_v1_mul_mat(ctx0,
                    ggml_v1_transpose(ctx0, model.layers[il].c_attn_proj_w),
                    cur);

            cur = ggml_v1_add(ctx0,
                    ggml_v1_repeat(ctx0, model.layers[il].c_attn_proj_b, cur),
                    cur);
        }

        // add the input
        cur = ggml_v1_add(ctx0, cur, inpL);

        struct ggml_v1_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_v1_norm(ctx0, inpFF);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_v1_add(ctx0,
                        ggml_v1_mul(ctx0,
                            ggml_v1_repeat(ctx0, model.layers[il].ln_2_g, cur),
                            cur),
                        ggml_v1_repeat(ctx0, model.layers[il].ln_2_b, cur));
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_v1_mul_mat(ctx0,
                    ggml_v1_transpose(ctx0, model.layers[il].c_mlp_fc_w),
                    cur);

            cur = ggml_v1_add(ctx0,
                    ggml_v1_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                    cur);

            // GELU activation
            // [3072, N]
            cur = ggml_v1_gelu(ctx0, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_v1_mul_mat(ctx0,
                    model.layers[il].c_mlp_proj_w_trans,
                    cur);

            cur = ggml_v1_add(ctx0,
                    ggml_v1_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                    cur);
        }

        // input for next layer
        inpL = ggml_v1_add(ctx0, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_v1_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_v1_add(ctx0,
                ggml_v1_mul(ctx0,
                    ggml_v1_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_v1_repeat(ctx0, model.ln_f_b, inpL));
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.wte
    // [ 768, N]     - inpL
    inpL = ggml_v1_mul_mat(ctx0, model.wte, inpL);

    // logits -> probs
    //inpL = ggml_v1_soft_max(ctx0, inpL);

    // run the computation
    ggml_v1_build_forward_expand(&gf, inpL);
    ggml_v1_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_v1_graph_print   (&gf);
    //    ggml_v1_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_v1_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_v1_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_v1_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_v1_used_mem(ctx0));

    ggml_v1_free(ctx0);

    return true;
}

// int main(int argc, char ** argv) {
//     ggml_v1_time_init();
//     const int64_t t_main_start_us = ggml_v1_time_us();

//     gpt_params params;
//     params.model = "models/gpt-2-117M/ggml-model.bin";

//     if (utils_gpt_params_parse(argc, argv, params) == false) {
//         return 1;
//     }

//     if (params.seed < 0) {
//         params.seed = time(NULL);
//     }

//     printf("%s: seed = %d\n", __func__, params.seed);

//     std::mt19937 rng(params.seed);
//     if (params.prompt.empty()) {
//         if( !isatty(STDIN_FILENO) ){
//             std::string line;
//             while( std::getline(std::cin, line) ){
//                 params.prompt = params.prompt + "\n" + line;
//             }
//         } else {
//             params.prompt = utils_gpt_random_prompt(rng);
//         }
//     }

//     int64_t t_load_us = 0;

//     gpt_vocab vocab;
//     gpt2_model model;

//     // load the model
//     {
//         const int64_t t_start_us = ggml_v1_time_us();

//         if (!gpt2_model_load(params.model, model, vocab, FileFormat::GPT2)) {
//             fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
//             return 1;
//         }

//         t_load_us = ggml_v1_time_us() - t_start_us;
//     }

//     int n_past = 0;

//     int64_t t_sample_us  = 0;
//     int64_t t_predict_us = 0;

//     std::vector<float> logits;

//     // tokenize the prompt
//     std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

//     params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

//     printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
//     printf("\n");

//     // submit the input prompt token-by-token
//     // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
//     std::vector<gpt_vocab::id> embd;

//     // determine the required inference memory per token:
//     size_t mem_per_token = 0;
//     gpt2_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token, FileFormat::GPT2);

//     for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
//         // predict
//         if (embd.size() > 0) {
//             const int64_t t_start_us = ggml_v1_time_us();

//             if (!gpt2_eval(model, params.n_threads, n_past, embd, logits, mem_per_token, FileFormat::GPT2)) {
//                 printf("Failed to predict\n");
//                 return 1;
//             }

//             t_predict_us += ggml_v1_time_us() - t_start_us;
//         }

//         n_past += embd.size();
//         embd.clear();

//         if (i >= embd_inp.size()) {
//             // sample next token
//             const int   top_k = params.top_k;
//             const float top_p = params.top_p;
//             const float temp  = params.temp;

//             const int n_vocab = model.hparams.n_vocab;

//             gpt_vocab::id id = 0;

//             {
//                 const int64_t t_start_sample_us = ggml_v1_time_us();

//                 id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

//                 t_sample_us += ggml_v1_time_us() - t_start_sample_us;
//             }

//             // add it to the context
//             embd.push_back(id);
//         } else {
//             // if here, it means we are still processing the input prompt
//             for (int k = i; k < embd_inp.size(); k++) {
//                 embd.push_back(embd_inp[k]);
//                 if (embd.size() >= params.n_batch) {
//                     break;
//                 }
//             }
//             i += embd.size() - 1;
//         }

//         // display text
//         for (auto id : embd) {
//             printf("%s", vocab.id_to_token[id].c_str());
//         }
//         fflush(stdout);

//         // end of text token
//         if (embd.back() == 50256) {
//             break;
//         }
//     }

//     // report timing
//     {
//         const int64_t t_main_end_us = ggml_v1_time_us();

//         printf("\n\n");
//         printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
//         printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
//         printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
//         printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
//         printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
//     }

//     ggml_v1_free(model.ctx);

//     return 0;
// }