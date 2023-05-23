#include "ggml_v2.h"
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

#include "model_adapter.h"



// load the model's weights from a file
ModelLoadResult gptj_v2_model_load(const std::string & fname, gptj_v2_model & model, gpt_vocab & vocab, int gpulayers) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

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
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_V2_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_V2_QNT_VERSION_FACTOR;
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
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            buf.resize(len);
            fin.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_v2_type wtype = ggml_v2_ftype_to_ggml_v2_type((ggml_v2_ftype) (model.hparams.ftype));
    if (wtype == GGML_V2_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return ModelLoadResult::FAIL;
    }

    auto & ctx = model.ctx;

    auto memory_type = GGML_V2_TYPE_F16;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*ggml_v2_type_sizef(GGML_V2_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_v2_type_sizef(GGML_V2_TYPE_F32); // ln_f_b

        ctx_size += n_embd*n_vocab*ggml_v2_type_sizef(wtype); // wte

        ctx_size += n_embd*n_vocab*ggml_v2_type_sizef(wtype);         // lmh_g
        ctx_size +=        n_vocab*ggml_v2_type_sizef(GGML_V2_TYPE_F32); // lmh_b

        ctx_size += n_layer*(n_embd*ggml_v2_type_sizef(GGML_V2_TYPE_F32)); // ln_1_g
        ctx_size += n_layer*(n_embd*ggml_v2_type_sizef(GGML_V2_TYPE_F32)); // ln_1_b

        ctx_size += n_layer*(n_embd*n_embd*ggml_v2_type_sizef(wtype)); // c_attn_q_proj_w
        ctx_size += n_layer*(n_embd*n_embd*ggml_v2_type_sizef(wtype)); // c_attn_k_proj_w
        ctx_size += n_layer*(n_embd*n_embd*ggml_v2_type_sizef(wtype)); // c_attn_v_proj_w

        ctx_size += n_layer*(n_embd*n_embd*ggml_v2_type_sizef(wtype)); // c_attn_proj_w

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_v2_type_sizef(wtype));         // c_mlp_fc_w
        ctx_size += n_layer*(       4*n_embd*ggml_v2_type_sizef(GGML_V2_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer*(4*n_embd*n_embd*ggml_v2_type_sizef(wtype));         // c_mlp_proj_w
        ctx_size += n_layer*(         n_embd*ggml_v2_type_sizef(GGML_V2_TYPE_F32)); // c_mlp_proj_b

        ctx_size += n_ctx*n_layer*n_embd*ggml_v2_type_sizef(memory_type); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_v2_type_sizef(memory_type); // memory_v

        ctx_size += (5 + 10*n_layer)*512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_v2_init_params params;
        params.mem_size   = ctx_size;
        params.mem_buffer = NULL;
        params.no_alloc   = false;
        

        model.ctx = ggml_v2_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_v2_init() failed\n", __func__);
            return ModelLoadResult::FAIL;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.wte    = ggml_v2_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        model.ln_f_g = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, n_embd);
        model.ln_f_b = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, n_embd);

        model.lmh_g  = ggml_v2_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        model.lmh_b  = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, n_vocab);

        // map by name
        model.tensors["transformer.wte.weight"] = model.wte;

        model.tensors["transformer.ln_f.weight"] = model.ln_f_g;
        model.tensors["transformer.ln_f.bias"]   = model.ln_f_b;

        model.tensors["lm_head.weight"] = model.lmh_g;
        model.tensors["lm_head.bias"]   = model.lmh_b;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g          = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32,   n_embd);
            layer.ln_1_b          = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32,   n_embd);

            layer.c_attn_q_proj_w = ggml_v2_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_k_proj_w = ggml_v2_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_v_proj_w = ggml_v2_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);

            layer.c_attn_proj_w   = ggml_v2_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);

            layer.c_mlp_fc_w      = ggml_v2_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b      = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w    = ggml_v2_new_tensor_2d(ctx, wtype,         4*n_embd,   n_embd);
            layer.c_mlp_proj_b    = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32,   n_embd);

            // map by name
            model.tensors["transformer.h." + std::to_string(i) + ".ln_1.weight"]          = layer.ln_1_g;
            model.tensors["transformer.h." + std::to_string(i) + ".ln_1.bias"]            = layer.ln_1_b;

            model.tensors["transformer.h." + std::to_string(i) + ".attn.q_proj.weight"]   = layer.c_attn_q_proj_w;
            model.tensors["transformer.h." + std::to_string(i) + ".attn.k_proj.weight"]   = layer.c_attn_k_proj_w;
            model.tensors["transformer.h." + std::to_string(i) + ".attn.v_proj.weight"]   = layer.c_attn_v_proj_w;

            model.tensors["transformer.h." + std::to_string(i) + ".attn.out_proj.weight"] = layer.c_attn_proj_w;

            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_in.weight"]     = layer.c_mlp_fc_w;
            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_in.bias"]       = layer.c_mlp_fc_b;

            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_out.weight"]    = layer.c_mlp_proj_w;
            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_out.bias"]      = layer.c_mlp_proj_b;
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

        model.memory_k = ggml_v2_new_tensor_1d(ctx, memory_type, n_elements);
        model.memory_v = ggml_v2_new_tensor_1d(ctx, memory_type, n_elements);

        const size_t memory_size = ggml_v2_nbytes(model.memory_k) + ggml_v2_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

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
            if (ggml_v2_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return ModelLoadResult::FAIL;
            }
          

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {

                //test for transposition and retry older loader
                if(tensor->ne[0]==ne[1] && tensor->ne[1]==ne[0] && should_transpose_layer(name))
                {
                    printf("\nFound a transposed tensor. This could be an older or newer model. Retrying load...");
                    ggml_v2_free(ctx);
                    return ModelLoadResult::RETRY_LOAD;
                }
                else
                {
                    fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                            __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                    return ModelLoadResult::FAIL;
                }
               
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ggml_v2_type_name(ggml_v2_type(ttype)), ggml_v2_nbytes(tensor)/1024.0/1024.0, ggml_v2_nbytes(tensor));
            }

            const size_t bpe = ggml_v2_type_size(ggml_v2_type(ttype));

            if ((nelements*bpe)/ggml_v2_blck_size(tensor->type) != ggml_v2_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_v2_nbytes(tensor), nelements*bpe);
                return ModelLoadResult::FAIL;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_v2_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ttype == 0 ? "float" : "f16", ggml_v2_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_v2_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

//         //gpu offload for gptj
// #if defined(GGML_USE_CLBLAST)
//     if(gpulayers>0)
//     {
//         const auto & hparams = model.hparams;
//         const int n_gpu = std::min(gpulayers, int(hparams.n_layer));
//         if(GetQuantsUnshuffled())
//         {

//         fprintf(stderr, "%s: [opencl] offloading %d layers to GPU\n", __func__, n_gpu);

//         size_t vram_total = 0;

//         for (int i = 0; i < n_gpu; ++i) {
//             const auto & layer = model.layers[i];

//             ggml_v2_cl_transform_tensor(layer.ln_1_g); vram_total += ggml_v2_nbytes(layer.ln_1_g);
//             ggml_v2_cl_transform_tensor(layer.ln_1_b); vram_total += ggml_v2_nbytes(layer.ln_1_b);
//             ggml_v2_cl_transform_tensor(layer.c_attn_q_proj_w); vram_total += ggml_v2_nbytes(layer.c_attn_q_proj_w);
//             ggml_v2_cl_transform_tensor(layer.c_attn_k_proj_w); vram_total += ggml_v2_nbytes(layer.c_attn_k_proj_w);
//             ggml_v2_cl_transform_tensor(layer.c_attn_v_proj_w); vram_total += ggml_v2_nbytes(layer.c_attn_v_proj_w);
//             ggml_v2_cl_transform_tensor(layer.c_attn_proj_w); vram_total += ggml_v2_nbytes(layer.c_attn_proj_w);
//             ggml_v2_cl_transform_tensor(layer.c_mlp_fc_w); vram_total += ggml_v2_nbytes(layer.c_mlp_fc_w);
//             ggml_v2_cl_transform_tensor(layer.c_mlp_fc_b); vram_total += ggml_v2_nbytes(layer.c_mlp_fc_b);
//             ggml_v2_cl_transform_tensor(layer.c_mlp_proj_w); vram_total += ggml_v2_nbytes(layer.c_mlp_proj_w);
//             ggml_v2_cl_transform_tensor(layer.c_mlp_proj_b); vram_total += ggml_v2_nbytes(layer.c_mlp_proj_b);
//         }

//         fprintf(stderr, "%s: [opencl] total VRAM used: %zu MB\n", __func__, vram_total / 1024 / 1024);
//         }
//         else
//         {
//             if(n_gpu>0)
//             {
//                 printf("\n[WARNING: Old format does not support GPU offloading! It will be deactivated!]\n");
//             }
//         }
//     }
// #endif


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
// The GPT-J model requires about 16MB of memory per input token.
//
bool gptj_v2_eval(
        const gptj_v2_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;

    static size_t buf_size = 256u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && (mem_per_token*N*2 + 64u*1024*1024) > buf_size) {
        const size_t buf_size_new = 320u*1024*1024 + 2*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        if (buf_size_new > buf_size)
        {
            buf_size = buf_size_new;
            buf = realloc(buf, buf_size);
            if (buf == nullptr)
            {
                fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
                return false;
            }
        }
    }

    struct ggml_v2_init_params params;
    params.mem_size   = buf_size;
    params.mem_buffer = buf;
    params.no_alloc   = false;
    

    struct ggml_v2_context * ctx0 = ggml_v2_init(params);
    struct ggml_v2_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_v2_tensor * embd = ggml_v2_new_tensor_1d(ctx0, GGML_V2_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_v2_element_size(embd));

    // wte
    struct ggml_v2_tensor * inpL = ggml_v2_get_rows(ctx0, model.wte, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_v2_tensor * cur;

        // norm
        {
            cur = ggml_v2_norm(ctx0, inpL);

            // cur = ln_1_g*cur + ln_1_b
            cur = ggml_v2_add(ctx0,
                    ggml_v2_mul(ctx0,
                        ggml_v2_repeat(ctx0, model.layers[il].ln_1_g, cur),
                        cur),
                    ggml_v2_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        struct ggml_v2_tensor * inpSA = cur;

        // self-attention
        {
            struct ggml_v2_tensor * Qcur = ggml_v2_rope_inplace(ctx0, ggml_v2_reshape_3d(ctx0, ggml_v2_mul_mat(ctx0, model.layers[il].c_attn_q_proj_w, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);
            struct ggml_v2_tensor * Kcur = ggml_v2_rope_inplace(ctx0, ggml_v2_reshape_3d(ctx0, ggml_v2_mul_mat(ctx0, model.layers[il].c_attn_k_proj_w, cur), n_embd/n_head, n_head, N), n_past, n_rot, 0);

            // store key and value to memory
            {
                struct ggml_v2_tensor * Vcur = ggml_v2_transpose(ctx0, ggml_v2_mul_mat(ctx0, model.layers[il].c_attn_v_proj_w, cur));

                struct ggml_v2_tensor * k = ggml_v2_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_v2_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_v2_tensor * v = ggml_v2_view_2d(ctx0, model.memory_v, N, n_embd,
                        (   n_ctx)*ggml_v2_element_size(model.memory_v),
                        (il*n_ctx)*ggml_v2_element_size(model.memory_v)*n_embd + n_past*ggml_v2_element_size(model.memory_v));

                ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(ctx0, Kcur, k));
                ggml_v2_build_forward_expand(&gf, ggml_v2_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_v2_tensor * Q =
                ggml_v2_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_v2_tensor * K =
                ggml_v2_permute(ctx0,
                        ggml_v2_reshape_3d(ctx0,
                            ggml_v2_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_v2_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_v2_tensor * KQ = ggml_v2_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_v2_tensor * KQ_scaled =
                ggml_v2_scale_inplace(ctx0,
                        KQ,
                        ggml_v2_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_v2_tensor * KQ_masked = ggml_v2_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_v2_tensor * KQ_soft_max = ggml_v2_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_v2_tensor * V =
                ggml_v2_view_3d(ctx0, model.memory_v,
                        n_past + N, n_embd/n_head, n_head,
                        n_ctx*ggml_v2_element_size(model.memory_v),
                        n_ctx*ggml_v2_element_size(model.memory_v)*n_embd/n_head,
                        il*n_ctx*ggml_v2_element_size(model.memory_v)*n_embd);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_v2_tensor * KQV = ggml_v2_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_v2_tensor * KQV_merged = ggml_v2_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_v2_cpy(ctx0,
                    KQV_merged,
                    ggml_v2_new_tensor_2d(ctx0, GGML_V2_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_v2_mul_mat(ctx0,
                    model.layers[il].c_attn_proj_w,
                    cur);
        }

        struct ggml_v2_tensor * inpFF = cur;

        // feed-forward network
        // this is independent of the self-attention result, so it could be done in parallel to the self-attention
        {
            // note here we pass inpSA instead of cur
            cur = ggml_v2_mul_mat(ctx0,
                    model.layers[il].c_mlp_fc_w,
                    inpSA);

            cur = ggml_v2_add(ctx0,
                    ggml_v2_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                    cur);

            // GELU activation
            cur = ggml_v2_gelu(ctx0, cur);

            // projection
            // cur = proj_w*cur + proj_b
            cur = ggml_v2_mul_mat(ctx0,
                    model.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_v2_add(ctx0,
                    ggml_v2_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                    cur);
        }

        // self-attention + FF
        cur  = ggml_v2_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = ggml_v2_add(ctx0, cur, inpL);
    }

    // norm
    {
        inpL = ggml_v2_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_v2_add(ctx0,
                ggml_v2_mul(ctx0,
                    ggml_v2_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_v2_repeat(ctx0, model.ln_f_b, inpL));
    }

    // lm_head
    {
        inpL = ggml_v2_mul_mat(ctx0, model.lmh_g, inpL);

        inpL = ggml_v2_add(ctx0,
                ggml_v2_repeat(ctx0, model.lmh_b, inpL),
                inpL);
    }

    // logits -> probs
    //inpL = ggml_v2_soft_max_inplace(ctx0, inpL);

    // run the computation
    ggml_v2_build_forward_expand(&gf, inpL);
    ggml_v2_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_v2_graph_print   (&gf);
    //    ggml_v2_graph_dump_dot(&gf, NULL, "gpt-j.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_v2_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_v2_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_v2_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_v2_used_mem(ctx0));

    ggml_v2_free(ctx0);

    return true;
}