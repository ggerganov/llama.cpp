#include "llama-context.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

static int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }
    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
    return relative_bucket;
}

enum ggml_status llama_context::compute_graph(
            ggml_cgraph * graph,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        set_threadpool_fn(backend_cpu, tp);
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}


llama_pos llama_context::pos_max() const {
    return kv_self.pos_max();
}

uint32_t llama_context::get_ctx_padding(const llama_cparams & cparams) const {
    return kv_self.get_padding(cparams);
}

// TODO: improve
void llama_context::reset() {
    inp_tokens          = nullptr;
    inp_embd            = nullptr;
    inp_pos             = nullptr;
    inp_out_ids         = nullptr;
    inp_mean            = nullptr;
    inp_cls             = nullptr;
    inp_embd_enc        = nullptr;
    inp_pos_bucket      = nullptr;
    inp_KQ_mask         = nullptr;
    inp_KQ_mask_cnv     = nullptr;
    inp_KQ_mask_swa     = nullptr;
    inp_KQ_mask_swa_cnv = nullptr;
    inp_KQ_mask_cross   = nullptr;
    inp_K_shift         = nullptr;
    inp_s_copy          = nullptr;
    inp_s_mask          = nullptr;
}

void llama_context::prepare_k_shift() {
}

void llama_context::prepare_defrag() {
}

void llama_context::prepare_decode(const llama_ubatch & /*ubatch*/) {
}

// llama input

void llama_context::set_inputs(const llama_ubatch & ubatch) {
    const llama_hparams & hparams = model.hparams;

    //
    // set input data
    //

    if (inp_K_shift) {
        assert(ggml_backend_buffer_is_host(inp_K_shift->buffer));

        int32_t * data = (int32_t *) inp_K_shift->data;

        for (uint32_t i = 0; i < kv_self.size; ++i) {
            data[i] = kv_self.cells[i].delta;
        }

        // the K-shift graph requires just this input
        return;
    }

    if (ubatch.token) {
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_set(inp_tokens, ubatch.token, 0, n_tokens*ggml_element_size(inp_tokens));
    }

    if (ubatch.embd) {
        const int64_t n_embd   = hparams.n_embd;
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_set(inp_embd, ubatch.embd, 0, n_tokens*n_embd*ggml_element_size(inp_embd));
    }

    if (ubatch.pos && inp_pos) {
        const int64_t n_tokens = ubatch.n_tokens;
        auto n_pos = n_pos_per_token;
        ggml_backend_tensor_set(inp_pos, ubatch.pos, 0, n_tokens*n_pos*ggml_element_size(inp_pos));
    }

    if (hparams.causal_attn || cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        //GGML_ASSERT(inp_out_ids && "every model that can must skip unused outputs");

        if (!inp_out_ids) {
            LLAMA_LOG_WARN("%s: 'inp_out_ids' is not created\n", __func__);
        } else {
            const int64_t n_tokens = ubatch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_out_ids->buffer));
            int32_t * data = (int32_t *) inp_out_ids->data;

            if (n_outputs == n_tokens) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[i] = i;
                }
            } else if (ubatch.output) {
                int32_t n_outputs = 0;
                for (int i = 0; i < n_tokens; ++i) {
                    if (ubatch.output[i]) {
                        data[n_outputs++] = i;
                    }
                }
                // the graph needs to have been passed the correct number of outputs
                GGML_ASSERT(n_outputs == n_outputs);
            } else if (n_outputs == 1) {
                // only keep last output
                data[0] = n_tokens - 1;
            } else {
                GGML_ASSERT(n_outputs == 0);
            }
        }
    }

    GGML_ASSERT(
            // (!a || b) is a logical implication (a -> b)
            // !hparams.causal_attn -> !cparams.causal_attn
            (hparams.causal_attn || !cparams.causal_attn) &&
            "causal attention is not supported by this model"
            );

    if (inp_KQ_mask || inp_KQ_mask_swa) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn && !is_encoding) {
            const int64_t n_kv         = kv_self.n;
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;

            float * data     = nullptr;
            float * data_swa = nullptr;

            if (inp_KQ_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask->buffer));
                data = (float *) inp_KQ_mask->data;
            }

            if (inp_KQ_mask_swa) {
                GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask_swa->buffer));
                data_swa = (float *) inp_KQ_mask_swa->data;
            }

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the ubatch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch.pos[s*n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float f;
                            if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }
                        }
                    }
                }

                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_swa) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_swa[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }
            }
        } else {
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;
            // when using kv cache, the mask needs to match the kv cache size
            const int64_t n_stride = hparams.causal_attn && !is_encoding ? kv_self.n : n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask->buffer));

            float * data = (float *) inp_KQ_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int s1 = 0; s1 < n_seqs; ++s1) {
                    const llama_seq_id seq_id = ubatch.seq_id[s1][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const int32_t tj = s1*n_seq_tokens + j;

                        for (int s0 = 0; s0 < n_seqs; ++s0) {
                            for (int i = 0; i < n_seq_tokens; ++i) {
                                const int32_t ti = s0*n_seq_tokens + i;
                                float f = -INFINITY;

                                for (int s = 0; s < ubatch.n_seq_id[s0]; ++s) {
                                    if (ubatch.seq_id[s0][s] == seq_id) {
                                        if (hparams.use_alibi) {
                                            f = -std::abs(ubatch.pos[ti] - ubatch.pos[tj]);
                                        } else {
                                            f = 0.0f;
                                        }
                                        break;
                                    }
                                }

                                data[h*(n_tokens*n_tokens) + tj*n_stride + ti] = f;
                            }
                        }

                        for (int i = n_tokens; i < n_stride; ++i) {
                            data[h*(n_tokens*n_tokens) + tj*n_stride + i] = -INFINITY;
                        }
                    }
                }
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(inp_mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_mean->buffer));

        float * data = (float *) inp_mean->data;
        memset(inp_mean->data, 0, n_tokens * n_tokens * ggml_element_size(inp_mean));

        std::vector<uint64_t> sum(n_tokens, 0);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += ubatch.n_seq_tokens;
        }

        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            for (int i = 0; i < n_seq_tokens; ++i) {
                data[seq_id*n_tokens + s*n_seq_tokens + i] = div[seq_id];
            }
        }
    }

    if (cparams.embeddings && (
                cparams.pooling_type == LLAMA_POOLING_TYPE_CLS ||
                cparams.pooling_type == LLAMA_POOLING_TYPE_RANK)) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_cls->buffer));

        uint32_t * data = (uint32_t *) inp_cls->data;
        memset(inp_cls->data, 0, n_tokens * ggml_element_size(inp_cls));

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == CLS or RANK");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch.pos[s*n_seq_tokens + i];

                if (pos == 0) {
                    data[seq_id] = s*n_seq_tokens + i;
                }
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_cls->buffer));

        uint32_t * data = (uint32_t *) inp_cls->data;
        memset(inp_cls->data, 0, n_tokens * ggml_element_size(inp_cls));

        std::vector<int> last_pos(n_tokens, -1);
        std::vector<int> last_row(n_tokens, -1);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == LAST");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch.pos[s*n_seq_tokens + i];

                if (pos >= last_pos[seq_id]) {
                    last_pos[seq_id] = pos;
                    last_row[seq_id] = s*n_seq_tokens + i;
                }
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            if (last_row[i] >= 0) {
                data[i] = last_row[i];
            }
        }
    }

    if (kv_self.recurrent) {
        const int64_t n_kv = kv_self.n;

        if (inp_s_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(inp_s_mask->buffer));
            float * data = (float *) inp_s_mask->data;

            // clear unused states
            for (int i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = kv_self.cells[cell_id];

                data[i] = (float) (kv_cell.src >= 0);

                // TODO: do not mutate the KV cache
                // only clear once
                if (kv_cell.src < 0) {
                    kv_cell.src = cell_id;
                }
            }
        }

        if (inp_s_copy) {
            GGML_ASSERT(ggml_backend_buffer_is_host(inp_s_copy->buffer));
            int32_t * data = (int32_t *) inp_s_copy->data;

            // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
            for (uint32_t i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = kv_self.cells[cell_id];

                // prevent out-of-bound sources
                if (kv_cell.src < 0 || (uint32_t) kv_cell.src >= kv_self.size) {
                    kv_cell.src = cell_id;
                }

                data[i] = kv_cell.src;

                // TODO: do not mutate the KV cache
                // ensure copy only happens once
                if (kv_cell.src != (int32_t) cell_id) {
                    kv_cell.src = cell_id;
                }
            }
        }
    }

    if (inp_pos_bucket) {
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(inp_pos_bucket->buffer));
        GGML_ASSERT(!ubatch.equal_seqs); // TODO: use ubatch.n_seqs instead of failing

        int32_t * data = (int32_t *) inp_pos_bucket->data;

        if (!is_encoding) {
            const int64_t n_kv = kv_self.n;
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_kv; ++i) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(kv_self.cells[i].pos, ubatch.pos[j], hparams.n_rel_attn_bkts, is_encoding);
                    }
                }
            }
        } else {
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_tokens; ++i) {
                        data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(ubatch.pos[i], ubatch.pos[j], hparams.n_rel_attn_bkts, is_encoding);
                    }
                }
            }
        }
    }

    if (!is_encoding && inp_embd_enc) {
        assert(inp_embd_enc->type == GGML_TYPE_F32);
        assert((size_t) ggml_nelements(inp_embd_enc) == embd_enc.size());

        ggml_backend_tensor_set(inp_embd_enc, embd_enc.data(), 0, ggml_nbytes(inp_embd_enc));
    }

    if (!is_encoding && inp_KQ_mask_cross) {
        const int64_t n_output_enc = embd_enc.size() / hparams.n_embd;
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(inp_KQ_mask_cross->buffer));
        GGML_ASSERT(!ubatch.equal_seqs); // TODO: use ubatch.n_seqs instead of failing

        float * data = (float *) inp_KQ_mask_cross->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_output_enc; ++i) {
                    float f = -INFINITY;
                    for (int s = 0; s < ubatch.n_seq_id[j]; ++s) {
                        const llama_seq_id seq_id = ubatch.seq_id[j][s];
                        if (seq_ids_enc[i].find(seq_id) != seq_ids_enc[i].end()) {
                            f = 0.0f;
                        }
                    }
                    data[h*(n_output_enc*n_tokens) + j*n_output_enc + i] = f;
                }
            }

            for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (int j = 0; j < n_output_enc; ++j) {
                    data[h*(n_output_enc*n_tokens) + i*n_output_enc + j] = -INFINITY;
                }
            }
        }
    }
}

// do mat_mul, while optionally apply lora
ggml_tensor * llama_context::build_lora_mm(
        ggml_context * ctx0,
         ggml_tensor * w,
         ggml_tensor * cur) {
    struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : loras) {
        struct llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        struct ggml_tensor * ab_cur = ggml_mul_mat(
            ctx0, lw->b,
            ggml_mul_mat(ctx0, lw->a, cur)
        );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

// do mat_mul_id, while optionally apply lora
ggml_tensor * llama_context::build_lora_mm_id(
        ggml_context * ctx0,
         ggml_tensor * w,
         ggml_tensor * cur,
         ggml_tensor * ids) {
    struct ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : loras) {
        struct llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        struct ggml_tensor * ab_cur = ggml_mul_mat_id(
            ctx0, lw->b,
            ggml_mul_mat_id(ctx0, lw->a, cur, ids),
            ids
        );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

void llama_context::kv_self_update() {
    auto & kv = kv_self;

    if (kv.has_shift) {
        if (!kv.can_shift) {
            GGML_ABORT("The current context does not support K-shift");
        }

        // apply K-shift if needed
        if (model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            prepare_k_shift();

            ggml_backend_sched_reset(sched.get());

            struct ggml_init_params params = {
                /*.mem_size   =*/ buf_compute_meta.size(),
                /*.mem_buffer =*/ buf_compute_meta.data(),
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx0 = ggml_init(params);

            reset();

            ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

            build_k_shift(ctx0, gf);

            ggml_backend_sched_alloc_graph(sched.get(), gf);

            set_inputs({});

            compute_graph(gf, false);

            ggml_free(ctx0);

            need_reserve = true;
        }

        {
            kv.has_shift = false;

            for (uint32_t i = 0; i < kv.size; ++i) {
                kv.cells[i].delta = 0;
            }
        }
    }

    // defragment the KV cache if needed
    if (kv.do_defrag) {
        prepare_defrag();

        ggml_backend_sched_reset(sched.get());

        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ggml_context * ctx0 = ggml_init(params);

        reset();

        ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        build_defrag(ctx0, gf);

        ggml_backend_sched_alloc_graph(sched.get(), gf);

        // no input
        //set_inputs({});

        compute_graph(gf, false);

        ggml_free(ctx0);

        kv.do_defrag = false;

        need_reserve = true;
    }
}

void llama_kv_self_update(llama_context * ctx) {
    ctx->kv_self_update();
}

void llama_context::build_attn_inp(
        ggml_context * ctx0,
             int32_t   n_tokens,
                bool   causal,
                bool   swa,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto n_kv = worst_case ? kv_self.size : kv_self.n;

    inp_KQ_mask = causal
        ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    //cb(inp_KQ_mask, "KQ_mask", -1);
    ggml_set_input(inp_KQ_mask);

    inp_KQ_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_KQ_mask, GGML_TYPE_F16) : inp_KQ_mask;

    if (swa) {
        GGML_ASSERT(hparams.n_swa > 0);

        inp_KQ_mask_swa = causal
            ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
            : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        //cb(inp_KQ_mask_swa, "KQ_mask_swa", -1);
        ggml_set_input(inp_KQ_mask_swa);

        inp_KQ_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_KQ_mask_swa, GGML_TYPE_F16) : inp_KQ_mask_swa;
    }
}

void llama_context::build_attn_kv_store(
        ggml_context * ctx0,
         ggml_cgraph * graph,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
             int32_t   n_tokens,
             int64_t   il,
             bool      worst_case) {
    const auto & hparams = model.hparams;

    const auto & n_ctx = cparams.n_ctx;

    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    GGML_ASSERT(kv_self.size == n_ctx);

    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx0, kv_self.k_l[il], n_tokens*n_embd_k_gqa, ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa)*kv_head);
    //cb(k_cache_view, "k_cache_view", il);

    // note: storing RoPE-ed version of K in the KV cache
    ggml_build_forward_expand(graph, ggml_cpy(ctx0, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_v_gqa && v_cur->ne[1] == n_tokens);

    struct ggml_tensor * v_cache_view = nullptr;

    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(ctx0, kv_self.v_l[il], n_tokens*n_embd_v_gqa, ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa)*kv_head);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_cache_view = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, n_embd_v_gqa,
                (  n_ctx)*ggml_element_size(kv_self.v_l[il]),
                (kv_head)*ggml_element_size(kv_self.v_l[il]));

        v_cur = ggml_transpose(ctx0, v_cur);
    }
    //cb(v_cache_view, "v_cache_view", il);

    ggml_build_forward_expand(graph, ggml_cpy(ctx0, v_cur, v_cache_view));
}

ggml_tensor * llama_context::build_attn_qkv(
        ggml_context * ctx0,
         ggml_cgraph * graph,
         ggml_tensor * wo,
         ggml_tensor * wo_b,
         ggml_tensor * q_cur,
             int32_t   n_tokens,
             float     kq_scale,
             int       il,
             bool      worst_case) {
    const auto & hparams = model.hparams;

    const auto & n_ctx         = cparams.n_ctx;
    const auto & n_embd_head_k = hparams.n_embd_head_k;
    const auto & n_embd_head_v = hparams.n_embd_head_v;

    // TODO: improve
    bool is_sliding = false;

    switch (model.arch) {
        case LLM_ARCH_COHERE2:
            {
                const int32_t sliding_window_pattern = 4;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_GEMMA2:
            {
                const int32_t sliding_window_pattern = 2;
                is_sliding = il % sliding_window_pattern < (sliding_window_pattern - 1);
            } break;
        case LLM_ARCH_PHI3:
            {
                is_sliding = hparams.n_swa > 0;
            } break;
        default:
            {
                is_sliding = false;
            }
    };

    const auto & kq_mask = is_sliding ? inp_KQ_mask_swa_cnv : inp_KQ_mask_cnv;

    const auto n_kv = worst_case ? kv_self.size : kv_self.n;

    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    struct ggml_tensor * k =
        ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                0);
    //cb(k, "k", il);

    struct ggml_tensor * cur;

    if (cparams.flash_attn) {
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);

        // split cached v into n_head heads (not transposed)
        struct ggml_tensor * v =
            ggml_view_3d(ctx0, kv_self.v_l[il],
                    n_embd_head_v, n_kv, n_head_kv,
                    ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                    ggml_row_size(kv_self.v_l[il]->type, n_embd_head_v),
                    0);
        //cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, n_embd_head_v*n_head, n_tokens);
    } else {
        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        //cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (model.arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, 0.08838834764831845f/30.0f));
            kq = ggml_scale(ctx0, kq, 30);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh(ctx0, kq);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        //cb(kq, "kq_soft_max_ext", il);

        GGML_ASSERT(kv_self.size == n_ctx);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx0, kv_self.v_l[il],
                    n_kv, n_embd_head_v, n_head_kv,
                    ggml_element_size(kv_self.v_l[il])*n_ctx,
                    ggml_element_size(kv_self.v_l[il])*n_ctx*n_embd_head_v,
                    0);
        //cb(v, "v", il);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        //cb(kqv, "kqv", il);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        //cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_head_v*n_head, n_tokens);
        //cb(cur, "kqv_merged_cont", il);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(graph, cur);

    if (wo) {
        cur = build_lora_mm(ctx0, wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llama_context::build_soft_max_ext(
        ggml_context * ctx0,
         ggml_tensor * kq,
             float     kq_scale) {
    const auto & hparams = model.hparams;

    return ggml_soft_max_ext(ctx0, kq, inp_KQ_mask_cnv, kq_scale, hparams.f_max_alibi_bias);
}

ggml_tensor * llama_context::get_rope_factors(int il) {
    const auto & hparams = model.hparams;

    // choose long/short freq factors based on the context size
    const auto n_ctx_pre_seq = cparams.n_ctx / cparams.n_seq_max;

    if (model.layers[il].rope_freqs != nullptr) {
        return model.layers[il].rope_freqs;
    }

    if (n_ctx_pre_seq > hparams.n_ctx_orig_yarn) {
        return model.layers[il].rope_long;
    }

    return model.layers[il].rope_short;
}

void llama_context::build_k_shift(
        ggml_context * ctx0,
         ggml_cgraph * graph) {
    const auto & n_ctx      = cparams.n_ctx;
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;
    const auto & freq_base  = cparams.rope_freq_base;
    const auto & freq_scale = cparams.rope_freq_scale;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;

    const auto & hparams = model.hparams;

    const auto & n_rot     = hparams.n_rot;
    const auto & n_layer   = hparams.n_layer;
    const auto & rope_type = hparams.rope_type;

    const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    GGML_ASSERT(kv_self.size == n_ctx);

    inp_K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
    //cb(inp_K_shift, "K_shift", -1);
    ggml_set_input(inp_K_shift);

    for (uint32_t il = 0; il < n_layer; ++il) {
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        struct ggml_tensor * rope_factors = get_rope_factors(il);

        struct ggml_tensor * k =
            ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head_k, n_head_kv, n_ctx,
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                0);

        struct ggml_tensor * tmp;
        if (ggml_is_quantized(k->type)) {
            // dequantize to f32 -> RoPE -> quantize back
            tmp = ggml_cast(ctx0, k, GGML_TYPE_F32);
            //cb(tmp, "K_f32", il);

            for (auto & backend : backends) {
                // Figure out which backend KV cache belongs to
                if (ggml_backend_supports_buft(backend.get(), ggml_backend_buffer_get_type(kv_self.k_l[il]->buffer))) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), tmp, backend.get());
                    break;
                }
            }
            tmp = ggml_rope_ext_inplace(ctx0, tmp,
                    inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
            //cb(tmp, "K_shifted_f32", il);

            tmp = ggml_cpy(ctx0, tmp, k);
        } else {
            // we rotate only the first n_rot dimensions
            tmp = ggml_rope_ext_inplace(ctx0, k,
                    inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
        }
        //cb(tmp, "K_shifted", il);

        ggml_build_forward_expand(graph, tmp);
    }
}

void llama_context::build_defrag(
        ggml_context * ctx0,
         ggml_cgraph * graph) {
    const auto & hparams = model.hparams;

    const uint32_t n_layer = hparams.n_layer;

    const uint32_t n_kv   = kv_self.cell_max();
    const uint32_t n_used = kv_self.used;

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // each move requires 6*n_layer tensors (see build_defrag)
    //   - source view, destination view, copy operation
    //   - x2 for keys and values
    //const uint32_t max_moves = model.max_nodes()/(6*n_layer);
    // TODO: tmp fix https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (model.max_nodes() - 2*n_layer)/(6*n_layer);

    // determine which KV cells to move where
    //
    //  cell i moves to ids[i]
    //
    //  if ids[i] == i || ids[i] == n_kv, then cell i is not moved
    //
    std::vector<uint32_t> ids(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = kv_self.cells[i0];

        if (!cell0.is_empty()) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && kv_self.cells[i0 + nh].is_empty()) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            const auto & cell1 = kv_self.cells[is];

            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // should we stop searching for the next move?
        bool stop = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = kv_self.cells[i1];

            if (cell1.is_empty() || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            // move the cell meta data
            kv_self.cells[i0 + nf] = cell1;

            // clear the old cell and move the head there
            cell1 = llama_kv_cell();
            kv_self.head = n_used;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return;
    }

    //LLAMA_LOG_INFO("(tmp log) KV defrag cell moves: %u\n", n_moves);

    //LLAMA_LOG_INFO("expected gf nodes: %u\n", 6*n_moves*n_layer);

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = kv_self.size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
        const size_t v_size    = ggml_row_size (kv_self.v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(kv_self.k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(kv_self.v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(kv_self.k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(kv_self.v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        for (uint32_t il = 0; il < n_layer; ++il) {
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self.k_l[il],
                    n_embd_k_gqa, nm,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self.k_l[il],
                    n_embd_k_gqa, nm,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src;
            ggml_tensor * view_v_dst;

            if (cparams.flash_attn) {
                // NOTE: the V cache is not transposed when using flash attention
                view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il],
                        n_embd_v_gqa, nm,
                        ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                        ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa*i));

                view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il],
                        n_embd_v_gqa, nm,
                        ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                        ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa*id));
            } else {
                view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il],
                        nm, n_embd_v_gqa,
                        ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                        ggml_row_size(kv_self.v_l[il]->type, i));

                view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il],
                        nm, n_embd_v_gqa,
                        ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                        ggml_row_size(kv_self.v_l[il]->type, id));
            }

            ggml_build_forward_expand(graph, ggml_cpy(ctx0, view_k_src, view_k_dst));
            ggml_build_forward_expand(graph, ggml_cpy(ctx0, view_v_src, view_v_dst));
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("graph->n_nodes = %d\n", graph->n_nodes);
#endif
}

ggml_tensor * llama_context::build_inp_s_copy(
        ggml_context * ctx0,
                bool   worst_case) {
    const auto n_kv    = worst_case ? kv_self.size : kv_self.n;

    inp_s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
    //cb(inp_s_copy, "inp_s_copy", -1);
    ggml_set_input(inp_s_copy);
    return inp_s_copy;
}

ggml_tensor * llama_context::build_inp_s_mask(
        ggml_context * ctx0,
                bool   worst_case) {
    const auto n_kv    = worst_case ? kv_self.size : kv_self.n;
    inp_s_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_kv);
    //cb(inp_s_mask, "inp_s_mask", -1);
    ggml_set_input(inp_s_mask);
    return inp_s_mask;
}

ggml_tensor * llama_context::build_copy_mask_state(
        ggml_context * ctx0,
         ggml_cgraph * graph,
         ggml_tensor * s,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
             int32_t   n_tokens,
             int32_t   n_state,
             int32_t   n_seqs,
                bool   worst_case) {
    const auto n_kv    = worst_case ? kv_self.size : kv_self.n;
    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    struct ggml_tensor * states = ggml_reshape_2d(ctx0, s, n_state, kv_self.size);

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between kv_head and kv_head + n_kv
    // this shrinks the tensors's ne[1] to n_kv
    states = ggml_get_rows(ctx0, states, state_copy);

    // clear states of sequences which are starting at the beginning of this batch
    // FIXME: zero-out NANs?
    states = ggml_mul(ctx0, states, state_mask);

    // copy states which won't be changed further (between n_seqs and n_kv)
    ggml_build_forward_expand(graph,
        ggml_cpy(ctx0,
            ggml_view_1d(ctx0, states, n_state*(n_kv - n_seqs), (n_seqs          )*n_state*ggml_element_size(states)),
            ggml_view_1d(ctx0, s,      n_state*(n_kv - n_seqs), (kv_head + n_seqs)*n_state*ggml_element_size(s))));

    // the part of the states that will be used and modified
    return ggml_view_2d(ctx0, states, n_state, n_seqs, states->nb[1], 0);
}

// TODO: split
ggml_tensor * llama_context::build_mamba_layer(
        ggml_context * ctx0,
         ggml_cgraph * graph,
         ggml_tensor * cur,
         ggml_tensor * state_copy,
         ggml_tensor * state_mask,
  const llama_ubatch & ubatch,
                 int   il,
                bool   worst_case) {
    const auto & hparams = model.hparams;

    const auto & n_tokens = ubatch.n_tokens;

    const auto kv_head = worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head;

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs  = ubatch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;
    // Use the same RMS norm as the final layer norm
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs);
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    struct ggml_tensor * conv_states_all = kv_self.k_l[il];
    struct ggml_tensor * ssm_states_all  = kv_self.v_l[il];

    // (ab)using the KV cache to store the states
    struct ggml_tensor * conv = build_copy_mask_state(
            ctx0, graph, conv_states_all, state_copy, state_mask,
            n_tokens, hparams.n_embd_k_s(), n_seqs, worst_case);
    conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);
    struct ggml_tensor * ssm = build_copy_mask_state(
            ctx0, graph, ssm_states_all, state_copy, state_mask,
            n_tokens, hparams.n_embd_v_s(), n_seqs, worst_case);
    ssm = ggml_reshape_3d(ctx0, ssm, d_state, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor * xz = build_lora_mm(ctx0, model.layers[il].ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor * x = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    struct ggml_tensor * z = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        struct ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        struct ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

        ggml_build_forward_expand(graph,
            ggml_cpy(ctx0, last_conv,
                ggml_view_1d(ctx0, conv_states_all,
                    (d_conv - 1)*(d_inner)*(n_seqs),
                    kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        x = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

        // bias
        x = ggml_add(ctx0, x, model.layers[il].ssm_conv1d_b);

        x = ggml_silu(ctx0, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        struct ggml_tensor * x_db = build_lora_mm(ctx0, model.layers[il].ssm_x, x);
        // split
        struct ggml_tensor * dt = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
        struct ggml_tensor * B  = ggml_view_3d(ctx0, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*dt_rank);
        struct ggml_tensor * C  = ggml_view_3d(ctx0, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*(dt_rank+d_state));

        // Some Mamba variants (e.g. FalconMamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms) {
            dt = ggml_rms_norm(ctx0, dt, norm_rms_eps);
            B = ggml_rms_norm(ctx0, B, norm_rms_eps);
            C = ggml_rms_norm(ctx0, C, norm_rms_eps);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = build_lora_mm(ctx0, model.layers[il].ssm_dt, dt);
        dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);

        // Custom operator to optimize the parallel associative scan
        // as described in the Annex D of the Mamba paper.
        // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
        struct ggml_tensor * y_ssm = ggml_ssm_scan(ctx0, ssm, x, dt, model.layers[il].ssm_a, B, C);

        // store last states
        ggml_build_forward_expand(graph,
            ggml_cpy(ctx0,
                ggml_view_1d(ctx0, y_ssm, d_state*d_inner*n_seqs, x->nb[3]),
                ggml_view_1d(ctx0, ssm_states_all, d_state*d_inner*n_seqs, kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

        struct ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[1], x->nb[2], 0);

        // TODO: skip computing output earlier for unused tokens

        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
        y = ggml_mul(ctx0, y, ggml_silu(ctx0, ggml_cont(ctx0, z)));

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(ctx0, model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    //cb(cur, "mamba_out", il);

    return cur;
}


// llama output

size_t llama_output_reserve(struct llama_context & lctx, size_t n_outputs) {
    const auto & cparams = lctx.cparams;
    const auto & hparams = lctx.model.hparams;
    const auto & vocab   = lctx.model.vocab;

    const size_t n_outputs_max = std::max(n_outputs, (size_t) cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = vocab.n_tokens();
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool has_logits = !cparams.embeddings;
    const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    const size_t logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    const size_t embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (lctx.output_ids.empty()) {
        // init, never resized afterwards
        lctx.output_ids.resize(n_batch);
    }

    const size_t prev_size = lctx.buf_output ? ggml_backend_buffer_get_size(lctx.buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!lctx.buf_output || prev_size < new_size) {
        if (lctx.buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            lctx.buf_output = nullptr;
            lctx.logits = nullptr;
            lctx.embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = lctx.model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        lctx.buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (lctx.buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(lctx.buf_output.get());

    lctx.logits = has_logits ? output_base               : nullptr;
    lctx.embd   = has_embd   ? output_base + logits_size : nullptr;

    lctx.output_size = n_outputs_max;
    lctx.logits_size = logits_size;
    lctx.embd_size   = embd_size;

    // set all ids as invalid (negative)
    std::fill(lctx.output_ids.begin(), lctx.output_ids.end(), -1);

    ggml_backend_buffer_clear(lctx.buf_output.get(), 0);

    lctx.n_outputs = 0;

    return n_outputs_max;
}

void llama_output_reorder(struct llama_context & ctx) {
    std::vector<size_t> & out_ids = ctx.sbatch.out_ids;
    if (!out_ids.empty()) {
        const uint32_t n_vocab = ctx.model.vocab.n_tokens();
        const uint32_t n_embd  = ctx.model.hparams.n_embd;

        const int32_t n_outputs = ctx.n_outputs;
        GGML_ASSERT((size_t) n_outputs == out_ids.size());

        // TODO: is there something more efficient which also minimizes swaps?
        // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
        for (int32_t i = 0; i < n_outputs - 1; ++i) {
            int32_t j_min = i;
            for (int32_t j = i + 1; j < n_outputs; ++j) {
                if (out_ids[j] < out_ids[j_min]) {
                    j_min = j;
                }
            }
            if (j_min == i) { continue; }
            std::swap(out_ids[i], out_ids[j_min]);
            if (ctx.logits_size > 0) {
                for (uint32_t k = 0; k < n_vocab; k++) {
                    std::swap(ctx.logits[i*n_vocab + k], ctx.logits[j_min*n_vocab + k]);
                }
            }
            if (ctx.embd_size > 0) {
                for (uint32_t k = 0; k < n_embd; k++) {
                    std::swap(ctx.embd[i*n_embd + k], ctx.embd[j_min*n_embd + k]);
                }
            }
        }
        std::fill(ctx.output_ids.begin(), ctx.output_ids.end(), -1);
        for (int32_t i = 0; i < n_outputs; ++i) {
            ctx.output_ids[out_ids[i]] = i;
        }
        out_ids.clear();
    }
}

//
// interface implementation
//

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const struct llama_context * ctx) {
    return ctx->cparams.n_ctx;
}

uint32_t llama_n_batch(const struct llama_context * ctx) {
    return ctx->cparams.n_batch;
}

uint32_t llama_n_ubatch(const struct llama_context * ctx) {
    return ctx->cparams.n_ubatch;
}

uint32_t llama_n_seq_max(const struct llama_context * ctx) {
    // TODO: add notion of n_seq_max to llama_kv_cache and use it here
    return ctx->kv_self.size;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->model;
}

llama_kv_cache * llama_get_kv_self(llama_context * ctx) {
    return &ctx->kv_self;
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->cparams.pooling_type;
}

void llama_attach_threadpool(
             struct llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->threadpool       = threadpool;
    ctx->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_detach_threadpool(struct llama_context * ctx) {
    ctx->threadpool       = nullptr;
    ctx->threadpool_batch = nullptr;
}

void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->cparams.n_threads       = n_threads;
    ctx->cparams.n_threads_batch = n_threads_batch;
}

int32_t llama_n_threads(struct llama_context * ctx) {
    return ctx->cparams.n_threads;
}

int32_t llama_n_threads_batch(struct llama_context * ctx) {
    return ctx->cparams.n_threads_batch;
}

void llama_set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->abort_callback      = abort_callback;
    ctx->abort_callback_data = abort_callback_data;

    for (auto & backend : ctx->backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), ctx->abort_callback, ctx->abort_callback_data);
        }
    }
}

void llama_set_embeddings(struct llama_context * ctx, bool embeddings) {
    ctx->cparams.embeddings = embeddings;
}

void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn) {
    ctx->cparams.causal_attn = causal_attn;
}

void llama_synchronize(struct llama_context * ctx) {
    ggml_backend_sched_synchronize(ctx->sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (ctx->n_queued_tokens == 1) {
        if (!ctx->cparams.no_perf) {
            ctx->t_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        }
        ctx->n_eval++;
    } else if (ctx->n_queued_tokens > 1) {
        if (!ctx->cparams.no_perf) {
            ctx->t_p_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        }
        ctx->n_p_eval += ctx->n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (ctx->n_queued_tokens > 0 && !ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    ctx->n_queued_tokens = 0;
    ctx->t_compute_start_us = 0;
}

float * llama_get_logits(struct llama_context * ctx) {
    llama_synchronize(ctx);

    // reorder logits for backward compatibility
    // TODO: maybe deprecate this
    llama_output_reorder(*ctx);

    return ctx->logits;
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    int32_t j = -1;

    llama_synchronize(ctx);

    try {
        if (ctx->logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {
            j = ctx->n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", ctx->n_outputs));
            }
        } else if ((size_t) i >= ctx->output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", ctx->output_ids.size()));
        } else {
            j = ctx->output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= ctx->n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
        }

        return ctx->logits + j*ctx->model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_get_embeddings(struct llama_context * ctx) {
    llama_synchronize(ctx);

    // reorder embeddings for backward compatibility
    // TODO: maybe deprecate this
    llama_output_reorder(*ctx);

    return ctx->embd;
}

float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i) {
    int32_t j = -1;

    llama_synchronize(ctx);

    try {
        if (ctx->embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        if (i < 0) {
            j = ctx->n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", ctx->n_outputs));
            }
        } else if ((size_t) i >= ctx->output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", ctx->output_ids.size()));
        } else {
            j = ctx->output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= ctx->n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
        }

        return ctx->embd + j*ctx->model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_synchronize(ctx);

    auto it = ctx->embd_seq.find(seq_id);
    if (it == ctx->embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

// llama adapter API

int32_t llama_set_adapter_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora * adapter,
            float scale) {
    ctx->loras[adapter] = scale;
    return 0;
}

int32_t llama_rm_adapter_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora * adapter) {
    auto pos = ctx->loras.find(adapter);
    if (pos != ctx->loras.end()) {
        ctx->loras.erase(pos);
        return 0;
    }

    return -1;
}

void llama_clear_adapter_lora(struct llama_context * ctx) {
    ctx->loras.clear();
}

int32_t llama_apply_adapter_cvec(
        struct llama_context * ctx,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end) {
    return ctx->cvec.apply(ctx->model, data, len, n_embd, il_start, il_end);
}

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const llama_context * ctx, int32_t n_seq_max) {
    return llama_kv_cache_view_init(ctx->kv_self, n_seq_max);
}

void llama_kv_cache_view_update(const llama_context * ctx, llama_kv_cache_view * view) {
    llama_kv_cache_view_update(view, ctx->kv_self);
}

//
// kv cache
//

// deprecated
int32_t llama_get_kv_cache_token_count(const llama_context * ctx) {
    return llama_kv_self_n_tokens(ctx);
}

int32_t llama_kv_self_n_tokens(const llama_context * ctx) {
    return llama_kv_cache_n_tokens(&ctx->kv_self);
}

// deprecated
int32_t llama_get_kv_cache_used_cells(const llama_context * ctx) {
    return llama_kv_self_used_cells(ctx);
}

int32_t llama_kv_self_used_cells(const llama_context * ctx) {
    return llama_kv_cache_used_cells(&ctx->kv_self);
}

// deprecated
void llama_kv_cache_clear(llama_context * ctx) {
    llama_kv_self_clear(ctx);
}

void llama_kv_self_clear(llama_context * ctx) {
    llama_kv_cache_clear(&ctx->kv_self);
}

// deprecated
bool llama_kv_cache_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_self_seq_rm(ctx, seq_id, p0, p1);
}

bool llama_kv_self_seq_rm(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_cache_seq_rm(&ctx->kv_self, seq_id, p0, p1);
}

// deprecated
void llama_kv_cache_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_self_seq_cp(ctx, seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_self_seq_cp(
        llama_context * ctx,
         llama_seq_id   seq_id_src,
         llama_seq_id   seq_id_dst,
            llama_pos   p0,
            llama_pos   p1) {
    return llama_kv_cache_seq_cp(&ctx->kv_self, seq_id_src, seq_id_dst, p0, p1);
}

// deprecated
void llama_kv_cache_seq_keep(
        llama_context * ctx,
         llama_seq_id   seq_id) {
    return llama_kv_self_seq_keep(ctx, seq_id);
}

void llama_kv_self_seq_keep(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_keep(&ctx->kv_self, seq_id);
}

// deprecated
void llama_kv_cache_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    return llama_kv_self_seq_add(ctx, seq_id, p0, p1, delta);
}

void llama_kv_self_seq_add(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
            llama_pos   delta) {
    return llama_kv_cache_seq_add(&ctx->kv_self, seq_id, p0, p1, delta);
}

// deprecated
void llama_kv_cache_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    return llama_kv_self_seq_div(ctx, seq_id, p0, p1, d);
}

void llama_kv_self_seq_div(
        llama_context * ctx,
         llama_seq_id   seq_id,
            llama_pos   p0,
            llama_pos   p1,
                  int   d) {
    return llama_kv_cache_seq_div(&ctx->kv_self, seq_id, p0, p1, d);
}

// deprecated
llama_pos llama_kv_cache_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_self_seq_pos_max(ctx, seq_id);
}

llama_pos llama_kv_self_seq_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_pos_max(&ctx->kv_self, seq_id);
}

// deprecated
void llama_kv_cache_defrag(llama_context * ctx) {
    return llama_kv_self_defrag(ctx);
}

void llama_kv_self_defrag(llama_context * ctx) {
    return llama_kv_cache_defrag(&ctx->kv_self);
}

// deprecated
bool llama_kv_cache_can_shift(const llama_context * ctx) {
    return llama_kv_self_can_shift(ctx);
}

bool llama_kv_self_can_shift(const llama_context * ctx) {
    return llama_kv_cache_can_shift(&ctx->kv_self);
}

// deprecated
void llama_kv_cache_update(llama_context * ctx) {
    llama_kv_self_update(ctx);
}

// llama state API

// deprecated
size_t llama_get_state_size(struct llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(struct llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// TODO: replace all non-fatal assertions with returned errors or exceptions
struct llama_data_write {
    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_write() = default;

    void write_string(const std::string & str) {
        uint32_t str_size = str.size();

        write(&str_size,  sizeof(str_size));
        write(str.data(), str_size);
    }

    void write_model_info(const struct llama_context * ctx) {
        const std::string arch_str = llm_arch_name(ctx->model.arch);
        write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    //void write_rng(const std::mt19937 & rng) {
    //    std::ostringstream rng_ss;
    //    rng_ss << rng;

    //    const std::string & rng_str = rng_ss.str();

    //    write_string(rng_str);
    //}

    void write_output_ids(struct llama_context * ctx) {
        llama_output_reorder(*ctx);

        const uint32_t n_outputs = ctx->n_outputs;

        std::vector<int32_t> output_pos;

        const size_t    n_batch = ctx->cparams.n_batch;
        const auto & output_ids = ctx->output_ids;

        GGML_ASSERT(n_outputs <= ctx->output_size);

        output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch; ++i) {
            // map an output id to a position in the batch
            int32_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT((uint32_t) pos < n_outputs);
                output_pos[pos] = i;
            }
        }

        write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            write(output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    void write_logits(const struct llama_context * ctx) {
        const uint64_t logits_size = std::min((uint64_t) ctx->logits_size, (uint64_t) ctx->n_outputs * ctx->model.vocab.n_tokens());

        write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            write(ctx->logits, logits_size * sizeof(float));
        }
    }

    void write_embeddings(const struct llama_context * ctx) {
        const uint64_t embeddings_size = std::min((uint64_t) ctx->embd_size, (uint64_t) ctx->n_outputs * ctx->model.hparams.n_embd);

        write(&embeddings_size, sizeof(embeddings_size));

        if (embeddings_size) {
            write(ctx->embd, embeddings_size * sizeof(float));
        }
    }
};

struct llama_data_read {
    virtual const uint8_t * read(size_t size) = 0;
    virtual void read_to(void * dst, size_t size) = 0;
    virtual size_t get_size_read() = 0;
    virtual ~llama_data_read() = default;

    void read_string(std::string & str) {
        uint32_t str_size;
        read_to(&str_size, sizeof(str_size));

        str.assign((const char *) read(str_size), str_size);
    }

    // validate model information
    void read_model_info(const struct llama_context * ctx) {
        const std::string cur_arch_str = llm_arch_name(ctx->model.arch);

        std::string arch_str;
        read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    //void read_rng(std::mt19937 & rng) {
    //    std::string rng_str;
    //    read_string(rng_str);

    //    std::istringstream rng_ss(rng_str);
    //    rng_ss >> rng;

    //    if (rng_ss.fail()) {
    //        throw std::runtime_error("failed to load RNG state");
    //    }
    //}

    void read_output_ids(struct llama_context * ctx) {
        std::vector<int32_t> output_pos;

        uint32_t n_outputs;
        read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > llama_output_reserve(*ctx, n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        if (n_outputs) {
            output_pos.resize(n_outputs);
            read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= ctx->cparams.n_batch) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, ctx->cparams.n_batch));
                }
                ctx->output_ids[id] = i;
            }

            ctx->n_outputs = n_outputs;
        }
    }

    void read_logits(struct llama_context * ctx) {
        uint64_t logits_size;
        read_to(&logits_size, sizeof(logits_size));

        if (ctx->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            read_to(ctx->logits, logits_size * sizeof(float));
        }
    }

    void read_embeddings(struct llama_context * ctx) {
        uint64_t embeddings_size;
        read_to(&embeddings_size, sizeof(embeddings_size));

        if (ctx->embd_size < embeddings_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embeddings_size) {
            read_to(ctx->embd, embeddings_size * sizeof(float));
        }
    }
};

struct llama_data_write_dummy : llama_data_write {
    size_t size_written = 0;

    llama_data_write_dummy() {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor_data(const struct ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_write_buffer : llama_data_write {
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    llama_data_write_buffer(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_read_buffer : llama_data_read {
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    llama_data_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t get_size_read() override {
        return size_read;
    }
};

struct llama_data_write_file : llama_data_write {
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;

    llama_data_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_read_file : llama_data_read {
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;

    llama_data_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t get_size_read() override {
        return size_read;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_write_file data_ctx(&file);
 * llama_state_get_data_internal(ctx, data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_write_buffer data_ctx(buf.data(), max_size);
 * llama_state_get_data_internal(ctx, data_ctx);
 *
*/
static size_t llama_state_get_data_internal(struct llama_context * ctx, llama_data_write & data_ctx) {
    llama_synchronize(ctx);

    data_ctx.write_model_info(ctx);

    // copy outputs
    data_ctx.write_output_ids(ctx);
    data_ctx.write_logits(ctx);
    data_ctx.write_embeddings(ctx);

    llama_kv_cache::io io = {
        /* .write = */ [&](const void * src, size_t size) {
            data_ctx.write(src, size);
        },
        /* .write_tensor_data = */ [&](const struct ggml_tensor * tensor, size_t offset, size_t size) {
            data_ctx.write_tensor_data(tensor, offset, size);
        },
        /* .read    = */ nullptr,
        /* .read_to = */ nullptr,
    };

    ctx->kv_self.state_write(io, ctx->model.hparams);

    return data_ctx.get_size_written();
}

size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size) {
    llama_data_write_buffer data_ctx(dst, size);
    try {
        return llama_state_get_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(struct llama_context * ctx) {
    llama_data_write_dummy data_ctx;
    try {
        return llama_state_get_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_set_data_internal(struct llama_context * ctx, llama_data_read & data_ctx) {
    llama_synchronize(ctx);

    data_ctx.read_model_info(ctx);

    // set outputs
    data_ctx.read_output_ids(ctx);
    data_ctx.read_logits(ctx);
    data_ctx.read_embeddings(ctx);

    llama_kv_cache::io io = {
        /* .write = */ nullptr,
        /* .write_tensor_data = */ nullptr,
        /* .read = */ [&](size_t size) {
            return data_ctx.read(size);
        },
        /* .read_to = */ [&](void * dst, size_t size) {
            data_ctx.read_to(dst, size);
        },
    };

    ctx->kv_self.state_read(io, ctx->model.hparams);

    return data_ctx.get_size_read();
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size) {
    llama_data_read_buffer data_ctx(src, size);
    try {
        return llama_state_set_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static bool llama_state_load_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_data_read_file data_ctx(&file);
        const size_t n_read = llama_state_set_data_internal(ctx, data_ctx);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }
    return true;
}

bool llama_state_load_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_state_load_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

static bool llama_state_save_file_internal(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_write_file data_ctx(&file);
    llama_state_get_data_internal(ctx, data_ctx);

    return true;
}

bool llama_state_save_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    try {
        return llama_state_save_file_internal(ctx, path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

static size_t llama_state_seq_get_data_internal(struct llama_context * ctx, llama_data_write & data_ctx, llama_seq_id seq_id) {
    llama_synchronize(ctx);

    llama_kv_cache::io io = {
        /* .write = */ [&](const void * src, size_t size) {
            data_ctx.write(src, size);
        },
        /* .write_tensor_data = */ [&](const struct ggml_tensor * tensor, size_t offset, size_t size) {
            data_ctx.write_tensor_data(tensor, offset, size);
        },
        /* .read = */    nullptr,
        /* .read_to = */ nullptr,
    };

    ctx->kv_self.state_write(io, ctx->model.hparams, seq_id);

    return data_ctx.get_size_written();
}

size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_data_write_dummy data_ctx;
    return llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
}

size_t llama_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    llama_data_write_buffer data_ctx(dst, size);
    try {
        return llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_seq_set_data_internal(struct llama_context * ctx, llama_data_read & data_ctx, llama_seq_id dest_seq_id) {
    llama_synchronize(ctx);

    llama_kv_cache::io io = {
        /* .write = */ nullptr,
        /* .write_tensor_data = */ nullptr,
        /* .read = */ [&](size_t size) {
            return data_ctx.read(size);
        },
        /* .read_to = */ [&](void * dst, size_t size) {
            data_ctx.read_to(dst, size);
        },
    };

    ctx->kv_self.state_read(io, ctx->model.hparams, dest_seq_id);

    return data_ctx.get_size_read();
}

size_t llama_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id) {
    llama_data_read_buffer data_ctx(src, size);
    try {
        return llama_state_seq_set_data_internal(ctx, data_ctx, dest_seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_seq_save_file_internal(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_write_file data_ctx(&file);
    llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + data_ctx.get_size_written());
    return res;
}

static size_t llama_state_seq_load_file_internal(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_data_read_file data_ctx(&file);
        const size_t nread = llama_state_seq_set_data_internal(ctx, data_ctx, dest_seq_id);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_state_seq_save_file(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    try {
        return llama_state_seq_save_file_internal(ctx, filepath, seq_id, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_state_seq_load_file_internal(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
) {
    return ctx->model.tensors_by_name;
}
