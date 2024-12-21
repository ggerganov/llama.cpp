#include "llama-context.h"

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
        std::string arch_str = LLM_ARCH_NAMES.at(ctx->model.arch);
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
        llama_output_reorder(ctx);

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
        const uint64_t logits_size = std::min((uint64_t) ctx->logits_size, (uint64_t) ctx->n_outputs * ctx->model.hparams.n_vocab);

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

    void write_kv_cache_meta(const llama_kv_cache & kv_self, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) {

        for (const auto & range : cell_ranges) {
            for (uint32_t i = range.first; i < range.second; ++i) {
                const auto & cell = kv_self.cells[i];
                const llama_pos pos      = cell.pos;
                const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

                write(&pos,      sizeof(pos));
                write(&n_seq_id, sizeof(n_seq_id));

                if (n_seq_id) {
                    for (auto seq_id : cell.seq_id) {
                        write(&seq_id, sizeof(seq_id));
                    }
                }
            }
        }
    }

    void write_kv_cache_data(const struct llama_context * ctx, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) {
        const struct llama_kv_cache & kv_self = ctx->kv_self;
        const struct llama_hparams & hparams = ctx->model.hparams;

        const uint32_t v_trans = kv_self.v_trans ? 1 : 0;
        const uint32_t n_layer = hparams.n_layer;

        write(&v_trans, sizeof(v_trans));
        write(&n_layer, sizeof(n_layer));

        std::vector<uint8_t> tmp_buf;

        // Iterate and write all the keys first, each row is a cell
        // Get whole range at a time
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

            // Write key type
            const int32_t k_type_i = (int32_t)kv_self.k_l[il]->type;
            write(&k_type_i, sizeof(k_type_i));

            // Write row size of key
            const uint64_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
            write(&k_size_row, sizeof(k_size_row));

            // Read each range of cells of k_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * k_size_row;
                write_tensor_data(kv_self.k_l[il], range.first * k_size_row, buf_size);
            }
        }

        if (!kv_self.v_trans) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Write value type
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                write(&v_type_i, sizeof(v_type_i));

                // Write row size of value
                const uint64_t v_size_row = ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa);
                write(&v_size_row, sizeof(v_size_row));

                // Read each range of cells of v_size length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t buf_size = range_size * v_size_row;
                    write_tensor_data(kv_self.v_l[il], range.first * v_size_row, buf_size);
                }
            }
        } else {
            // When v is transposed, we also need the element size and get the element ranges from each row
            const uint32_t kv_size = kv_self.size;
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Write value type
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                write(&v_type_i, sizeof(v_type_i));

                // Write element size
                const uint32_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
                write(&v_size_el, sizeof(v_size_el));

                // Write GQA embedding size
                write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

                // For each row, we get the element values of each cell
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    // Read each range of cells of v_size_el length each into tmp_buf and write out
                    for (const auto & range : cell_ranges) {
                        const size_t range_size = range.second - range.first;
                        const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                        const size_t buf_size = range_size * v_size_el;
                        write_tensor_data(kv_self.v_l[il], src_offset, buf_size);
                    }
                }
            }
        }
    }

    void write_kv_cache(const struct llama_context * ctx, llama_seq_id seq_id = -1) {
        const struct llama_kv_cache & kv_self = ctx->kv_self;
        std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
        uint32_t cell_count = 0;

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = kv_self.size;
        for (uint32_t i = 0; i < kv_self.size; ++i) {
            const auto & cell = kv_self.cells[i];
            if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
                ++cell_count;
                if (cell_range_begin == kv_self.size) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != kv_self.size) {
                    cell_ranges.emplace_back(cell_range_begin, i);
                    cell_range_begin = kv_self.size;
                }
            }
        }
        if (cell_range_begin != kv_self.size) {
            cell_ranges.emplace_back(cell_range_begin, kv_self.size);
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cell_ranges) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        write(&cell_count, sizeof(cell_count));

        write_kv_cache_meta(kv_self, cell_ranges, seq_id);
        write_kv_cache_data(ctx, cell_ranges);
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
        std::string cur_arch_str = LLM_ARCH_NAMES.at(ctx->model.arch);
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

    bool read_kv_cache_meta(struct llama_context * ctx, uint32_t cell_count, llama_seq_id dest_seq_id = -1) {
        struct llama_kv_cache & kv_self = ctx->kv_self;

        if (dest_seq_id != -1) {
            // single sequence

            llama_kv_cache_seq_rm(kv_self, dest_seq_id, -1, -1);

            llama_ubatch batch = ctx->sbatch.reserve_ubatch(cell_count, /* has_embd */ false);
            batch.n_tokens = cell_count;
            batch.n_seq_tokens = cell_count;
            batch.n_seqs = 1;

            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_pos pos;
                uint32_t n_seq_id;

                read_to(&pos, sizeof(pos));
                read_to(&n_seq_id, sizeof(n_seq_id));

                if (n_seq_id != 0) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                    return false;
                }

                batch.pos[i] = pos;
            }
            batch.n_seq_id[0] = 1;
            batch.seq_id[0] = &dest_seq_id;
            if (!llama_kv_cache_find_slot(kv_self, batch)) {
                LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
                return false;
            }

            // DEBUG CHECK: kv_self.head should be our first cell, kv_self.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
            // Assume that this is one contiguous block of cells
            GGML_ASSERT(kv_self.head + cell_count <= kv_self.size);
            GGML_ASSERT(kv_self.cells[kv_self.head].pos == batch.pos[0]);
            GGML_ASSERT(kv_self.cells[kv_self.head + cell_count - 1].pos == batch.pos[cell_count - 1]);
            GGML_ASSERT(kv_self.cells[kv_self.head].has_seq_id(dest_seq_id));
            GGML_ASSERT(kv_self.cells[kv_self.head + cell_count - 1].has_seq_id(dest_seq_id));
        } else {
            // whole KV cache restore

            if (cell_count > kv_self.size) {
                LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
                return false;
            }

            llama_kv_cache_clear(kv_self);

            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_kv_cell & cell = kv_self.cells[i];

                llama_pos pos;
                uint32_t  n_seq_id;

                read_to(&pos,      sizeof(pos));
                read_to(&n_seq_id, sizeof(n_seq_id));

                cell.pos = pos;

                for (uint32_t j = 0; j < n_seq_id; ++j) {
                    llama_seq_id seq_id;
                    read_to(&seq_id, sizeof(seq_id));

                    if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx)) {
                        LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, llama_n_seq_max(ctx));
                        return false;
                    }

                    cell.seq_id.insert(seq_id);

                    if (kv_self.recurrent) {
                        int32_t & tail = kv_self.cells[seq_id].tail;
                        if (tail != -1) {
                            LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                            return false;
                        }
                        tail = i;
                    }
                }
            }

            kv_self.head = 0;
            kv_self.used = cell_count;
        }

        if (kv_self.recurrent) {
            for (uint32_t i = 0; i < cell_count; ++i) {
                uint32_t cell_id = kv_self.head + i;
                // make sure the recurrent states will keep their restored state
                kv_self.cells[cell_id].src = cell_id;
            }
        }

        return true;
    }

    bool read_kv_cache_data(struct llama_context * ctx, uint32_t cell_count) {
        const struct llama_hparams & hparams = ctx->model.hparams;
        struct llama_kv_cache & kv_self = ctx->kv_self;
        uint32_t v_trans;
        uint32_t n_layer;
        read_to(&v_trans, sizeof(v_trans));
        read_to(&n_layer, sizeof(n_layer));

        if (n_layer != hparams.n_layer) {
            LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
            return false;
        }
        if (cell_count > kv_self.size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, kv_self.size);
            return false;
        }
        if (kv_self.v_trans != (bool) v_trans) {
            LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
            return false;
        }

        // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

            // Read type of key
            int32_t k_type_i_ref;
            read_to(&k_type_i_ref, sizeof(k_type_i_ref));
            const int32_t k_type_i = (int32_t)kv_self.k_l[il]->type;
            if (k_type_i != k_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
                return false;
            }

            // Read row size of key
            uint64_t k_size_row_ref;
            read_to(&k_size_row_ref, sizeof(k_size_row_ref));
            const size_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
            if (k_size_row != k_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the keys for the whole cell range
                ggml_backend_tensor_set(kv_self.k_l[il], read(cell_count * k_size_row), kv_self.head * k_size_row, cell_count * k_size_row);
            }
        }

        if (!kv_self.v_trans) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Read type of value
                int32_t v_type_i_ref;
                read_to(&v_type_i_ref, sizeof(v_type_i_ref));
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                // Read row size of value
                uint64_t v_size_row_ref;
                read_to(&v_size_row_ref, sizeof(v_size_row_ref));
                const size_t v_size_row = ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa);
                if (v_size_row != v_size_row_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                    return false;
                }

                if (cell_count) {
                    // Read and set the values for the whole cell range
                    ggml_backend_tensor_set(kv_self.v_l[il], read(cell_count * v_size_row), kv_self.head * v_size_row, cell_count * v_size_row);
                }
            }
        } else {
            // For each layer, read the values for each cell (transposed)
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Read type of value
                int32_t v_type_i_ref;
                read_to(&v_type_i_ref, sizeof(v_type_i_ref));
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                // Read element size of value
                uint32_t v_size_el_ref;
                read_to(&v_size_el_ref, sizeof(v_size_el_ref));
                const size_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
                if (v_size_el != v_size_el_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                    return false;
                }

                // Read GQA embedding size
                uint32_t n_embd_v_gqa_ref;
                read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
                if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                    return false;
                }

                if (cell_count) {
                    // For each row in the transposed matrix, read the values for the whole cell range
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (kv_self.head + j * kv_self.size) * v_size_el;
                        ggml_backend_tensor_set(kv_self.v_l[il], read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                    }
                }
            }
        }
        return true;
    }

    void read_kv_cache(struct llama_context * ctx, llama_seq_id seq_id = -1) {
        uint32_t cell_count;
        read_to(&cell_count, sizeof(cell_count));

        bool res = read_kv_cache_meta(ctx, cell_count, seq_id) && read_kv_cache_data(ctx, cell_count);

        if (!res) {
            if (seq_id == -1) {
                llama_kv_cache_clear(ctx);
            } else {
                llama_kv_cache_seq_rm(ctx, seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
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

    data_ctx.write_kv_cache(ctx);

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

    data_ctx.read_kv_cache(ctx);

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
        const size_t n_state_size_cur = file.size - file.tell();

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

    data_ctx.write_kv_cache(ctx, seq_id);

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

    data_ctx.read_kv_cache(ctx, dest_seq_id);

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
        const size_t state_size = file.size - file.tell();
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

