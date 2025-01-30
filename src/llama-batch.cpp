#include "llama-batch.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>
#include <cassert>

class llama_batch_allocr {
public:
    llama_batch batch;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id*> seq_id;
    std::vector<int8_t> logits;

    llama_batch_allocr(const llama_batch& in_batch, llama_pos p0) {
        batch = in_batch;
        assert(batch.n_tokens > 0);
        initialize_batch(p0);
    }

private:
    void initialize_batch(llama_pos p0) {
        if (!batch.pos) {
            pos.resize(batch.n_tokens);
            for (int32_t i = 0; i < batch.n_tokens; i++) {
                pos[i] = i + p0;
            }
            batch.pos = pos.data();
        }
        if (!batch.n_seq_id) {
            n_seq_id.resize(batch.n_tokens);
            for (int32_t i = 0; i < batch.n_tokens; i++) {
                n_seq_id[i] = seq_id_0.size();
            }
            batch.n_seq_id = n_seq_id.data();
        }
        if (!batch.seq_id) {
            seq_id.resize(batch.n_tokens + 1);
            seq_id[batch.n_tokens] = nullptr;
            for (int32_t i = 0; i < batch.n_tokens; i++) {
                seq_id[i] = seq_id_0.data();
            }
            batch.seq_id = seq_id.data();
        }
        if (!batch.logits) {
            logits.resize(batch.n_tokens, 0);
            logits[logits.size() - 1] = true;
            batch.logits = logits.data();
        }
    }
};

class llama_sbatch {
public:
    const llama_batch* batch;
    size_t n_embd;
    bool logits_all;
    size_t n_tokens;
    std::vector<size_t> ids;
    std::vector<llama_sbatch_seq> seq;
    std::vector<size_t> out_ids;

    llama_sbatch() : batch(nullptr), n_embd(0), logits_all(false), n_tokens(0) {}

    llama_ubatch reserve_ubatch(size_t n_ubatch, bool has_embd) {
        clear_empty_sequences();
        resize_ubatch_data(n_ubatch, has_embd);
        return build_ubatch(n_ubatch, has_embd);
    }

    void add_seq_to_ubatch(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        assert(batch != nullptr);
        assert(length <= seq.length);
        validate_sequence_length(ubatch, seq, length);

        handle_tokens_and_embeddings(ubatch, seq, length);
        handle_positions_and_sequence_ids(ubatch, seq, length);
        handle_logits(ubatch, seq, length);

        update_ubatch_and_seq(ubatch, seq, length);
    }

private:
    void clear_empty_sequences() {
        for (size_t i = seq.size(); i-- > 0;) {
            if (seq[i].length == 0) {
                seq.pop_back();
            } else {
                break;
            }
        }
    }

    void resize_ubatch_data(size_t n_ubatch, bool has_embd) {
        ubatch_token.resize(!has_embd ? n_ubatch : 0);
        ubatch_embd.resize(has_embd ? n_embd * n_ubatch : 0);
        ubatch_pos.resize(n_ubatch);
        ubatch_n_seq_id.resize(n_ubatch);
        ubatch_seq_id.resize(n_ubatch);
        ubatch_output.resize(n_ubatch);
    }

    llama_ubatch build_ubatch(size_t n_ubatch, bool has_embd) {
        return {
            true, 0, 0, 0,
            !has_embd ? ubatch_token.data() : nullptr,
            has_embd ? ubatch_embd.data() : nullptr,
            ubatch_pos.data(),
            ubatch_n_seq_id.data(),
            ubatch_seq_id.data(),
            ubatch_output.data()
        };
    }

    void validate_sequence_length(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        assert(seq.n_seq_id == 0 || ubatch.n_seqs == 0 || length == ubatch.n_tokens / ubatch.n_seqs);
        assert((seq.n_seq_id != 0) == ubatch.equal_seqs);
    }

    void handle_tokens_and_embeddings(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        if (batch->token) {
            if (ubatch.equal_seqs) {
                for (size_t i = 0; i < length; ++i) {
                    ubatch.token[ubatch.n_tokens + i] = batch->token[ids[seq.offset + i]];
                }
            } else {
                ubatch.token = batch->token + seq.offset;
            }
        }
        if (batch->embd) {
            if (ubatch.equal_seqs) {
                for (size_t i = 0; i < length; ++i) {
                    std::memcpy(ubatch.embd + n_embd * (ubatch.n_tokens + i),
                                batch->embd + n_embd * ids[seq.offset + i],
                                n_embd * sizeof(float));
                }
            } else {
                ubatch.embd = batch->embd + n_embd * seq.offset;
            }
        }
    }

    void handle_positions_and_sequence_ids(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                ubatch.pos[ubatch.n_tokens + i] = batch->pos[ids[seq.offset + i]];
            }
        } else {
            ubatch.pos = batch->pos + seq.offset;
        }

        if (ubatch.equal_seqs) {
            ubatch.n_seq_id[ubatch.n_seqs] = seq.n_seq_id;
            if (seq.seq_id) {
                ubatch.seq_id[ubatch.n_seqs] = seq.seq_id;
            }
        } else {
            if (batch->n_seq_id) {
                ubatch.n_seq_id = batch->n_seq_id + seq.offset;
            } else {
                for (size_t i = 0; i < length; ++i) {
                    ubatch.n_seq_id[ubatch.n_seqs + i] = 1;
                }
            }
            if (batch->seq_id) {
                ubatch.seq_id = batch->seq_id + seq.offset;
            }
        }
    }

    void handle_logits(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        if (logits_all) {
            for (size_t i = 0; i < length; ++i) {
                ubatch.output[ubatch.n_tokens + i] = 1;
                out_ids.push_back(ids[seq.offset + i]);
            }
        } else if (batch->logits) {
            if (ubatch.equal_seqs) {
                for (size_t i = 0; i < length; ++i) {
                    size_t id = ids[seq.offset + i];
                    int8_t is_output = batch->logits[id];
                    ubatch.output[ubatch.n_tokens + i] = is_output;
                    if (is_output) {
                        out_ids.push_back(id);
                    }
                }
            } else {
                ubatch.output = batch->logits + seq.offset;
                for (size_t i = 0; i < length; ++i) {
                    if (ubatch.output[i] != 0) {
                        out_ids.push_back(seq.offset + i);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < length; ++i) {
                size_t id = ids[seq.offset + i];
                int8_t is_last = id == ids.size() - 1;
                ubatch.output[ubatch.n_tokens + i] = is_last;
                if (is_last) {
                    out_ids.push_back(id);
                }
            }
        }
    }

    void update_ubatch_and_seq(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        if (ubatch.n_tokens == 0 && ubatch.n_seqs == 0) {
            ubatch.n_seq_tokens = ubatch.equal_seqs ? length : 1;
        }
        ubatch.n_tokens += length;
        ubatch.n_seqs += ubatch.equal_seqs ? 1 : length;
        seq.offset += length;
        seq.length -= length;
        n_tokens -= length;

        assert(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);
    }
};

// Initialization functions remain unchanged
llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens) {
    return {
        n_tokens,
        tokens,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    };
}

llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = {0};

    if (embd) {
        batch.embd = new float[n_tokens_alloc * embd]();
    } else {
        batch.token = new llama_token[n_tokens_alloc]();
    }

    batch.pos = new llama_pos[n_tokens_alloc]();
    batch.n_seq_id = new int32_t[n_tokens_alloc]();
    batch.seq_id = new llama_seq_id*[n_tokens_alloc + 1];
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = new llama_seq_id[n_seq_max]();
    }
    batch.seq_id[n_tokens_alloc] = nullptr;
    batch.logits = new int8_t[n_tokens_alloc]();

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    delete[] batch.token;
    delete[] batch.embd;
    delete[] batch.pos;
    delete[] batch.n_seq_id;
    for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
        delete[] batch.seq_id[i];
    }
    delete[] batch.seq_id;
    delete[] batch.logits;
}
