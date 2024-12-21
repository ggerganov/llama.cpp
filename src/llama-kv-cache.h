#pragma once

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <set>
#include <vector>

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = -1; // used by recurrent state models to copy states
    int32_t   tail  = -1;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool v_trans   = true;  // the value tensor is transposed

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    size_t total_size() {
        size_t size = 0;
        for (auto & buf : bufs) {
            size += ggml_backend_buffer_get_size(buf.get());
        }
        return size;
    }
};

//
// kv cache helpers
//

// a structure holds information about the slot found in llama_kv_cache_find_slot
struct llama_kv_cache_slot_info {
    std::pair<uint32_t, uint32_t> boundaries; // slot boundaries [begin, end)
    bool found = false;                       // the slot was found

    explicit llama_kv_cache_slot_info(bool found_) : found{found_} {}
    llama_kv_cache_slot_info(uint32_t begin, uint32_t end) : boundaries{begin, end}, found{true} {}

    operator bool() const { return found; }
};
static const llama_kv_cache_slot_info llama_kv_cache_slot_info_failed{false};

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// returns a structure holding information about the slot found
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
static struct llama_kv_cache_slot_info llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
       const struct llama_ubatch & batch) {
    const uint32_t n_tokens = batch.n_tokens;
    const uint32_t n_seqs   = batch.n_seqs;
    const uint32_t n_seq_tokens = batch.n_seq_tokens;

    if (cache.recurrent) {
        // For recurrent state architectures (like Mamba or RWKV),
        // each cache cell can store the state for a whole sequence.
        // A slot should be always be contiguous.

        // can only process batches with an equal number of new tokens in each sequence
        GGML_ASSERT(batch.equal_seqs);

        int32_t min = cache.size - 1;
        int32_t max = 0;

        // everything should fit if all seq_ids are smaller than the max
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const uint32_t n_seq_id = batch.n_seq_id[s];
            for (uint32_t j = 0; j < n_seq_id; ++j) {
                const llama_seq_id seq_id = batch.seq_id[s][j];

                if (seq_id < 0 || (uint32_t) seq_id >= cache.size) {
                    // too big seq_id
                    // TODO: would it be possible to resize the cache instead?
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%d Try using a bigger --parallel value\n", __func__, seq_id, cache.size);
                    return llama_kv_cache_slot_info_failed;
                }
                if (j > 0) {
                    llama_kv_cell & seq = cache.cells[seq_id];
                    if (seq.tail >= 0) {
                        llama_kv_cell & cell = cache.cells[seq.tail];
                        // clear cells from seq_ids that become shared
                        // (should not normally happen, but let's handle it anyway)
                        cell.seq_id.erase(seq_id);
                        seq.tail = -1;
                        if (cell.seq_id.empty()) {
                            cell.pos = -1;
                            cell.src = -1;
                            cache.used -= 1;
                        }
                    }
                }
            }
        }

#ifndef NDEBUG
        {
            std::vector<int32_t> tails_verif;
            tails_verif.assign(cache.size, -1);
            for (uint32_t i = 0; i < cache.size; ++i) {
                llama_kv_cell & cell = cache.cells[i];
                for (llama_seq_id seq_id : cell.seq_id) {
                    if (tails_verif[seq_id] != -1) {
                        LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                    }
                    tails_verif[seq_id] = i;
                }
            }
            for (uint32_t i = 0; i < cache.size; ++i) {
                if (tails_verif[i] != cache.cells[i].tail) {
                    LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cache.cells[i].tail, tails_verif[i]);
                }
            }
        }
#endif

        // find next empty cell
        uint32_t next_empty_cell = cache.head;

        for (uint32_t i = 0; i < cache.size; ++i) {
            if (next_empty_cell >= cache.size) { next_empty_cell -= cache.size; }
            llama_kv_cell & cell = cache.cells[next_empty_cell];
            if (cell.is_empty()) { break; }
            next_empty_cell += 1;
        }

        // find usable cell range
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = batch.seq_id[s][0];
            llama_kv_cell & seq_meta = cache.cells[seq_id];
            bool has_cell = false;
            if (seq_meta.tail >= 0) {
                llama_kv_cell & cell = cache.cells[seq_meta.tail];
                GGML_ASSERT(cell.has_seq_id(seq_id));
                // does this seq_id "own" the cell?
                if (cell.seq_id.size() == 1) { has_cell = true; }
            }
            if (!has_cell) {
                llama_kv_cell & empty_cell = cache.cells[next_empty_cell];
                GGML_ASSERT(empty_cell.is_empty());
                // copy old tail into the empty cell
                if (seq_meta.tail >= 0) {
                    llama_kv_cell & orig_cell = cache.cells[seq_meta.tail];
                    empty_cell.pos = orig_cell.pos;
                    empty_cell.src = orig_cell.src;
                    orig_cell.seq_id.erase(seq_id);
                    empty_cell.seq_id.insert(seq_id); // will be overwritten
                }
                seq_meta.tail = next_empty_cell;
                // find next empty cell
                if (s + 1 < n_seqs) {
                    next_empty_cell += 1;
                    for (uint32_t i = 0; i < cache.size; ++i) {
                        if (next_empty_cell >= cache.size) { next_empty_cell -= cache.size; }
                        llama_kv_cell & cell = cache.cells[next_empty_cell];
                        if (cell.is_empty()) { break; }
                        next_empty_cell += 1;
                    }
                }
            }
            if (min > seq_meta.tail) { min = seq_meta.tail; }
            if (max < seq_meta.tail) { max = seq_meta.tail; }
        }

        // gather and re-order
        for (uint32_t s = 0; s < n_seqs; ++s) {
            int32_t dst_id = s + min;
            int32_t src_id = cache.cells[batch.seq_id[s][0]].tail;
            if (dst_id != src_id) {
                llama_kv_cell & dst_cell = cache.cells[dst_id];
                llama_kv_cell & src_cell = cache.cells[src_id];

                std::swap(dst_cell.pos, src_cell.pos);
                std::swap(dst_cell.src, src_cell.src);
                std::swap(dst_cell.seq_id, src_cell.seq_id);

                // swap tails (assuming they NEVER overlap)
                for (const llama_seq_id seq_id : src_cell.seq_id) {
                    cache.cells[seq_id].tail = src_id;
                }
                for (const llama_seq_id seq_id : dst_cell.seq_id) {
                    cache.cells[seq_id].tail = dst_id;
                }
            }
        }

        // update the pos of the used seqs
        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_pos last_pos = batch.pos[n_seq_tokens * s + n_seq_tokens - 1];
            int32_t cell_id = s + min;
            llama_kv_cell & cell = cache.cells[cell_id];

            if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
                // What should happen when the pos backtracks or skips a value?
                // Clearing the state mid-batch would require special-casing which isn't done.
                LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                    __func__, last_pos, cell.pos, batch.seq_id[s][0], n_seq_tokens);
            }
            cell.pos = last_pos;
            cell.seq_id.clear();
            for (int32_t j = 0; j < batch.n_seq_id[s]; ++j) {
                const llama_seq_id seq_id = batch.seq_id[s][j];
                cell.seq_id.insert(seq_id);
                cache.cells[seq_id].tail = cell_id;
            }
        }

        // allow getting the range of used cells, from head to head + n
        cache.head = min;
        cache.n    = max - min + 1;
        cache.used = std::count_if(cache.cells.begin(), cache.cells.end(),
            [](const llama_kv_cell& cell){ return !cell.is_empty(); });

        // sanity check
        return llama_kv_cache_slot_info(cache.n >= n_seqs);
    }
    // otherwise, one cell per token.

    if (n_tokens > cache.size) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > cache.size=%d\n", __func__, n_tokens, cache.size);
        return llama_kv_cache_slot_info_failed;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > cache.size) {
            n_tested += cache.size - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= cache.size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return llama_kv_cache_slot_info_failed;
        }
    }

    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k = s*n_seq_tokens + i;
            cache.cells[cache.head + k].pos = batch.pos[k];

            for (int32_t j = 0; j < batch.n_seq_id[s]; j++) {
                cache.cells[cache.head + k].seq_id.insert(batch.seq_id[s][j]);
            }
        }
    }

    cache.used += n_tokens;

    return llama_kv_cache_slot_info(cache.head, cache.head + n_tokens);
}

// find how many cells are currently in use
static uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size; i > 0; --i) {
        const llama_kv_cell & cell = cache.cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

static void llama_kv_cache_clear(struct llama_kv_cache & cache) {
    for (int32_t i = 0; i < (int32_t) cache.size; ++i) {
        cache.cells[i].pos = -1;
        cache.cells[i].seq_id.clear();
        cache.cells[i].src = -1;
        cache.cells[i].tail = -1;
    }
    cache.head = 0;
    cache.used = 0;

    for (auto & buf : cache.bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

static bool llama_kv_cache_seq_rm(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    // models like Mamba or RWKV can't have a state partially erased
    if (cache.recurrent) {
        if (seq_id >= (int64_t) cache.size) {
            // could be fatal
            return false;
        }
        if (0 <= seq_id) {
            int32_t & tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                const llama_kv_cell & cell = cache.cells[tail_id];
                // partial intersection is invalid
                if ((0 < p0 && p0 <= cell.pos) || (0 < p1 && p1 <= cell.pos)) {
                    return false;
                }
                // invalidate tails which will be cleared
                if (p0 <= cell.pos && cell.pos < p1) {
                    tail_id = -1;
                }
            }
        } else {
            // seq_id is negative, then the range should include everything or nothing
            if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
                return false;
            }
        }
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            if (seq_id < 0) {
                cache.cells[i].seq_id.clear();
            } else if (cache.cells[i].has_seq_id(seq_id)) {
                cache.cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cache.cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cache.cells[i].pos >= 0) cache.used--;

                cache.cells[i].pos = -1;
                cache.cells[i].src = -1;
                if (new_head == cache.size) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;

    return true;
}

static void llama_kv_cache_seq_cp(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id_src,
                 llama_seq_id   seq_id_dst,
                    llama_pos   p0,
                    llama_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        if ((uint32_t) seq_id_dst < cache.size && (uint32_t) seq_id_src < cache.size) {
            llama_kv_cell & tail_src = cache.cells[seq_id_src];
            llama_kv_cell & tail_dst = cache.cells[seq_id_dst];
            if (tail_dst.tail >= 0) {
                // clear destination seq_id if it wasn't empty
                llama_kv_cell & cell_dst = cache.cells[tail_dst.tail];

                cell_dst.seq_id.erase(seq_id_dst);
                tail_dst.tail = -1;
                if (cell_dst.seq_id.empty()) {
                    cell_dst.pos = -1;
                    cell_dst.delta = -1;
                    cell_dst.src = -1;
                    cache.used -= 1;
                }
            }
            if (tail_src.tail >= 0) {
                llama_kv_cell & cell_src = cache.cells[tail_src.tail];

                cell_src.seq_id.insert(seq_id_dst);
                tail_dst.tail = tail_src.tail;
            }
        }

        return;
    }
    // otherwise, this is the KV cache of a Transformer-like model

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

static void llama_kv_cache_seq_keep(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    uint32_t new_head = cache.size;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.recurrent && (llama_seq_id) i != seq_id) {
            cache.cells[i].tail = -1;
        }
        if (!cache.cells[i].has_seq_id(seq_id)) {
            if (cache.cells[i].pos >= 0) cache.used--;
            cache.cells[i].pos = -1;
            cache.cells[i].src = -1;
            cache.cells[i].seq_id.clear();
            if (new_head == cache.size) new_head = i;
        } else {
            cache.cells[i].seq_id.clear();
            cache.cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;
}

static void llama_kv_cache_seq_add(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) return;

    if (cache.recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be shifted
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            const int32_t tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cache.cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos += delta;
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;
            cache.cells[i].pos   += delta;
            cache.cells[i].delta += delta;

            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;
}

static void llama_kv_cache_seq_div(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                          int   d) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) return;

    if (cache.recurrent) {
        // for Mamba-like or RWKV models, only the pos needs to be changed
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            const int32_t tail_id = cache.cells[seq_id].tail;
            if (tail_id >= 0) {
                llama_kv_cell & cell = cache.cells[tail_id];
                if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                    cell.pos /= d;
                }
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;

            {
                llama_pos p_old = cache.cells[i].pos;
                cache.cells[i].pos   /= d;
                cache.cells[i].delta += cache.cells[i].pos - p_old;
            }
        }
    }
}

static llama_pos llama_kv_cache_seq_pos_max(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    llama_pos result = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cache.cells[i].pos);
        }
    }

    return result;
}

static void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    if (!cache.recurrent) {
        cache.do_defrag = true;
    }
}

// saves the kv_cache state for future recovery.
// used to rollback llama_kv_cache_find_slot changes.
struct llama_kv_slot_restorer {
    struct llama_kv_cache_state {
        uint32_t head = 0;
        uint32_t n    = 0;
    } old_state;

    // for non-recurrent models only
    // list of slots to restore
    std::vector<std::pair<uint32_t, uint32_t>> slot_boundaries;

    bool do_restore = false;

    explicit llama_kv_slot_restorer(const struct llama_kv_cache & cache) {
        old_state.head  = cache.head;
        old_state.n     = cache.n;
    }

    // saves a slot information for future restoration
    void save(const struct llama_kv_cache_slot_info & slot) {
        if (slot) {
            do_restore = true;
            if (slot.boundaries.first != slot.boundaries.second) {
                slot_boundaries.push_back(slot.boundaries);
            }
        }
    }

    // must be explicitly called to restore the kv_cache state
    // and rollback changes from all llama_kv_cache_find_slot calls
    void restore(struct llama_kv_cache & cache) {
        if (do_restore) {
            cache.head  = old_state.head;
            cache.n     = old_state.n;

            if (cache.recurrent) { // recurrent models like Mamba or RWKV can't have a state partially erased
                llama_kv_cache_seq_rm(cache, -1, -1, -1);
            } else {
                for (auto & slot : slot_boundaries) {
                    llama_kv_cache_seq_rm(cache, -1, slot.first, slot.second);
                }
            }
        }
    }
};
