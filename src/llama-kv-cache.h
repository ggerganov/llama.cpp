#pragma once

#include "llama.h"

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
    bool can_shift = false;

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

    int32_t n_tokens() const {
        int32_t result = 0;

        for (uint32_t i = 0; i < size; i++) {
            result += cells[i].seq_id.size();
        }

        return result;
    }

    size_t total_size() const {
        size_t size = 0;
        for (const auto & buf : bufs) {
            size += ggml_backend_buffer_get_size(buf.get());
        }

        return size;
    }

    // TODO: better data structures to reduce the cost of this operation
    llama_pos max_pos() const {
        llama_pos max_pos = -1;
        for (const auto & cell : cells) {
            max_pos = std::max(max_pos, cell.pos);
        }

        return max_pos;
    }

    void clear() {
        for (int32_t i = 0; i < (int32_t) size; ++i) {
            cells[i].pos = -1;
            cells[i].seq_id.clear();
            cells[i].src = -1;
            cells[i].tail = -1;
        }
        head = 0;
        used = 0;

        for (auto & buf : bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        uint32_t new_head = size;

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        // models like Mamba or RWKV can't have a state partially erased
        if (recurrent) {
            if (seq_id >= (int64_t) size) {
                // could be fatal
                return false;
            }
            if (0 <= seq_id) {
                int32_t & tail_id = cells[seq_id].tail;
                if (tail_id >= 0) {
                    const llama_kv_cell & cell = cells[tail_id];
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

        for (uint32_t i = 0; i < size; ++i) {
            if (cells[i].pos >= p0 && cells[i].pos < p1) {
                if (seq_id < 0) {
                    cells[i].seq_id.clear();
                } else if (cells[i].has_seq_id(seq_id)) {
                    cells[i].seq_id.erase(seq_id);
                } else {
                    continue;
                }
                if (cells[i].is_empty()) {
                    // keep count of the number of used cells
                    if (cells[i].pos >= 0) {
                        used--;
                    }

                    cells[i].pos = -1;
                    cells[i].src = -1;

                    if (new_head == size) {
                        new_head = i;
                    }
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != size && new_head < head) {
            head = new_head;
        }

        return true;
    }

    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        if (recurrent) {
            if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
                llama_kv_cell & tail_src = cells[seq_id_src];
                llama_kv_cell & tail_dst = cells[seq_id_dst];
                if (tail_dst.tail >= 0) {
                    // clear destination seq_id if it wasn't empty
                    llama_kv_cell & cell_dst = cells[tail_dst.tail];

                    cell_dst.seq_id.erase(seq_id_dst);
                    tail_dst.tail = -1;
                    if (cell_dst.seq_id.empty()) {
                        cell_dst.pos = -1;
                        cell_dst.delta = -1;
                        cell_dst.src = -1;
                        used -= 1;
                    }
                }
                if (tail_src.tail >= 0) {
                    llama_kv_cell & cell_src = cells[tail_src.tail];

                    cell_src.seq_id.insert(seq_id_dst);
                    tail_dst.tail = tail_src.tail;
                }
            }

            return;
        }

        // otherwise, this is the KV of a Transformer-like model
        head = 0;

        for (uint32_t i = 0; i < size; ++i) {
            if (cells[i].has_seq_id(seq_id_src) && cells[i].pos >= p0 && cells[i].pos < p1) {
                cells[i].seq_id.insert(seq_id_dst);
            }
        }
    }

    void seq_keep(llama_seq_id seq_id) {
        uint32_t new_head = size;

        for (uint32_t i = 0; i < size; ++i) {
            if (recurrent && (llama_seq_id) i != seq_id) {
                cells[i].tail = -1;
            }

            if (!cells[i].has_seq_id(seq_id)) {
                if (cells[i].pos >= 0) {
                    used--;
                }

                cells[i].pos = -1;
                cells[i].src = -1;
                cells[i].seq_id.clear();

                if (new_head == size){
                    new_head = i;
                }
            } else {
                cells[i].seq_id.clear();
                cells[i].seq_id.insert(seq_id);
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != size && new_head < head) {
            head = new_head;
        }
    }

    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
        if (delta == 0) {
            return;
        }

        uint32_t new_head = size;

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        // If there is no range then return early to avoid looping over the
        if (p0 == p1) {
            return;
        }

        if (recurrent) {
            // for Mamba-like or RWKV models, only the pos needs to be shifted
            if (0 <= seq_id && seq_id < (int64_t) size) {
                const int32_t tail_id = cells[seq_id].tail;
                if (tail_id >= 0) {
                    llama_kv_cell & cell = cells[tail_id];
                    if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                        cell.pos += delta;
                    }
                }
            }
            return;
        }

        for (uint32_t i = 0; i < size; ++i) {
            if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
                has_shift = true;
                cells[i].pos   += delta;
                cells[i].delta += delta;

                if (cells[i].pos < 0) {
                    if (!cells[i].is_empty()) {
                        used--;
                    }
                    cells[i].pos = -1;
                    cells[i].seq_id.clear();
                    if (new_head == size) {
                        new_head = i;
                    }
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        // Otherwise we just start the next search from the beginning.
        head = new_head != size ? new_head : 0;
    }

    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
        if (d == 1) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        // If there is no range then return early to avoid looping over the cache.
        if (p0 == p1) {
            return;
        }

        if (recurrent) {
            // for Mamba-like or RWKV models, only the pos needs to be changed
            if (0 <= seq_id && seq_id < (int64_t) size) {
                const int32_t tail_id = cells[seq_id].tail;
                if (tail_id >= 0) {
                    llama_kv_cell & cell = cells[tail_id];
                    if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                        cell.pos /= d;
                    }
                }
            }

            return;
        }

        for (uint32_t i = 0; i < size; ++i) {
            if (cells[i].has_seq_id(seq_id) && cells[i].pos >= p0 && cells[i].pos < p1) {
                has_shift = true;

                {
                    llama_pos p_old = cells[i].pos;
                    cells[i].pos   /= d;
                    cells[i].delta += cells[i].pos - p_old;
                }
            }
        }
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) {
        llama_pos result = 0;

        for (uint32_t i = 0; i < size; ++i) {
            if (cells[i].has_seq_id(seq_id)) {
                result = std::max(result, cells[i].pos);
            }
        }

        return result;
    }

    void defrag() {
        if (!recurrent) {
            do_defrag = true;
        }
    }
};

// a structure holds information about the slot found in llama_kv_cache_find_slot
struct llama_kv_cache_slot_info {
    std::pair<uint32_t, uint32_t> boundaries; // slot boundaries [begin, end)
    bool found = false;                       // the slot was found

    explicit llama_kv_cache_slot_info(bool found_) : found{found_} {}
    llama_kv_cache_slot_info(uint32_t begin, uint32_t end) : boundaries{begin, end}, found{true} {}

    operator bool() const { return found; }
};

// TODO: maybe not needed
uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams);

bool llama_kv_cache_init(
        struct llama_kv_cache & cache,
            const llama_model & model,
          const llama_cparams & cparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                     uint32_t   kv_size,
                         bool   offload);

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// returns a structure holding information about the slot found
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
struct llama_kv_cache_slot_info llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
       const struct llama_ubatch & batch);

// find how many cells are currently in use
uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache);

//
// kv cache view
//

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_kv_cache & kv, int32_t n_seq_max);

void llama_kv_cache_view_update(struct llama_kv_cache_view * view, const struct llama_kv_cache & kv);

//
// kv cache restore
//

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
        old_state.head = cache.head;
        old_state.n    = cache.n;
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
            cache.head = old_state.head;
            cache.n    = old_state.n;

            if (cache.recurrent) { // recurrent models like Mamba or RWKV can't have a state partially erased
                cache.seq_rm(-1, -1, -1);
            } else {
                for (auto & slot : slot_boundaries) {
                    cache.seq_rm(-1, slot.first, slot.second);
                }
            }
        }
    }
};

