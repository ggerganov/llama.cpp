#pragma once

#include "common.h"
#include "llama.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// LMCache: keeps the saved state of finished conversations in RAM (and optionally on disk) and restores the one that saves the most
// prompt processing for a new prompt. Works across slots and, with the disk tier, across restarts.
//
// - An entry is the whole sequence state of a slot after a request (KV cache and recurrent state), the exact tokens it holds, and up to
//   max_checkpoints of the slot's context checkpoints (the earliest and the latest).
// - Restoring an entry puts the slot exactly where it would be had it processed the entry itself: its state, its tokens and its
//   checkpoints. The server's own prompt-reuse logic then continues as usual: it keeps the tokens the entry shares with the new prompt
//   and, for a model with recurrent layers, goes back to a checkpoint at or before the first difference. So an entry is used both when
//   it is a whole prefix of the prompt and when the prompt changes inside it (a compacted conversation keeps its system prompt, an
//   edited message), as long as the point the server can resume from is further than what the slot already has.
// - At most two entries per conversation: a new entry drops every earlier turn of the same conversation except the previous one (that
//   one serves a regenerated answer).
// - Entries are indexed by a hash of every chunk-aligned prefix of their tokens; a candidate is always checked token by token.
// - Disk tier: new entries are written only after the server has been quiet for disk_idle_sec (and at shutdown), and an entry that a
//   longer entry of the same conversation continues is not written, so a long agent session costs one write per quiet period.
class server_lmcache {
public:
    // recurrent: the model has recurrent layers (its state cannot be cut back; only a checkpoint brings it back to an earlier position)
    server_lmcache(const common_params_lmcache & params, bool recurrent);
    ~server_lmcache();

    // store the state of sequence seq of ctx, which holds exactly these tokens, with the slot's checkpoints
    void save(llama_context * ctx, llama_seq_id seq, const llama_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints);

    struct restored {
        llama_tokens tokens;                             // the tokens the slot now holds (the whole entry)
        size_t       n_common = 0;                       // of them, the ones the prompt starts with (the server's n_past)
        size_t       n_resume = 0;                       // where the server will resume (n_common, or a checkpoint before it)
        std::list<common_prompt_checkpoint> checkpoints; // the entry's checkpoints
    };

    // restore into sequence seq of ctx the entry from which the prompt can resume furthest, if that is beyond n_have (what the slot can
    // already reuse); returns false if none (the sequence is then untouched)
    bool restore(llama_context * ctx, llama_seq_id seq, const llama_tokens & prompt, size_t n_have, restored & out);

private:
    struct payload {
        std::vector<uint8_t> state;
        std::list<common_prompt_checkpoint> checkpoints;

        size_t bytes() const;
    };

    struct entry {
        uint64_t hash = 0; // hash of all tokens, also the disk file name
        llama_tokens tokens;
        std::vector<uint64_t> keys;      // hash of every chunk-aligned prefix, keys[c - 1] for c chunks
        std::vector<int32_t> ckpt_pos;   // pos_max of its checkpoints, also when only on disk
        std::shared_ptr<payload> data; // null when only on disk
        size_t bytes = 0;              // RAM it takes when in RAM
        bool on_disk = false;          // a file holds it
        bool pending = false;          // to be written at the next quiet period
    };
    using entry_it = std::list<entry>::iterator;

    common_params_lmcache params;
    bool recurrent = false;

    std::mutex mutex;
    std::list<entry> entries; // most recently used first
    std::unordered_multimap<uint64_t, entry_it> index; // every chunk key of every entry
    size_t ram_bytes = 0;

    // disk tier: one writer thread
    std::thread writer;
    std::condition_variable cv;
    bool stop = false;
    std::chrono::steady_clock::time_point last_activity = std::chrono::steady_clock::now();

    std::vector<uint64_t> chunk_keys(const llama_tokens & tokens) const;
    size_t resume_point(const entry & e, size_t n_common) const;
    void erase(entry_it it, bool remove_file);
    void drop_earlier_turns(entry_it newest);
    void evict_ram();
    void evict_disk();
    void load_disk_index();
    std::string path_of(uint64_t hash) const;
    bool write_file(const entry & e) const;
    bool read_file(const entry & e, payload & out) const;
    void writer_loop();
};
