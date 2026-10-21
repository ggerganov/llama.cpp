#include "server-lmcache.h"

#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace {

// file: magic, n_tokens (u64), state size (u64), n_checkpoints (u32), tokens, state,
//       then per checkpoint: n_tokens (i64), pos_min (i32), pos_max (i32), size (u64), data
constexpr char     LMCACHE_MAGIC[4] = { 'L', 'M', 'C', '2' };
constexpr uint64_t FNV_OFFSET = 1469598103934665603ULL;
constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

uint64_t hash_tokens(const llama_token * tokens, size_t n, uint64_t h = FNV_OFFSET) {
    const auto * p = reinterpret_cast<const uint8_t *>(tokens);
    for (size_t i = 0; i < n*sizeof(llama_token); ++i) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

bool is_prefix(const llama_tokens & a, const llama_tokens & b) { // a is a strict prefix of b
    return a.size() < b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <typename T> void put(std::ofstream & out, const T & v) {
    out.write((const char *) &v, sizeof(v));
}

template <typename T> bool get(std::ifstream & in, T & v) {
    in.read((char *) &v, sizeof(v));
    return (bool) in;
}

} // namespace

size_t server_lmcache::payload::bytes() const {
    size_t res = state.size();
    for (const auto & c : checkpoints) {
        res += c.size();
    }
    return res;
}

server_lmcache::server_lmcache(const common_params_lmcache & p, bool recurrent) : params(p), recurrent(recurrent) {
    params.chunk_size      = std::max(params.chunk_size, 16);
    params.max_checkpoints = std::max(params.max_checkpoints, 0);
    if (params.disk_enabled && !params.disk_dir.empty()) {
        std::error_code ec;
        fs::create_directories(params.disk_dir, ec);
        load_disk_index();
        writer = std::thread(&server_lmcache::writer_loop, this);
    }
    LOG_INF("%s: LMCache on: chunk %d tokens, RAM %d MiB, %d checkpoints per entry, disk %s (%zu entries found, written after %d s without requests)\n",
            __func__, params.chunk_size, params.ram_mb, params.max_checkpoints, params.disk_enabled ? params.disk_dir.c_str() : "off",
            entries.size(), params.disk_idle_sec);
}

server_lmcache::~server_lmcache() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    if (writer.joinable()) {
        writer.join();
    }
}

std::vector<uint64_t> server_lmcache::chunk_keys(const llama_tokens & tokens) const {
    const size_t n_chunks = tokens.size() / (size_t) params.chunk_size;
    std::vector<uint64_t> keys(n_chunks);
    uint64_t h = FNV_OFFSET;
    for (size_t c = 0; c < n_chunks; ++c) {
        h = hash_tokens(tokens.data() + c*(size_t) params.chunk_size, params.chunk_size, h);
        keys[c] = h;
    }
    return keys;
}

// where the server resumes when the prompt shares n_common tokens with entry e: everything shared when the entry is a whole prefix
// (or the model can cut its state back), else the position after the latest checkpoint inside the shared part
size_t server_lmcache::resume_point(const entry & e, size_t n_common) const {
    if (n_common >= e.tokens.size() || !recurrent) {
        return n_common;
    }
    size_t best = 0;
    for (int32_t pos : e.ckpt_pos) {
        if (pos >= 0 && (size_t) pos < n_common) {
            best = std::max(best, (size_t) pos + 1);
        }
    }
    return best;
}

std::string server_lmcache::path_of(uint64_t hash) const {
    char name[32];
    snprintf(name, sizeof(name), "%016" PRIx64 ".bin", hash);
    return (fs::path(params.disk_dir) / name).string();
}

void server_lmcache::save(llama_context * ctx, llama_seq_id seq, const llama_tokens & tokens, const std::list<common_prompt_checkpoint> & checkpoints) {
    const size_t n_chunks = tokens.size() / (size_t) params.chunk_size;
    if (n_chunks == 0) {
        return;
    }
    const uint64_t hash = hash_tokens(tokens.data(), tokens.size());
    std::vector<uint64_t> keys = chunk_keys(tokens);
    {
        std::lock_guard<std::mutex> lock(mutex);
        last_activity = std::chrono::steady_clock::now();
        auto range = index.equal_range(keys.back());
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second->hash == hash && it->second->tokens == tokens) {
                entries.splice(entries.begin(), entries, it->second); // already stored
                return;
            }
        }
    }

    auto data = std::make_shared<payload>();
    const size_t size = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (size == 0) {
        return;
    }
    data->state.resize(size);
    if (llama_state_seq_get_data_ext(ctx, data->state.data(), size, seq, LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
        return;
    }

    // the checkpoints that belong to these tokens: the earliest ones (a compacted conversation keeps the start of the prompt) and the
    // latest ones (an edited or regenerated last message)
    std::vector<const common_prompt_checkpoint *> valid;
    for (const auto & c : checkpoints) {
        if (!c.empty() && c.data_dft.empty() && c.pos_max >= 0 && (size_t) c.pos_max < tokens.size()) {
            valid.push_back(&c);
        }
    }
    std::sort(valid.begin(), valid.end(), [](const auto * a, const auto * b) { return a->pos_max < b->pos_max; });
    const size_t n_keep  = std::min(valid.size(), (size_t) params.max_checkpoints);
    const size_t n_first = (n_keep + 1) / 2;
    for (size_t i = 0; i < valid.size(); ++i) {
        if (i < n_first || i >= valid.size() - (n_keep - n_first)) {
            common_prompt_checkpoint c = *valid[i];
            c.id_task = -1; // restored later: a checkpoint that does not load falls back to processing the prompt
            c.data_spec.clear();
            data->checkpoints.push_back(std::move(c));
        }
    }

    entry e;
    e.hash    = hash;
    e.tokens  = tokens;
    e.keys    = std::move(keys);
    for (const auto & c : data->checkpoints) {
        e.ckpt_pos.push_back(c.pos_max);
    }
    e.data    = data;
    e.bytes   = data->bytes();
    e.pending = writer.joinable();

    std::lock_guard<std::mutex> lock(mutex);
    last_activity = std::chrono::steady_clock::now();
    entries.push_front(std::move(e));
    for (uint64_t k : entries.front().keys) {
        index.emplace(k, entries.begin());
    }
    ram_bytes += entries.front().bytes;
    drop_earlier_turns(entries.begin());
    evict_ram();
    LOG_INF("%s: LMCache: saved %zu tokens, %.1f MiB (%zu checkpoints)\n", __func__, tokens.size(), size/1024.0/1024.0, data->checkpoints.size());
}

bool server_lmcache::restore(llama_context * ctx, llama_seq_id seq, const llama_tokens & prompt, size_t n_have, restored & out) {
    if (prompt.empty()) {
        return false;
    }
    const std::vector<uint64_t> keys = chunk_keys(prompt);
    // at least one prompt token is evaluated, so at most prompt.size() - 1 can be reused
    const size_t n_max = prompt.size() - 1;

    entry best;
    size_t best_common = 0;
    size_t best_resume = n_have;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        last_activity = std::chrono::steady_clock::now();
        std::vector<const entry *> seen;
        // the longest shared chunk-aligned prefix first; an entry first met at c chunks shares fewer than (c + 1) chunks with the prompt
        for (size_t c = keys.size(); c >= 1; --c) {
            if (c*(size_t) params.chunk_size + (size_t) params.chunk_size <= best_resume) {
                break; // nothing below can resume further
            }
            auto range = index.equal_range(keys[c - 1]);
            for (auto it = range.first; it != range.second; ++it) {
                const entry & e = *it->second;
                if (std::find(seen.begin(), seen.end(), &e) != seen.end()) {
                    continue;
                }
                seen.push_back(&e);
                size_t n_common = 0;
                const size_t n_cmp = std::min(e.tokens.size(), prompt.size());
                while (n_common < n_cmp && e.tokens[n_common] == prompt[n_common]) {
                    n_common++;
                }
                n_common = std::min(n_common, n_max);
                const size_t resume = resume_point(e, n_common);
                if (resume > best_resume || (found && resume == best_resume && n_common > best_common)) {
                    best        = e; // a copy: the payload is shared, the writer thread may evict the entry meanwhile
                    best_common = n_common;
                    best_resume = resume;
                    found       = true;
                    entries.splice(entries.begin(), entries, it->second);
                }
            }
        }
    }
    if (!found) {
        return false;
    }

    payload from_disk;
    const payload * data = best.data.get();
    if (data == nullptr) {
        if (!read_file(best, from_disk)) {
            return false;
        }
        data = &from_disk;
    }

    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
    if (llama_state_seq_set_data_ext(ctx, data->state.data(), data->state.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        LOG_WRN("%s: LMCache: could not restore an entry of %zu tokens\n", __func__, best.tokens.size());
        return false;
    }
    out.tokens      = best.tokens;
    out.n_common    = best_common;
    out.n_resume    = best_resume;
    out.checkpoints = data->checkpoints;
    LOG_INF("%s: LMCache: restored an entry of %zu tokens, the prompt shares %zu, resuming at %zu of %zu (%zu checkpoints)%s\n", __func__,
            best.tokens.size(), best_common, best_resume, prompt.size(), out.checkpoints.size(), best.data ? "" : " (from disk)");
    return true;
}

void server_lmcache::erase(entry_it it, bool remove_file) {
    if (it->data) {
        ram_bytes -= std::min(ram_bytes, it->bytes);
    }
    if (remove_file && it->on_disk) {
        std::error_code ec;
        fs::remove(path_of(it->hash), ec);
    }
    for (uint64_t k : it->keys) {
        auto range = index.equal_range(k);
        for (auto jt = range.first; jt != range.second; ++jt) {
            if (jt->second == it) {
                index.erase(jt);
                break;
            }
        }
    }
    entries.erase(it);
}

void server_lmcache::drop_earlier_turns(entry_it newest) {
    // the earlier turns of this conversation are the entries whose tokens start it; keep the longest (the previous turn)
    std::vector<entry_it> earlier;
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (it != newest && is_prefix(it->tokens, newest->tokens)) {
            earlier.push_back(it);
        }
    }
    std::sort(earlier.begin(), earlier.end(), [](entry_it a, entry_it b) { return a->tokens.size() > b->tokens.size(); });
    for (size_t i = 1; i < earlier.size(); ++i) {
        erase(earlier[i], true);
    }
}

void server_lmcache::evict_ram() {
    const size_t max_bytes = (size_t) std::max(params.ram_mb, 0)*1024*1024;
    // the oldest entries give up their state: an entry that is on disk stays restorable from there, the others are dropped
    for (auto it = entries.end(); ram_bytes > max_bytes && it != entries.begin(); ) {
        --it;
        if (!it->data) {
            continue;
        }
        if (it->on_disk) {
            ram_bytes -= std::min(ram_bytes, it->bytes);
            it->data.reset();
            it->pending = false;
        } else {
            auto next = std::next(it);
            erase(it, false);
            it = next;
        }
    }
}

void server_lmcache::evict_disk() {
    if (!params.disk_enabled || params.disk_max_mb <= 0) {
        return;
    }
    const uintmax_t max_bytes = (uintmax_t) params.disk_max_mb*1024*1024;
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    uintmax_t total = 0;
    std::error_code ec;
    for (const auto & f : fs::directory_iterator(params.disk_dir, ec)) {
        if (f.is_regular_file() && f.path().extension() == ".bin") {
            files.emplace_back(f.last_write_time(), f.path());
            total += f.file_size();
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto & [t, p] : files) {
        if (total <= max_bytes) {
            break;
        }
        const uintmax_t sz = fs::file_size(p, ec);
        fs::remove(p, ec);
        total -= std::min(total, sz);
        // its entry is gone unless RAM still holds the state
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->on_disk && path_of(it->hash) == p.string()) {
                it->on_disk = false;
                if (!it->data) {
                    erase(it, false);
                }
                break;
            }
        }
    }
}

void server_lmcache::load_disk_index() {
    std::error_code ec;
    for (const auto & f : fs::directory_iterator(params.disk_dir, ec)) {
        if (!f.is_regular_file() || f.path().extension() != ".bin") {
            continue;
        }
        std::ifstream in(f.path(), std::ios::binary);
        char magic[4];
        uint64_t n_tokens = 0, state_size = 0;
        uint32_t n_ckpt = 0;
        in.read(magic, 4);
        if (!in || memcmp(magic, LMCACHE_MAGIC, 4) != 0 || !get(in, n_tokens) || !get(in, state_size) || !get(in, n_ckpt) ||
            n_tokens == 0 || n_tokens > (1u << 26)) {
            continue;
        }
        entry e;
        e.tokens.resize(n_tokens);
        in.read((char *) e.tokens.data(), n_tokens*sizeof(llama_token));
        if (!in) {
            continue;
        }
        e.keys = chunk_keys(e.tokens);
        if (e.keys.empty()) {
            continue; // written with a larger chunk size
        }
        // the checkpoint positions, after the state
        in.seekg((std::streamoff) state_size, std::ios::cur);
        for (uint32_t i = 0; i < n_ckpt && in; ++i) {
            int64_t n = 0;
            int32_t pmin = 0, pmax = 0;
            uint64_t size = 0;
            if (!get(in, n) || !get(in, pmin) || !get(in, pmax) || !get(in, size)) {
                break;
            }
            e.ckpt_pos.push_back(pmax);
            in.seekg((std::streamoff) size, std::ios::cur);
        }
        e.hash    = hash_tokens(e.tokens.data(), e.tokens.size());
        e.bytes   = (size_t) f.file_size(ec);
        e.on_disk = true;
        entries.push_back(std::move(e));
        for (uint64_t k : entries.back().keys) {
            index.emplace(k, std::prev(entries.end()));
        }
    }
}

bool server_lmcache::write_file(const entry & e) const {
    const std::string path = path_of(e.hash);
    const std::string tmp  = path + ".tmp";
    bool ok = true;
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(LMCACHE_MAGIC, 4);
        put(out, (uint64_t) e.tokens.size());
        put(out, (uint64_t) e.data->state.size());
        put(out, (uint32_t) e.data->checkpoints.size());
        out.write((const char *) e.tokens.data(), e.tokens.size()*sizeof(llama_token));
        out.write((const char *) e.data->state.data(), e.data->state.size());
        for (const auto & c : e.data->checkpoints) {
            put(out, (int64_t) c.n_tokens);
            put(out, (int32_t) c.pos_min);
            put(out, (int32_t) c.pos_max);
            put(out, (uint64_t) c.data_tgt.size());
            out.write((const char *) c.data_tgt.data(), c.data_tgt.size());
        }
        ok = (bool) out;
    }
    std::error_code ec;
    if (ok) {
        fs::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        fs::remove(tmp, ec);
    }
    return ok;
}

bool server_lmcache::read_file(const entry & e, payload & out) const {
    std::ifstream in(path_of(e.hash), std::ios::binary);
    char magic[4];
    uint64_t n_tokens = 0, state_size = 0;
    uint32_t n_ckpt = 0;
    in.read(magic, 4);
    if (!in || memcmp(magic, LMCACHE_MAGIC, 4) != 0 || !get(in, n_tokens) || !get(in, state_size) || !get(in, n_ckpt) ||
        n_tokens != e.tokens.size()) {
        return false;
    }
    llama_tokens tokens(n_tokens);
    in.read((char *) tokens.data(), n_tokens*sizeof(llama_token));
    if (!in || tokens != e.tokens) {
        return false;
    }
    out.state.resize(state_size);
    in.read((char *) out.state.data(), state_size);
    out.checkpoints.clear();
    for (uint32_t i = 0; i < n_ckpt && in; ++i) {
        common_prompt_checkpoint c;
        int64_t n = 0;
        int32_t pmin = 0, pmax = 0;
        uint64_t size = 0;
        if (!get(in, n) || !get(in, pmin) || !get(in, pmax) || !get(in, size)) {
            break;
        }
        c.n_tokens = n;
        c.pos_min  = pmin;
        c.pos_max  = pmax;
        c.id_task  = -1;
        c.data_tgt.resize(size);
        in.read((char *) c.data_tgt.data(), size);
        if (in) {
            out.checkpoints.push_back(std::move(c));
        }
    }
    return out.state.size() == state_size && !out.state.empty();
}

void server_lmcache::writer_loop() {
    const auto idle = std::chrono::seconds(std::max(params.disk_idle_sec, 0));
    for (;;) {
        std::vector<entry> todo;
        bool last = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::seconds(1), [&] { return stop; });
            last = stop;
            if (!last && std::chrono::steady_clock::now() - last_activity < idle) {
                continue;
            }
            for (auto & e : entries) {
                if (!e.pending || !e.data) {
                    continue;
                }
                e.pending = false;
                // an earlier turn of a conversation that a longer entry continues is not worth a write
                const bool superseded = std::any_of(entries.begin(), entries.end(), [&](const entry & f) { return is_prefix(e.tokens, f.tokens); });
                if (!superseded) {
                    todo.push_back(e); // shares the payload
                }
            }
        }
        size_t n_written = 0;
        for (const entry & e : todo) {
            if (!write_file(e)) {
                continue;
            }
            n_written++;
            std::lock_guard<std::mutex> lock(mutex);
            bool alive = false;
            for (auto & x : entries) {
                if (x.hash == e.hash && x.tokens == e.tokens) {
                    x.on_disk = true;
                    alive = true;
                }
            }
            if (!alive) {
                // the entry was dropped while it was written (a later turn replaced it)
                std::error_code ec;
                fs::remove(path_of(e.hash), ec);
            }
        }
        if (n_written > 0) {
            LOG_INF("%s: LMCache: wrote %zu entr%s to disk\n", __func__, n_written, n_written == 1 ? "y" : "ies");
            evict_disk();
        }
        if (last) {
            return;
        }
    }
}
