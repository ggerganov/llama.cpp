#include "llama-moe-cache.h"
#include "llama-moe-cache-policy.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-ext.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-moe-single.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

using timer = std::chrono::steady_clock;

static bool moe_cache_read_profile(std::istream & input, uint32_t n_layer, uint32_t n_expert,
        std::vector<std::pair<int32_t, int32_t>> & ranked) {
    if (n_layer == 0 || n_layer > 65535 || n_expert == 0 || n_expert > 65535) {
        return false;
    }
    const uint64_t total = uint64_t(n_layer) * n_expert;
    if (total > (1u << 20)) {
        return false;
    }
    auto read_u32 = [&](uint32_t & value) {
        uint8_t bytes[4];
        if (!input.read(reinterpret_cast<char *>(bytes), sizeof(bytes))) {
            return false;
        }
        value = uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
        return true;
    };
    char magic[4];
    uint32_t header[5];
    if (!input.read(magic, sizeof(magic)) || std::memcmp(magic, "STRP", sizeof(magic)) != 0) {
        return false;
    }
    for (auto & value : header) {
        if (!read_u32(value)) {
            return false;
        }
    }
    if (header[0] != 1 || header[1] != n_layer || header[2] != n_expert || header[3] != total || header[4] != total) {
        return false;
    }
    std::vector<std::pair<int32_t, int32_t>> entries;
    entries.reserve(total);
    std::vector<int32_t> inverse(total, -1);
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t pair;
        if (!read_u32(pair)) {
            return false;
        }
        const uint32_t il = pair & 65535, expert = pair >> 16;
        if (il >= n_layer || expert >= n_expert) {
            return false;
        }
        const size_t key = size_t(il) * n_expert + expert;
        if (inverse[key] >= 0) {
            return false;
        }
        inverse[key] = i;
        entries.emplace_back(il, expert);
    }
    for (size_t i = 0; i < total; ++i) {
        uint32_t value;
        if (!read_u32(value) || value != uint32_t(inverse[i])) {
            return false;
        }
    }
    if (input.peek() != std::char_traits<char>::eof() || input.bad()) {
        return false;
    }
    ranked.swap(entries);
    return true;
}

static ggml_context_ptr moe_cpu_context() {
    return ggml_context_ptr(ggml_init({256*1024, nullptr, true}));
}

struct moe_cpu_worker {
    std::mutex mutex;
    std::condition_variable changed;
    std::function<void()> job;
    std::atomic<bool> ready{false};
    bool started = false;
    bool stopping = false;
    bool busy = false;
    std::exception_ptr error;
    std::thread thread;

    explicit moe_cpu_worker(int32_t first_cpu) : thread([this, first_cpu] {
        std::exception_ptr startup_error;
        try {
#ifdef __linux__
            if (first_cpu >= 0) {
                cpu_set_t cpus;
                CPU_ZERO(&cpus);
                CPU_SET(first_cpu, &cpus);
                if (sched_setaffinity(0, sizeof(cpus), &cpus)) throw std::runtime_error("MoE CPU affinity failed");
            }
#endif
        } catch (...) {
            startup_error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            error = startup_error;
            started = true;
            changed.notify_all();
        }
        if (startup_error) return;
        const bool spin = std::getenv("LLAMA_MOE_WORKER_SPIN") != nullptr;
        for (;;) {
            if (spin) {
                const auto deadline=timer::now()+std::chrono::milliseconds(20);
                while (!ready.load(std::memory_order_acquire) && timer::now()<deadline) {
#if defined(__x86_64__)
                    _mm_pause();
#else
                    std::this_thread::yield();
#endif
                }
            }
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [this] { return stopping || bool(job); });
            if (stopping) break;
            auto work = std::move(job);
            ready.store(false,std::memory_order_relaxed);
            lock.unlock();
            std::exception_ptr failure;
            try { work(); } catch (...) { failure = std::current_exception(); }
            lock.lock();
            error = failure;
            busy = false;
            changed.notify_all();
        }
    }) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this] { return started; });
        if (error) {
            const auto failure = error;
            lock.unlock();
            thread.join();
            std::rethrow_exception(failure);
        }
    }

    void submit(std::function<void()> work) {
        std::lock_guard<std::mutex> lock(mutex);
        if (busy) throw std::runtime_error("worker still owns input");
        busy = true;
        error = nullptr;
        job = std::move(work);
        ready.store(true,std::memory_order_release);
        changed.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this] { return !busy; });
        if (error) std::rethrow_exception(error);
    }

    ~moe_cpu_worker() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [this] { return !busy; });
            stopping = true;
            ready.store(true,std::memory_order_release);
            changed.notify_all();
        }
        thread.join();
    }
};

static std::vector<int32_t> moe_cpu_mask(const char * text) {
    std::vector<int32_t> cpus;
    if (!text) return cpus;
    if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) text += 2;
    const size_t size = std::strlen(text);
    if (!size || size > GGML_MAX_N_THREADS/4) throw std::runtime_error("invalid MoE CPU mask width");
    for (size_t i = 0; i < size; ++i) {
        const char c = text[size - 1 - i];
        const int nibble = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (nibble < 0) throw std::runtime_error("invalid MoE CPU mask");
        for (int bit = 0; bit < 4; ++bit) if (nibble & (1 << bit)) cpus.push_back(int32_t(4*i + bit));
    }
    if (cpus.empty()) throw std::runtime_error("empty MoE CPU mask");
    return cpus;
}

struct moe_cpu_runtime {
    using traits_fn = const ggml_type_traits_cpu * (*)(ggml_type);
    traits_fn traits = nullptr;
    ggml_backend_ptr backend;
    ggml_threadpool_t pool = nullptr;
    decltype(ggml_threadpool_free) * free_pool = nullptr;

    moe_cpu_runtime(int threads, const std::vector<int32_t> & cpus, ggml_sched_priority priority) : backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)) {
        if (!backend) throw std::runtime_error("no CPU backend");
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto make = reinterpret_cast<decltype(ggml_threadpool_new) *>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
        free_pool = reinterpret_cast<decltype(ggml_threadpool_free) *>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
        auto set_pool = reinterpret_cast<decltype(ggml_backend_cpu_set_threadpool) *>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (!make || !free_pool || !set_pool || !set_threads) throw std::runtime_error("CPU threadpool API missing");
        auto p = ggml_threadpool_params_default(threads);
        p.poll = 50;
        p.paused = true;
        p.strict_cpu = true;
        p.prio = priority;
        if (!cpus.empty() && cpus.size() < size_t(threads)) throw std::runtime_error("MoE CPU mask has too few CPUs");
        for (int32_t cpu : cpus) {
            if (cpu < 0 || cpu >= GGML_MAX_N_THREADS) throw std::runtime_error("invalid MoE CPU ID");
            p.cpumask[cpu] = true;
        }
        traits = reinterpret_cast<traits_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu"));
        if (!traits) throw std::runtime_error("CPU dot trait API missing");
        pool = make(&p);
        if (!pool) throw std::runtime_error("CPU pool allocation failed");
        set_threads(backend.get(), threads);
        set_pool(backend.get(), pool);
    }

    ~moe_cpu_runtime() { if (pool) free_pool(pool); }
};

struct moe_cpu_graph {
    ggml_backend_t backend;
    ggml_context_ptr ctx = moe_cpu_context();
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * x;
    ggml_tensor * xq;
    ggml_tensor * ids;
    ggml_tensor * out;
    ggml_tensor * gate_up_node;
    ggml_backend_graph_plan_t plan;
    const ggml_type_traits_cpu * up_traits;
    const ggml_type_traits_cpu * gate_traits;
    const ggml_type_traits_cpu * down_traits;
    ggml_from_float_t quant_x;
    ggml_from_float_t quant_hidden;

    using traits_fn = const ggml_type_traits_cpu * (*)(ggml_type);
    moe_cpu_graph(ggml_backend_t b, ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down, int misses, traits_fn traits) : backend(b) {
        up_traits = traits(up->type);
        gate_traits = traits(gate->type);
        down_traits = traits(down->type);
        if (!up_traits->vec_dot || !gate_traits->vec_dot || !down_traits->vec_dot || up_traits->vec_dot_type != gate_traits->vec_dot_type) throw std::runtime_error("unsupported native row dots");
        quant_x = traits(up_traits->vec_dot_type)->from_float;
        quant_hidden = traits(down_traits->vec_dot_type)->from_float;
        if (!quant_x || !quant_hidden || up->ne[0] % ggml_blck_size(up_traits->vec_dot_type) || down->ne[0] % ggml_blck_size(down_traits->vec_dot_type)) throw std::runtime_error("unsupported native activation quantization");
        x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, up->ne[0], 1, 1);
        xq = ggml_new_tensor_1d(ctx.get(), up_traits->vec_dot_type, up->ne[0]);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, misses, 1);
        ggml_tensor * gu_args[] = {up,gate,xq,ids};
        auto gu = ggml_custom_4d(ctx.get(),GGML_TYPE_F32,up->ne[1],misses,1,1,gu_args,4,gate_up,GGML_N_TASKS_MAX,this);
        gate_up_node = gu;
        ggml_tensor * q_args[] = {gu};
        auto hidden = ggml_custom_4d(ctx.get(),down_traits->vec_dot_type,up->ne[1],misses,1,1,q_args,1,quantize,GGML_N_TASKS_MAX,this);
        ggml_tensor * down_args[] = {down,hidden,ids};
        out = ggml_custom_4d(ctx.get(),GGML_TYPE_F32,down->ne[1],misses,1,1,down_args,3,project_down,GGML_N_TASKS_MAX,this);
        auto graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph,out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));
        if (!buffer) throw std::runtime_error("native row graph allocation failed");
        plan = ggml_backend_graph_plan_create(backend,graph);
        if (!plan) throw std::runtime_error("native row graph plan failed");
    }

    static void gate_up(ggml_tensor * dst,int ith,int nth,void * userdata) {
        auto & self = *static_cast<moe_cpu_graph *>(userdata);
        const auto * up = dst->src[0];
        const auto * gate = dst->src[1];
        const auto * input = dst->src[2];
        const auto * ids = static_cast<const int32_t *>(dst->src[3]->data);
        const int64_t n = ggml_nelements(dst);
        const int64_t first = n*ith/nth, last = n*(ith+1)/nth;
        auto * output = static_cast<float *>(dst->data);
        for (int64_t i=first;i<last;++i) {
            const int64_t e = ids[i/dst->ne[0]], r = i%dst->ne[0];
            const auto * u = static_cast<const uint8_t *>(up->data)+e*up->nb[2]+r*up->nb[1];
            const auto * g = static_cast<const uint8_t *>(gate->data)+e*gate->nb[2]+r*gate->nb[1];
            float uv=0,gv=0;
            self.up_traits->vec_dot(up->ne[0],&uv,0,u,0,input->data,0,1);
            self.gate_traits->vec_dot(gate->ne[0],&gv,0,g,0,input->data,0,1);
            output[i] = (gv/(1.0f+std::exp(-gv)))*uv;
        }
    }

    static void quantize(ggml_tensor * dst,int ith,int nth,void * userdata) {
        auto & self = *static_cast<moe_cpu_graph *>(userdata);
        const auto * input = dst->src[0];

        for (int64_t e=ith;e<dst->ne[1];e+=nth) self.quant_hidden(static_cast<const float *>(input->data)+e*input->ne[0],static_cast<uint8_t *>(dst->data)+e*dst->nb[1],dst->ne[0]);
    }

    static void project_down(ggml_tensor * dst,int ith,int nth,void * userdata) {
        auto & self = *static_cast<moe_cpu_graph *>(userdata);
        const auto * weights = dst->src[0];
        const auto * input = dst->src[1];
        const auto * ids = static_cast<const int32_t *>(dst->src[2]->data);
        const int64_t n = ggml_nelements(dst);
        const int64_t first = n*ith/nth, last = n*(ith+1)/nth;
        for (int64_t i=first;i<last;++i) {
            const int64_t e = i/dst->ne[0], r=i%dst->ne[0];
            const auto * w = static_cast<const uint8_t *>(weights->data)+ids[e]*weights->nb[2]+r*weights->nb[1];
            const auto * x = static_cast<const uint8_t *>(input->data)+e*input->nb[1];
            self.down_traits->vec_dot(weights->ne[0],static_cast<float *>(dst->data)+i,0,w,0,x,0,1);
        }
    }

    void set_weights(ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) {
        gate_up_node->src[0] = up;
        gate_up_node->src[1] = gate;
        out->src[0] = down;
    }

    ~moe_cpu_graph() { ggml_backend_graph_plan_free(backend,plan); }
};

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
        llama_moe_cache_policy policy;
        std::vector<std::unique_ptr<moe_cpu_graph>> cpu_plans;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * cpu_result = nullptr;
        ggml_tensor * cpu_input = nullptr;
        bool cpu_supported = false;
        bool cpu_graph_allowed = false;
        ggml_moe_single_layer * single = nullptr;
        uint64_t single_calls = 0;
        uint64_t single_misses = 0;
        uint64_t observed_calls = 0;
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    const bool experimental = std::getenv("LLAMA_MOE_STATIC_CACHE") != nullptr;
    const bool hybrid = std::getenv("LLAMA_MOE_HYBRID") != nullptr;
    bool frozen = false;
    const bool single_requested = std::getenv("LLAMA_MOE_SINGLE_GRAPH") != nullptr;
    const ggml_moe_single_api * single_api = nullptr;
    bool single_supported = false;
    bool single_active = false;
    bool single_canceled = false;
    bool poisoned = false;
    bool adaptive_requested = false;
    const ggml_moe_single_update_api_v1 * update_api = nullptr;
    uint32_t bootstrap_calls = 0;
    uint64_t skipped_observations = 0;
    uint64_t maintenance_count = 0;
    uint64_t maintenance_bytes = 0;
    uint64_t maintenance_us = 0;
    size_t maintenance_next_group = 0;
    struct adaptive_update {
        size_t group;
        llama_moe_cache_policy::plan plan;
    };
    std::vector<adaptive_update> pending_updates;
    std::atomic<bool> single_stop{false};
    size_t copies_async = 0;
    std::unique_ptr<moe_cpu_runtime> cpu;
    std::unique_ptr<moe_cpu_worker> worker;
    std::unordered_map<const ggml_tensor *, bool> cpu_lookups;
    std::unordered_map<const ggml_tensor *, int32_t> cpu_outputs;
    std::vector<int32_t> upload_map;
    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const llama_moe_cpu_params * cpu_params) :
            experimental((!cpu_params || cpu_params->enabled) && std::getenv("LLAMA_MOE_STATIC_CACHE")),
            hybrid((!cpu_params || cpu_params->enabled) && std::getenv("LLAMA_MOE_HYBRID")),
            single_requested((!cpu_params || cpu_params->enabled) && std::getenv("LLAMA_MOE_SINGLE_GRAPH")),
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        if (single_requested && !hybrid) throw std::runtime_error("single-graph experts require hybrid mode");
        if (hybrid && !experimental) throw std::runtime_error("hybrid experts require the static cache experiment");
        if (hybrid) {
            const char * count = std::getenv("LLAMA_MOE_CPU_THREADS");
            const int threads = cpu_params ? int(cpu_params->expert_cpus.size()) : count ? std::stoi(count) : 4;
            if (threads < 1 || threads > 32) throw std::runtime_error("MoE CPU threads must be 1..32");
            const auto cpus = cpu_params ? cpu_params->expert_cpus : moe_cpu_mask(std::getenv("LLAMA_MOE_CPU_MASK"));
            cpu = std::make_unique<moe_cpu_runtime>(threads, cpus, cpu_params ? cpu_params->priority : GGML_SCHED_PRIO_NORMAL);
            worker = std::make_unique<moe_cpu_worker>(cpus.empty() ? -1 : cpus.front());
        }

        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + (experimental ? n_expert_used + 2 : 1)), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(hybrid ? 3 : 1);
            g.cpu_plans.resize(n_expert_used + 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        if (hybrid) {
            auto pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
            if (pinned) buft_host = pinned;
        }
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + (experimental ? n_expert_used + 2 : 1));
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + (experimental ? n_expert_used + 1 : 0), bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                if (hybrid) {
                    const auto & ml = model.layers[il];
                    const auto * up = ml.ffn_up_exps;
                    const auto * gate = ml.ffn_gate_exps;
                    const auto * down = ml.ffn_down_exps;
                    l.cpu_supported = up && gate && down && up->ne[0] == model.hparams.n_embd && up->ne[0] <= 65536 && up->ne[1] <= 65536 && up->ne[0] == gate->ne[0] && up->ne[1] == gate->ne[1] && down->ne[0] == up->ne[1] && down->ne[1] == up->ne[0] && cpu->traits(up->type)->vec_dot && cpu->traits(gate->type)->vec_dot && cpu->traits(down->type)->vec_dot && cpu->traits(up->type)->vec_dot_type == cpu->traits(gate->type)->vec_dot_type;
                    l.cpu_supported = l.cpu_supported && up->ne[0] % ggml_blck_size(cpu->traits(up->type)->vec_dot_type) == 0 && down->ne[0] % ggml_blck_size(cpu->traits(down->type)->vec_dot_type) == 0 && cpu->traits(cpu->traits(up->type)->vec_dot_type)->from_float && cpu->traits(cpu->traits(down->type)->vec_dot_type)->from_float;
                    if (l.cpu_supported) {
                        l.cpu_input = ggml_new_tensor_3d(ctx_host.get(), GGML_TYPE_F32, model.hparams.n_embd, 1, 1);
                        l.cpu_result = ggml_new_tensor_3d(ctx_host.get(), GGML_TYPE_F32, model.hparams.n_embd, n_expert_used, 1);
                        ggml_format_name(l.cpu_result, "moe_cache.cpu_result-%d", il);
                        cpu_outputs[l.cpu_result] = il;
                        buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_input), alignment_host) + GGML_PAD(ggml_nbytes(l.cpu_result), alignment_host);
                    }
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            for (auto & l : layers) if (l.cpu_result) std::memset(l.cpu_result->data, 0, ggml_nbytes(l.cpu_result));
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        if (single_requested && !model.hparams.no_alloc && devices.size() == 1 && std::strncmp(ggml_backend_name(devices[0].backend), "ROCm", 4) == 0) {
            auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
            auto get_api = reinterpret_cast<ggml_moe_single_get_api_t>(ggml_backend_reg_get_proc_address(reg, "ggml_moe_single_get_api"));
            single_api = get_api ? get_api() : nullptr;
            single_supported = single_api != nullptr;
            for (size_t il = 0; il < layers.size(); ++il) {
                const auto & ml = model.layers[il];
                if (llama_moe_cache_layer_experts(ml).empty()) continue;
                const auto & l = layers[il];
                single_supported &= l.ig >= 0 && l.cpu_supported && model.hparams.n_expert_used(il) == uint32_t(n_expert_used) &&
                    !ml.ffn_gate_up_exps && !ml.ffn_up_exps_b && !ml.ffn_gate_exps_b && !ml.ffn_down_exps_b &&
                    !ml.ffn_up_exps_s && !ml.ffn_gate_exps_s && !ml.ffn_down_exps_s && model.hparams.swiglu_clamp_exp[il] <= 1e-6f;
            }
            if (!single_supported) LLAMA_LOG_WARN("llama_moe_cache: single-graph admission declined\n");
            const char * adaptive = std::getenv("LLAMA_MOE_ADAPTIVE");
            adaptive_requested = single_supported && adaptive && std::strcmp(adaptive, "1") == 0 && !std::getenv("LLAMA_MOE_FREEZE");
            if (adaptive && std::strcmp(adaptive, "1") == 0 && std::getenv("LLAMA_MOE_FREEZE")) {
                LLAMA_LOG_WARN("llama_moe_cache: LLAMA_MOE_FREEZE keeps fixed residency; adaptive learning is disabled\n");
            }
            if (adaptive_requested) {
                const auto get_update = reinterpret_cast<ggml_moe_single_get_update_api_v1_t>(ggml_backend_reg_get_proc_address(reg, "ggml_moe_single_get_update_api_v1"));
                update_api = get_update ? get_update() : nullptr;
                if (update_api && update_api->apply) {
                    for (auto & g : groups) if (g.n_slots) g.policy.init(layers.size(), g.slru.n_expert);
                } else {
                    update_api = nullptr;
                    LLAMA_LOG_WARN("llama_moe_cache: backend has no checked cache update API; residency will remain static after bootstrap\n");
                }
                LLAMA_LOG_INFO("llama_moe_cache: adaptive bootstrap uses 16 completed single-token decode calls\n");
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        cancel_single_graph();
        worker.reset();
        for (auto & d : devices) ggml_backend_synchronize(d.backend);
        collect_single_stats();
        for (auto & l : layers) if (l.single) {
            single_api->destroy(l.single);
        }
        for (auto & g : groups) g.cpu_plans.clear();
        cpu.reset();
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (experimental && (frozen || std::getenv("LLAMA_MOE_FREEZE")) && n_tokens != 1) return nullptr;
        const int64_t n_slots = groups[l.ig].n_slots;

        if (n_tokens <= 0 || n_expert_used <= 0 || (n_slots < l.slot_map->ne[1] && n_tokens > n_slots/n_expert_used)) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    ggml_tensor * get_cpu_result(int32_t il, ggml_tensor * slots, bool allowed) {
        if (!slots || !hybrid || il < 0 || il >= (int32_t) layers.size()) return nullptr;
        auto * lookup = slots;
        while (lookup->op != GGML_OP_GET_ROWS && lookup->src[0]) lookup = lookup->src[0];
        const bool supported = allowed && slots->ne[0] == n_expert_used && layers[il].cpu_supported;
        cpu_lookups[lookup] = supported;
        layers[il].cpu_graph_allowed = supported;
        return supported ? layers[il].cpu_result : nullptr;
    }

    void fill_profile_if_requested() {
        const char * path = std::getenv("LLAMA_MOE_CACHE_PROFILE");
        if (!path || !single_requested || !single_supported || single_active || devices.size() != 1 || groups.empty()) {
            return;
        }
        const int32_t n_expert = groups[0].slru.n_expert;
        if (layers.empty() || layers.size() > 65535 || n_expert < 1) {
            return;
        }
        for (const auto & l : layers) {
            if (l.single || l.ig < 0 || size_t(l.ig) >= groups.size() ||
                groups[l.ig].slru.n_expert != n_expert || !groups[l.ig].slru.slot_map[&l - layers.data()]) {
                return;
            }
        }
        std::ifstream input(path, std::ios::binary);
        std::vector<std::pair<int32_t, int32_t>> ranked;
        if (!input || !moe_cache_read_profile(input, layers.size(), n_expert, ranked)) {
            LLAMA_LOG_WARN("llama_moe_cache: invalid optional expert profile, keeping warmup residency\n");
            return;
        }
        struct assignment {
            int32_t layer;
            int32_t expert;
            int32_t slot;
        };
        std::vector<assignment> pending;
        std::vector<int32_t> heads;
        for (auto & g : groups) {
            heads.push_back(g.slru.head[0]);
            g.slru.uniq.reserve(1);
        }
        size_t bytes = 0, largest = 0;
        for (const auto & entry : ranked) {
            const auto & l = layers[entry.first];
            const auto & g = groups[l.ig];
            const int32_t slot = heads[l.ig];
            if (g.slru.slot_map[entry.first][entry.second] >= 0 || slot < 0) {
                continue;
            }
            if (size_t(slot) >= g.slru.key_of.size() || g.slru.key_of[slot] != -1) {
                continue;
            }
            pending.push_back({entry.first, entry.second, slot});
            heads[l.ig] = g.slru.next[slot];
            for (const auto * w : l.experts) {
                bytes += w->nb[2];
                largest = std::max(largest, w->nb[2]);
            }
        }
        if (pending.empty()) {
            return;
        }
        const bool verify = std::getenv("LLAMA_MOE_CACHE_PROFILE_VERIFY") != nullptr;
        std::vector<uint8_t> readback(verify ? largest : 0);
        std::vector<moe_cache_slru::fill> committed;
        committed.reserve(1);
        worker->wait();
        for (const auto & fill : pending) {
            const auto & l = layers[fill.layer];
            const auto & g = groups[l.ig];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const auto * w = l.experts[ip];
                ggml_backend_tensor_set_async(devices[g.id].backend, g.banks[ip],
                    static_cast<const uint8_t *>(w->data) + size_t(fill.expert) * w->nb[2], size_t(fill.slot) * w->nb[2], w->nb[2]);
            }
        }
        for (auto & d : devices) {
            ggml_backend_synchronize(d.backend);
        }
        if (verify) {
            for (const auto & fill : pending) {
                const auto & l = layers[fill.layer];
                const auto & g = groups[l.ig];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const auto * w = l.experts[ip];
                    ggml_backend_tensor_get(g.banks[ip], readback.data(), size_t(fill.slot) * w->nb[2], w->nb[2]);
                    if (std::memcmp(readback.data(), static_cast<const uint8_t *>(w->data) + size_t(fill.expert) * w->nb[2], w->nb[2]) != 0) {
                        throw std::runtime_error("expert profile upload verification failed");
                    }
                }
            }
        }
        for (const auto & fill : pending) {
            auto & g = groups[layers[fill.layer].ig];
            if (g.slru.head[0] != fill.slot || g.slru.key_of[fill.slot] != -1 || g.slru.slot_map[fill.layer][fill.expert] >= 0) {
                throw std::runtime_error("expert profile residency changed during upload");
            }
            size_t hits;
            if (!g.slru.plan(fill.layer, &fill.expert, 1, committed, hits) || hits ||
                committed.size() != 1 || committed[0].slot != fill.slot) {
                throw std::runtime_error("expert profile empty-slot assignment failed");
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache: profile filled %zu empty slots, %zu bytes, verified = %d\n", pending.size(), bytes, int(verify));
    }

    void freeze_if_requested(bool allow_adaptive = false) {
        if (experimental && !frozen && (std::getenv("LLAMA_MOE_FREEZE") || (allow_adaptive && adaptive_requested && bootstrap_calls >= 16))) {
            for (auto & d : devices) ggml_backend_synchronize(d.backend);
            fill_profile_if_requested();
            frozen = true;

            if (adaptive_requested) LLAMA_LOG_INFO("llama_moe_cache: single-graph cache bootstrap complete\n");
            else LLAMA_LOG_INFO("llama_moe_cache: experimental residency frozen\n");
        }
    }

    void collect_single_stats() {
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            if (!l.single) continue;
            const auto * control = l.single->host;
            const uint64_t calls = control->calls;
            const uint64_t misses = control->misses;
            stats_small.hits += (calls - l.single_calls)*n_expert_used - (misses - l.single_misses);
            stats_small.misses += misses - l.single_misses;

            l.single_calls = calls;
            l.single_misses = misses;
        }
    }

    void check_single_graph(bool synchronized) {
        if (!single_requested) return;
        if (synchronized) collect_single_stats();
        if (single_canceled || poisoned) return;
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & l = layers[il];
            if (!l.single) continue;
            const auto error = __atomic_load_n(&l.single->host->error.value, __ATOMIC_ACQUIRE);
            if (error) throw std::runtime_error("single-graph MoE failure at layer " + std::to_string(il) + ", code " + std::to_string(error));
        }
    }

    void cancel_single_graph() {
        if (!single_requested) return;
        single_canceled = true;
        for (auto & l : layers) if (l.single) {
            __atomic_store_n(&l.single->host->cancel.value, 1, __ATOMIC_RELEASE);
            __atomic_store_n(l.single->host_abort, 1, __ATOMIC_RELEASE);
        }
        single_stop.store(true, std::memory_order_release);
    }

    void poison(const char * reason) noexcept {
        if (!poisoned) LLAMA_LOG_ERROR("llama_moe_cache: context failed: %s; recreate the context before retrying\n", reason);
        poisoned = true;
        cancel_single_graph();
    }

    bool service_healthy() const {
        if (poisoned || single_canceled) return false;
        for (const auto & l : layers) if (l.single) {
            const auto * m = l.single;
            if (__atomic_load_n(&m->host->error.value, __ATOMIC_ACQUIRE) ||
                    __atomic_load_n(&m->host->cancel.value, __ATOMIC_ACQUIRE) ||
                    __atomic_load_n(m->host_abort, __ATOMIC_ACQUIRE) ||
                    __atomic_load_n(&m->host->request.value, __ATOMIC_ACQUIRE) !=
                    __atomic_load_n(&m->host->completed.value, __ATOMIC_ACQUIRE)) return false;
        }
        return true;
    }

    void observe_completed(uint32_t decode_calls, bool only_decode) noexcept {
        if (!adaptive_requested || poisoned || single_canceled) return;
        if (!single_active) {
            bootstrap_calls += std::min(decode_calls, 16 - bootstrap_calls);
            return;
        }
        if (!service_healthy()) {
            poison("invalid service state after decode completion");
            return;
        }
        const bool sample = only_decode && decode_calls == 1 && update_api;
        if (sample) for (auto & g : groups) if (g.n_slots) g.policy.tick();
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            if (!l.single) continue;
            const uint64_t calls = l.single->host->calls;
            if (calls < l.observed_calls) { poison("expert observation counter regressed"); return; }
            const uint64_t delta = calls - l.observed_calls;
            l.observed_calls = calls;
            if (!delta) continue;
            if (!sample || delta != 1) { skipped_observations += delta; continue; }
            if (!groups[l.ig].policy.observe(int32_t(il), l.single->trace_ids, n_expert_used, static_cast<const int32_t *>(l.slot_map->data))) {
                poison("invalid completed expert trace");
                return;
            }
        }
    }

    bool maintenance_pending() const {
        if (!single_active || !update_api || poisoned) return false;
        for (const auto & g : groups) if (g.n_slots && g.policy.step - g.policy.considered >= 16) return true;
        return false;
    }

    ggml_status maintain(bool (*abort_callback)(void *), void * abort_data) noexcept {
        if (poisoned) return GGML_STATUS_FAILED;
        if (!maintenance_pending()) return GGML_STATUS_SUCCESS;
        bool parked = false, mutation_started = false, aborted = false, callback_failed = false;
        const auto abort_requested = [&] {
            try {
                if (abort_callback && abort_callback(abort_data)) aborted = true;
            } catch (...) {
                callback_failed = true;
                throw;
            }
            return aborted;
        };
        struct elapsed_guard {
            uint64_t & total;
            timer::time_point started = timer::now();
            ~elapsed_guard() { total += std::chrono::duration_cast<std::chrono::microseconds>(timer::now() - started).count(); }
        } elapsed{maintenance_us};
        try {
            if (abort_requested()) return GGML_STATUS_ABORTED;
            if (!service_healthy()) throw std::runtime_error("cache maintenance requires completed requests");
            pending_updates.clear();
            std::vector<ggml_moe_single_bank_write> writes;
            std::vector<ggml_moe_single_map_write> maps;
            size_t budget = 32*1024*1024, remaining = 4;
            const size_t first = maintenance_next_group % groups.size();
            maintenance_next_group = (first + 1) % groups.size();
            for (size_t offset = 0; offset < groups.size(); ++offset) {
                const size_t ig = (first + offset) % groups.size();
                auto & g = groups[ig];
                if (!g.n_slots) continue;
                size_t bytes = 0;
                for (const auto * w : g.ref) bytes += w->nb[2];
                if (!bytes) throw std::runtime_error("empty adaptive expert layout");
                auto p = g.policy.propose(g.slru.slot_map, g.slru.key_of, std::min(remaining, budget/bytes));
                if (p.moves.empty()) continue;
                for (const auto & m : p.moves) {
                    const int32_t il = m.incoming/g.slru.n_expert, expert = m.incoming%g.slru.n_expert;
                    const auto & l = layers[il];
                    if (l.ig != int32_t(ig) || m.slot < 0 || m.slot >= g.n_slots || l.experts.size() != g.banks.size()) throw std::runtime_error("invalid adaptive bank assignment");
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const auto * w = l.experts[ip];
                        auto * bank = g.banks[ip];
                        if (!w->data || !llama_moe_cache_is_host_weight(w) || !ggml_is_contiguous(w) ||
                                w->nb[2] != bank->nb[2] || expert >= w->ne[2] || bank->ne[2] <= g.n_slots) throw std::runtime_error("invalid adaptive weight layout");
                        writes.push_back({bank, static_cast<const uint8_t *>(w->data) + size_t(expert)*w->nb[2], size_t(m.slot)*w->nb[2], w->nb[2]});
                    }
                }
                remaining -= p.moves.size();
                budget -= p.moves.size()*bytes;
                pending_updates.push_back({ig, std::move(p)});
            }
            for (auto & update : pending_updates) {
                const auto & g = groups[update.group];
                for (int32_t il : g.layers) {
                    auto & values = update.plan.maps[il];
                    if (std::memcmp(values.data(), layers[il].slot_map->data, values.size()*sizeof(int32_t))) {
                        if (!layers[il].single || layers[il].single->n_expert != int32_t(values.size())) throw std::runtime_error("invalid adaptive device map");
                        maps.push_back({layers[il].single, values.data(), g.n_slots});
                    }
                }
            }
            if (writes.empty()) return GGML_STATUS_SUCCESS;
            std::function<void()> restart = [this] { single_service(); };
            if (abort_requested()) return GGML_STATUS_ABORTED;
            single_stop.store(true, std::memory_order_release);
            parked = true;
            worker->wait();
            if (!service_healthy()) throw std::runtime_error("cache service was canceled before maintenance");
            if (abort_requested()) throw std::runtime_error("cache maintenance was aborted while parked");
            if (!update_api->apply(devices[0].backend, writes.data(), writes.size(), maps.data(), maps.size(), &mutation_started)) {
                if (mutation_started) throw std::runtime_error("cache bank or map transfer failed");
                LLAMA_LOG_WARN("llama_moe_cache: checked backend declined cache update; keeping current residency\n");
                update_api = nullptr;
            } else {
                if (abort_requested()) throw std::runtime_error("cache maintenance was aborted after transfer");
                for (const auto & update : pending_updates) {
                    auto & g = groups[update.group];
                    for (int32_t il : g.layers) std::memcpy(layers[il].slot_map->data, update.plan.maps[il].data(), size_t(g.slru.n_expert)*sizeof(int32_t));
                    std::copy(update.plan.keys.begin(), update.plan.keys.end(), g.slru.key_of.begin());
                    for (const auto & m : update.plan.moves) g.slru.touch(m.slot);
                    g.policy.committed(update.plan);
                }
                ++maintenance_count;
                for (const auto & w : writes) maintenance_bytes += w.size;
            }
            if (!service_healthy()) throw std::runtime_error("cache service was canceled during maintenance");
            if (abort_requested()) throw std::runtime_error("cache maintenance was aborted before service restart");
            single_stop.store(false, std::memory_order_release);
            worker->submit(std::move(restart));
            parked = false;
            pending_updates.clear();
            return GGML_STATUS_SUCCESS;
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("llama_moe_cache: adaptive update: %s\n", e.what());
        } catch (...) {
            LLAMA_LOG_ERROR("llama_moe_cache: adaptive update failed\n");
        }
        if (callback_failed || mutation_started || parked || !service_healthy()) {
            poison("adaptive cache transaction failed");
            return aborted && !callback_failed ? GGML_STATUS_ABORTED : GGML_STATUS_FAILED;
        }
        update_api = nullptr;
        pending_updates.clear();
        LLAMA_LOG_WARN("llama_moe_cache: adaptive preflight failed; keeping current residency\n");
        return GGML_STATUS_SUCCESS;
    }

    void single_service() {
        ggml_moe_single_layer * current = nullptr;
        try {
            while (!single_stop.load(std::memory_order_acquire)) {
                bool work = false;
                for (auto & l : layers) {
                    if (!l.single) continue;
                    auto * m = l.single;
                    auto * control = m->host;
                    if (__atomic_load_n(&control->cancel.value, __ATOMIC_ACQUIRE) || __atomic_load_n(m->host_abort, __ATOMIC_ACQUIRE)) continue;
                    const uint32_t request = __atomic_load_n(&control->request.value, __ATOMIC_ACQUIRE);
                    if (request == __atomic_load_n(&control->completed.value, __ATOMIC_RELAXED)) continue;
                    work = true;
                    current = m;
                    group & g = groups[l.ig];
                    std::vector<int32_t> positions;
                    for (int j = 0; j < n_expert_used; ++j) {
                        const int id = m->ids[j];
                        if (id < 0 || id >= m->n_expert) throw std::runtime_error("invalid mapped expert id");
                        if (static_cast<const int32_t *>(l.slot_map->data)[id] < 0) positions.push_back(j);
                    }
                    const int misses = positions.size();
                    if (!misses) throw std::runtime_error("empty mapped CPU request");

                    if (!g.cpu_plans[misses]) g.cpu_plans[misses] = std::make_unique<moe_cpu_graph>(cpu->backend.get(), l.experts[1], l.experts[0], l.experts[2], misses, cpu->traits);
                    auto * plan = g.cpu_plans[misses].get();
                    plan->set_weights(l.experts[1], l.experts[0], l.experts[2]);

                    std::memcpy(plan->x->data, m->input, ggml_nbytes(plan->x));
                    for (int j = 0; j < misses; ++j) static_cast<int32_t *>(plan->ids->data)[j] = m->ids[positions[j]];

                    plan->quant_x(static_cast<const float *>(plan->x->data), plan->xq->data, plan->x->ne[0]);

                    if (ggml_backend_graph_plan_compute(cpu->backend.get(), plan->plan) != GGML_STATUS_SUCCESS) throw std::runtime_error("mapped CPU graph failed");

                    for (int j = 0; j < misses; ++j) std::memcpy(m->output + positions[j]*m->n_embd, static_cast<const float *>(plan->out->data) + j*m->n_embd, m->n_embd*sizeof(float));

                    __atomic_store_n(&control->completed.value, request, __ATOMIC_RELEASE);

                }
                if (!work) {
#if defined(__x86_64__)
                    _mm_pause();
#else
                    std::this_thread::yield();
#endif
                }
            }
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("llama_moe_cache: mapped worker: %s\n", e.what());
            if (current) {
                __atomic_store_n(&current->host->error.value, GGML_MOE_SINGLE_WORKER, __ATOMIC_RELEASE);
                __atomic_store_n(current->host_abort, 1, __ATOMIC_RELEASE);
            }
        } catch (...) {
            if (current) {
                __atomic_store_n(&current->host->error.value, GGML_MOE_SINGLE_WORKER, __ATOMIC_RELEASE);
                __atomic_store_n(current->host_abort, 1, __ATOMIC_RELEASE);
            }
        }
    }

    bool prepare_single_graph(bool allowed) {
        if (poisoned) return false;
        if (!single_requested) return false;
        check_single_graph(false);
        bool enable = allowed && single_supported && (std::getenv("LLAMA_MOE_FREEZE") || (adaptive_requested && bootstrap_calls >= 16)) && !single_canceled && !poisoned;
        for (const auto & l : layers) if (l.ig >= 0 && !l.cpu_graph_allowed) enable = false;
        if (enable == single_active) return false;
        for (auto & d : devices) ggml_backend_synchronize(d.backend);
        collect_single_stats();
        if (single_active) {
            single_stop.store(true, std::memory_order_release);
            worker->wait();
            single_active = false;
        }
        if (enable) {
            worker->wait();
            freeze_if_requested(true);
            for (auto & l : layers) {
                if (l.ig < 0 || l.single) continue;
                const auto & g = groups[l.ig];
                l.single = single_api->create(devices[0].backend, l.slot_map->ne[1], n_expert_used, l.cpu_input->ne[0], g.n_slots + n_expert_used, static_cast<const int32_t *>(l.slot_map->data));
                if (!l.single) throw std::runtime_error("coherent single-graph mailbox allocation failed");
            }
            ggml_moe_single_layer * first = nullptr;
            for (auto & l : layers) if (l.single) {
                if (!first) first = l.single;
                l.single->host_abort = first->host_abort;
                l.single->device_abort = first->device_abort;
            }
            single_stop.store(false, std::memory_order_release);
            worker->submit([this] { single_service(); });
            single_active = true;
            LLAMA_LOG_INFO("llama_moe_cache: single captured graph path active\n");
        }
        return true;
    }

    bool copy_cpu_result_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        if (!hybrid || devices.size() != 1 || backend != devices[0].backend || std::strncmp(ggml_backend_name(backend), "ROCm", 4) != 0 || cpu_outputs.find(src) == cpu_outputs.end()) {
            return false;
        }
        const auto host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (!host || !src->buffer || !dst->buffer || ggml_backend_buffer_get_type(src->buffer) != host || ggml_backend_buffer_get_type(dst->buffer) != devices[0].buft || src->view_src || dst->view_src) {
            return false;
        }
        if (src->type != dst->type || !ggml_are_same_shape(src, dst) || !ggml_are_same_stride(src, dst)) throw std::runtime_error("invalid asynchronous CPU return layout");
        worker->wait();
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        ++copies_async;
        return true;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto result = cpu_outputs.find(src);
        if (result != cpu_outputs.end()) {
            worker->wait();
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
            return true;
        }
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        freeze_if_requested();
        const bool use_cpu = frozen && hybrid && cpu_lookups[lookup];
        const ggml_tensor * input = nullptr;
        if (use_cpu) {
            for (int i=0; i<n_nodes; ++i) {
                const auto * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] == get_experts(l.experts[1]) && node->src[1]->type == GGML_TYPE_F32 && ggml_nelements(node->src[1]) == ggml_nelements(l.cpu_input) && ggml_is_contiguous(node->src[1])) { input = node->src[1]; break; }
            }
            if (!input) throw std::runtime_error("hybrid cache activation is absent from split");
            if (!std::getenv("LLAMA_MOE_CONDITIONAL_INPUT")) ggml_backend_tensor_get_async(backend, input, l.cpu_input->data, 0, ggml_nbytes(input));
        }
        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (frozen) {
            if (ids.size() > (size_t) n_expert_used) throw std::runtime_error("static cache experiment supports one token");
            upload_map.assign(static_cast<int32_t *>(l.slot_map->data), static_cast<int32_t *>(l.slot_map->data) + l.slot_map->ne[1]);
            std::vector<int32_t> positions;
            for (size_t i=0; i<ids.size(); ++i) {
                if (ids[i] < 0 || ids[i] >= l.slot_map->ne[1]) throw std::runtime_error("invalid static expert id");
                if (static_cast<const int32_t *>(l.slot_map->data)[ids[i]] < 0) {
                    positions.push_back(i);
                    upload_map[ids[i]] = use_cpu ? g.n_slots + n_expert_used : g.n_slots + positions.size() - 1;
                }
            }

            if (use_cpu) {
                if (!positions.empty() && std::getenv("LLAMA_MOE_CONDITIONAL_INPUT")) {
                    ggml_backend_tensor_get_async(backend, input, l.cpu_input->data, 0, ggml_nbytes(input));
                    ggml_backend_synchronize(backend);
                }
                std::memset(l.cpu_result->data, 0, ggml_nbytes(l.cpu_result));
                if (!positions.empty()) {
                    const int misses = positions.size();
                    if (!g.cpu_plans[misses]) g.cpu_plans[misses] = std::make_unique<moe_cpu_graph>(cpu->backend.get(),l.experts[1],l.experts[0],l.experts[2],misses,cpu->traits);
                    auto * plan = g.cpu_plans[misses].get();
                    const auto routed = ids;
                    worker->submit([this,il,plan,positions,routed] {
                        const auto & layer = layers[il];
                        plan->set_weights(layer.experts[1],layer.experts[0],layer.experts[2]);
                        std::memcpy(plan->x->data,layer.cpu_input->data,ggml_nbytes(plan->x));
                        for (size_t j=0; j<positions.size(); ++j) static_cast<int32_t *>(plan->ids->data)[j] = routed[positions[j]];
                        plan->quant_x(static_cast<const float *>(plan->x->data),plan->xq->data,plan->x->ne[0]);
                        if (ggml_backend_graph_plan_compute(cpu->backend.get(),plan->plan) != GGML_STATUS_SUCCESS) throw std::runtime_error("hybrid CPU graph failed");
                        for (size_t j=0; j<positions.size(); ++j) std::memcpy(static_cast<float *>(layer.cpu_result->data)+positions[j]*plan->out->ne[0],static_cast<const float *>(plan->out->data)+j*plan->out->ne[0],plan->out->ne[0]*sizeof(float));
                    });
                }
            } else {
                for (int32_t position : positions) for (size_t ip=0; ip<l.experts.size(); ++ip) {
                    const auto * w = l.experts[ip];
                    const int64_t slot = upload_map[ids[position]];
                    ggml_backend_tensor_set_async(backend,g.banks[ip],static_cast<const uint8_t *>(w->data)+ids[position]*w->nb[2],slot*w->nb[2],w->nb[2]);
                }
            }
            stats_small.hits += ids.size() - positions.size();
            stats_small.misses += positions.size();
            if (!use_cpu) for (const auto * w : l.experts) stats_small.bytes += positions.size()*w->nb[2];
            ggml_backend_tensor_set_async(backend,dst,upload_map.data(),0,ggml_nbytes(src));
            return true;
        }
        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        if (copies_async) LLAMA_LOG_INFO("llama_moe_cache: asynchronous CPU returns = %zu\n", copies_async);
        if (adaptive_requested) LLAMA_LOG_INFO("llama_moe_cache: adaptive updates = %llu, uploaded = %llu bytes, maintenance = %llu us, skipped observations = %llu\n",
                (unsigned long long) maintenance_count, (unsigned long long) maintenance_bytes, (unsigned long long) maintenance_us, (unsigned long long) skipped_observations);
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const llama_moe_cpu_params * cpu) :
    pimpl(new impl(model, backends, bufts, size, cpu)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

bool llama_moe_cache::copy_cpu_result_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    return pimpl->copy_cpu_result_async(backend, src, dst);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

ggml_tensor * llama_moe_cache::get_cpu_result(int32_t il, ggml_tensor * slots, bool allowed) const {
    return pimpl->get_cpu_result(il, slots, allowed);
}

bool llama_moe_cache::prepare_single_graph(bool allowed) noexcept {
    try {
        return pimpl->prepare_single_graph(allowed);
    } catch (const std::exception & e) {
        pimpl->poison(e.what());
    } catch (...) {
        pimpl->poison("single-graph transition failed");
    }
    return false;
}

bool llama_moe_cache::single_graph_active() const {
    return pimpl->single_active;
}

ggml_tensor * llama_moe_cache::build_single_slots(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * input, int32_t il, bool allowed) const {
    if (!pimpl->single_active || il < 0 || il >= int32_t(pimpl->layers.size()) || !pimpl->layers[il].single) return nullptr;
    if (!allowed) throw std::runtime_error("single-graph CPU eligibility changed after admission");
    if (!ggml_is_contiguous(ids)) ids = ggml_cont(ctx, ids);
    if (!ggml_is_contiguous(input)) input = ggml_cont(ctx, input);
    return pimpl->single_api->publish(ctx, ids, input, pimpl->layers[il].single);
}

ggml_tensor * llama_moe_cache::build_single_merge(ggml_context * ctx, ggml_tensor * gpu, ggml_tensor * slots, int32_t il) const {
    return pimpl->single_api->merge(ctx, gpu, slots, pimpl->layers[il].single);
}

void llama_moe_cache::check_single_graph(bool synchronized) {
    pimpl->check_single_graph(synchronized);
}

void llama_moe_cache::cancel_single_graph() {
    pimpl->cancel_single_graph();
}

void llama_moe_cache::observe_completed(uint32_t decode_calls, bool only_decode) noexcept {
    pimpl->observe_completed(decode_calls, only_decode);
}

bool llama_moe_cache::maintenance_pending() const {
    return pimpl->maintenance_pending();
}

ggml_status llama_moe_cache::maintain(bool (*abort_callback)(void *), void * abort_data) noexcept {
    return pimpl->maintain(abort_callback, abort_data);
}

bool llama_moe_cache::failed() const {
    return pimpl->poisoned;
}

void llama_moe_cache::poison(const char * reason) noexcept {
    pimpl->poison(reason);
}
