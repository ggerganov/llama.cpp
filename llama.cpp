// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include "llama.h"

#include "ggml.h"

#if !defined(GGML_USE_CUBLAS)
#  include "ggml-alloc.h"
#  define LLAMA_USE_ALLOCATOR
#else
#  define LLAMA_USE_SCRATCH
#  define LLAMA_MAX_SCRATCH_BUFFERS 16
#endif

#ifdef GGML_USE_CUBLAS
#  include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#  include "ggml-opencl.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif
#ifdef GGML_USE_MPI
#  include "ggml-mpi.h"
#endif
#ifdef GGML_USE_K_QUANTS
#  ifndef QK_K
#    ifdef GGML_QKK_64
#      define QK_K 64
#    else
#      define QK_K 256
#    endif
#  endif
#endif

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <io.h>
    #include <stdio.h> // for _fseeki64
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// tensor names
#define TN_TOKEN_EMBD  "token_embd.weight"
#define TN_OUTPUT_NORM "output_norm.weight"
#define TN_OUTPUT      "output.weight"
#define TN_ATTN_NORM   "blk.%d.attn_norm.weight"
#define TN_ATTN_Q      "blk.%d.attn_q.weight"
#define TN_ATTN_K      "blk.%d.attn_k.weight"
#define TN_ATTN_V      "blk.%d.attn_v.weight"
#define TN_ATTN_OUTPUT "blk.%d.attn_output.weight"
#define TN_FFN_NORM    "blk.%d.ffn_norm.weight"
#define TN_FFN_GATE    "blk.%d.ffn_gate.weight"
#define TN_FFN_DOWN    "blk.%d.ffn_down.weight"
#define TN_FFN_UP      "blk.%d.ffn_up.weight"

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//
LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal        (llama_log_level level, const char* format, ...);
static void llama_log_callback_default(llama_log_level level, const char * text, void * user_data);

#define LLAMA_LOG_INFO(...)  llama_log_internal(LLAMA_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(LLAMA_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(LLAMA_LOG_LEVEL_ERROR, __VA_ARGS__)

//
// helpers
//

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

//
// ggml helpers
//

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

//
// llama helpers
//

#ifdef GGML_USE_CUBLAS
#   define llama_host_malloc(n)  ggml_cuda_host_malloc(n)
#   define llama_host_free(data) ggml_cuda_host_free(data)
#elif GGML_USE_METAL
#   define llama_host_malloc(n)  ggml_metal_host_malloc(n)
#   define llama_host_free(data) ggml_metal_host_free(data)
#else
#   define llama_host_malloc(n)  malloc(n)
#   define llama_host_free(data) free(data)
#endif

#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

struct llama_buffer {
    void * data = NULL;
    size_t size = 0;

    // fallback to malloc / free
    // useful in cases where CUDA can try to allocate PINNED memory
    bool fallback = false;

    void resize(size_t n) {
        llama_host_free(data);

        data = llama_host_malloc(n);
        if (!data) {
            fallback = true;
            data = malloc(n);
        } else {
            fallback = false;
        }

        GGML_ASSERT(data);
        size = n;
    }

    ~llama_buffer() {
        if (data) {
            if (fallback) { // NOLINT
                free(data);
            } else {
                llama_host_free(data);
            }
        }

        data = NULL;
    }
};

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

struct llama_mmap {
    void * addr;
    size_t size;

    llama_mmap(const llama_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            // Advise the kernel to preload the mapped memory
            if (madvise(addr, std::min(file->size, prefetch), MADV_WILLNEED)) {
                fprintf(stderr, "warning: madvise(.., MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (madvise(addr, file->size, MADV_RANDOM)) {
                fprintf(stderr, "warning: madvise(.., MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }
    }

    ~llama_mmap() {
        munmap(addr, size);
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, bool prefetch = true, bool numa = false) {
        (void) numa;

        size = file->size;

        HANDLE hFile = (HANDLE) _get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        DWORD error = GetLastError();

        if (hMapping == NULL) {
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        #if _WIN32_WINNT >= _WIN32_WINNT_WIN8
        if (prefetch) {
            // Advise the kernel to preload the mapped memory
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = addr;
            range.NumberOfBytes = (SIZE_T)size;
            if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                fprintf(stderr, "warning: PrefetchVirtualMemory failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
            }
        }
        #else
        #pragma message("warning: You are building for pre-Windows 8; prefetch not supported")
        #endif // _WIN32_WINNT >= _WIN32_WINNT_WIN8
    }

    ~llama_mmap() {
        if (!UnmapViewOfFile(addr)) {
            fprintf(stderr, "warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    llama_mmap(struct llama_file * file, bool prefetch = true, bool numa = false) {
        (void) file;
        (void) prefetch;
        (void) numa;

        throw std::runtime_error(std::string("mmap not supported"));
    }
#endif
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;

    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock &) = delete;

    ~llama_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0); // NOLINT
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

#ifdef _POSIX_MEMLOCK_RANGE
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    #ifdef __APPLE__
        #define MLOCK_SUGGESTION \
            "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
            "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MLOCK (ulimit -l).\n"
    #else
        #define MLOCK_SUGGESTION \
            "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
    #endif

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);

        // Check if the resource limit is fine after all
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }

        fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    #undef MLOCK_SUGGESTION

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                fprintf(stderr, "warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            // It failed but this was only the first try; increase the working
            // set size and try again.
            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                fprintf(stderr, "warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            // Per MSDN: "The maximum number of pages that a process can lock
            // is equal to the number of pages in its minimum working set minus
            // a small overhead."
            // Hopefully a megabyte is enough overhead:
            size_t increment = len + 1048576;
            // The minimum must be <= the maximum, so we need to increase both:
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                fprintf(stderr, "warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        fprintf(stderr, "warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif
};

typedef void (*offload_func_t)(struct ggml_tensor * tensor);

static void llama_nop(struct ggml_tensor * tensor) { // don't offload by default
    (void) tensor;
}

static std::string llama_token_to_text(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_str(ctx, token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_str(ctx, token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

//
// globals
//

struct llama_state {
    // We save the log callback globally
    llama_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_state g_state;

//
// memory sizes (calculated for n_batch == 512)
//

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_3B,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
    MODEL_70B,
};

static const size_t kB = 1024;
static const size_t MB = 1024*1024;

static std::map<e_model, size_t> MEM_REQ_SCRATCH0(int n_ctx)
{
    std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   ((size_t) n_ctx / 16ull +  92ull) * MB },
        { MODEL_7B,   ((size_t) n_ctx / 16ull + 100ull) * MB },
        { MODEL_13B,  ((size_t) n_ctx / 12ull + 120ull) * MB },
        { MODEL_30B,  ((size_t) n_ctx /  9ull + 160ull) * MB },
        { MODEL_65B,  ((size_t) n_ctx /  6ull + 256ull) * MB }, // guess
        { MODEL_70B,  ((size_t) n_ctx /  7ull + 164ull) * MB },
    };
    return k_sizes;
}

static const std::map<e_model, size_t> & MEM_REQ_SCRATCH1()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,  128ull * MB },
        { MODEL_7B,  160ull * MB },
        { MODEL_13B, 192ull * MB },
        { MODEL_30B, 256ull * MB },
        { MODEL_65B, 384ull * MB }, // guess
        { MODEL_70B, 304ull * MB },
    };
    return k_sizes;
}

// used to store the compute graph tensors + non-scratch data
static const std::map<e_model, size_t> & MEM_REQ_EVAL()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   8ull * MB },
        { MODEL_7B,  10ull * MB },
        { MODEL_13B, 12ull * MB },
        { MODEL_30B, 16ull * MB },
        { MODEL_65B, 24ull * MB }, // guess
        { MODEL_70B, 24ull * MB },
    };
    return k_sizes;
}

// amount of VRAM needed per batch size to hold temporary results
// the values for 3b are not derived from testing but instead chosen conservatively
static const std::map<e_model, size_t> & VRAM_REQ_SCRATCH_BASE()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,   512ull * kB },
        { MODEL_7B,   512ull * kB },
        { MODEL_13B,  640ull * kB },
        { MODEL_30B,  768ull * kB },
        { MODEL_65B, 1280ull * kB },
        { MODEL_70B, 1280ull * kB },
    };
    return k_sizes;
}

// amount of VRAM needed per batch size and context to hold temporary results
// the values for 3b are not derived from testing but instead chosen conservatively
static const std::map<e_model, size_t> & VRAM_REQ_SCRATCH_PER_CONTEXT()
{
    static std::map<e_model, size_t> k_sizes = {
        { MODEL_3B,  128ull },
        { MODEL_7B,  128ull },
        { MODEL_13B, 160ull },
        { MODEL_30B, 208ull },
        { MODEL_65B, 256ull },
        { MODEL_70B, 256ull },
    };
    return k_sizes;
}

// default hparams (LLaMA 7B)
struct llama_hparams {
    uint32_t n_vocab     = 32000;
    uint32_t n_ctx_train = 2048;  // the context size used during training
    uint32_t n_ctx       = 512;   // the context size used during inference
    uint32_t n_embd      = 4096;
    uint32_t n_head      = 32;
    uint32_t n_head_kv   = 32;
    uint32_t n_layer     = 32;
    uint32_t n_rot       = 64;
    uint32_t n_ff        = 11008;

    float f_norm_rms_eps = 1e-5;

    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;

    bool operator!=(const llama_hparams & other) const {
        return static_cast<bool>(memcmp(this, &other, sizeof(llama_hparams))); // NOLINT
    }

    uint32_t n_gqa() const {
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }

    size_t kv_size() const {
        size_t result = 2ull;
        result *= (size_t) n_embd_gqa();
        result *= (size_t) n_ctx;
        result *= (size_t) n_layer;
        result *= sizeof(ggml_fp16_t);
        return result;
    }
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_kv_cache {
    struct ggml_tensor * k = NULL;
    struct ggml_tensor * v = NULL;

    struct ggml_context * ctx = NULL;

    llama_buffer buf;

    int n; // number of tokens currently in the cache

    ~llama_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        ggml_cuda_free_data(k);
        ggml_cuda_free_data(v);
#endif // GGML_USE_CUBLAS
    }
};

struct llama_vocab {
    // TODO:
    // - add a vector of merges
    //   so that we can pass it to different types of tokenizers with a common interface

    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = -1;
    id special_sep_id = -1;
    id special_pad_id = -1;

    id linefeed_id = 13;
};

struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    llama_hparams hparams;
    llama_vocab   vocab;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;
    int n_gpu_layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    llama_buffer buf;

    // model memory mapped file
    std::unique_ptr<llama_mmap> mapping;

    // objects representing data potentially being locked in memory
    llama_mlock mlock_buf;
    llama_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    ~llama_model() {
        if (ctx) {
            ggml_free(ctx);
        }

#ifdef GGML_USE_CUBLAS
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cuda_free_data(tensors_by_name[i].second);
        }
        ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
        for (size_t i = 0; i < tensors_by_name.size(); ++i) {
            ggml_cl_free_data(tensors_by_name[i].second);
        }
#endif
    }
};

struct llama_context {
    llama_context(const llama_model & model) : model(model), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}
    ~llama_context() {
        if (model_owner) {
            delete &model;
        }
#ifdef GGML_USE_METAL
        if (ctx_metal) {
            ggml_metal_free(ctx_metal);
        }
#endif
#ifdef LLAMA_USE_ALLOCATOR
        if (alloc) {
            ggml_allocr_free(alloc);
        }
#endif
    }

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const llama_model & model;

    bool model_owner = false;

    int64_t t_load_us;
    int64_t t_start_us;

    // key + value cache for the self attention
    struct llama_kv_cache kv_self;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    llama_buffer buf_compute;

#ifdef LLAMA_USE_ALLOCATOR
    llama_buffer buf_alloc;
    ggml_allocr * alloc = NULL;
#endif

#ifdef LLAMA_USE_SCRATCH
    llama_buffer buf_scratch[LLAMA_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[LLAMA_MAX_SCRATCH_BUFFERS] = { 0 };
#endif

#ifdef GGML_USE_METAL
    ggml_metal_context * ctx_metal = NULL;
#endif

#ifdef GGML_USE_MPI
    ggml_mpi_context * ctx_mpi = NULL;
#endif

    void use_buf(struct ggml_context * ctx, int i) { // NOLINT
#if defined(LLAMA_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.data, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) { // NOLINT
#if defined(LLAMA_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

//
// kv cache helpers
//

static bool llama_kv_cache_init(
        const struct llama_hparams & hparams,
             struct llama_kv_cache & cache,
                         ggml_type   wtype,
                               int   n_ctx,
                               int   n_gpu_layers) {
    const int n_embd  = hparams.n_embd_gqa();
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);
    cache.n = 0;

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.data;
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        LLAMA_LOG_ERROR("%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    ggml_set_name(cache.k, "cache_k");
    ggml_set_name(cache.v, "cache_v");

    (void) n_gpu_layers;
#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer + 1) {
        ggml_cuda_assign_buffers_no_scratch(cache.v);
    }
    if (n_gpu_layers > n_layer + 2) {
        ggml_cuda_assign_buffers_no_scratch(cache.k);
    }
#endif // GGML_USE_CUBLAS

    return true;
}

//
// model loading and saving
//

enum llama_file_version {
    GGUF_FILE_VERSION_V1 = 1,
};

static const char * llama_file_version_name(llama_file_version version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (latest)";
    }

    return "unknown";
}

static std::string llama_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5u", ne.at(i));
    }
    return buf;
}

static std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
    }
    return buf;
}

struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;

    bool use_mmap = false;

    llama_file file;
    llama_ftype ftype;
    llama_file_version fver;

    std::unique_ptr<llama_mmap> mapping;

    struct gguf_context * ctx_gguf = NULL;
    struct ggml_context * ctx_meta = NULL;

    llama_model_loader(const std::string & fname, bool use_mmap) : file(fname.c_str(), "rb") {
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        ctx_gguf = gguf_init_from_file(fname.c_str(), params);
        if (!ctx_gguf) {
            throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
        }

        n_kv      = gguf_get_n_kv(ctx_gguf);
        n_tensors = gguf_get_n_tensors(ctx_gguf);

        fver = (enum llama_file_version) gguf_get_version(ctx_gguf);

        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            n_elements += ggml_nelements(t);
        }

        LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
                __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

        // determine file type based on the number of tensors for each quantization and print meta data
        // TODO: make optional
        {
            std::map<enum ggml_type, uint32_t> n_type;

            uint32_t n_type_max = 0;
            enum ggml_type type_max = GGML_TYPE_F32;

            for (int i = 0; i < n_tensors; i++) {
                const char * name = gguf_get_tensor_name(ctx_gguf, i);
                struct ggml_tensor * meta = ggml_get_tensor(ctx_meta, name);

                n_type[meta->type]++;

                if (n_type_max < n_type[meta->type]) {
                    n_type_max = n_type[meta->type];
                    type_max   = meta->type;
                }

                LLAMA_LOG_INFO("%s: - tensor %4d: %32s %-8s [ %s ]\n", __func__, i, name, ggml_type_name(meta->type), llama_format_tensor_shape(meta).c_str());
            }

            switch (type_max) {
                case GGML_TYPE_F32:  ftype = LLAMA_FTYPE_ALL_F32;       break;
                case GGML_TYPE_F16:  ftype = LLAMA_FTYPE_MOSTLY_F16;    break;
                case GGML_TYPE_Q4_0: ftype = LLAMA_FTYPE_MOSTLY_Q4_0;   break;
                case GGML_TYPE_Q4_1: ftype = LLAMA_FTYPE_MOSTLY_Q4_1;   break;
                case GGML_TYPE_Q5_0: ftype = LLAMA_FTYPE_MOSTLY_Q5_0;   break;
                case GGML_TYPE_Q5_1: ftype = LLAMA_FTYPE_MOSTLY_Q5_1;   break;
                case GGML_TYPE_Q8_0: ftype = LLAMA_FTYPE_MOSTLY_Q8_0;   break;
                case GGML_TYPE_Q2_K: ftype = LLAMA_FTYPE_MOSTLY_Q2_K;   break;
                case GGML_TYPE_Q3_K: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M; break;
                case GGML_TYPE_Q4_K: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M; break;
                case GGML_TYPE_Q5_K: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M; break;
                case GGML_TYPE_Q6_K: ftype = LLAMA_FTYPE_MOSTLY_Q6_K;   break;
                default:
                     {
                         LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                         ftype = LLAMA_FTYPE_ALL_F32;
                     } break;
            }

            for (int i = 0; i < n_kv; i++) {
                const char * name         = gguf_get_key(ctx_gguf, i);
                const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

                LLAMA_LOG_INFO("%s: - kv %3d: %42s %-8s\n", __func__, i, name, gguf_type_name(type));
            }

            // print type counts
            for (auto & kv : n_type) {
                if (kv.second == 0) {
                    continue;
                }

                LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
            }
        }

        if (!llama_mmap::SUPPORTED) {
            LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
            use_mmap = false;
        }

        this->use_mmap = use_mmap;
    }

    ~llama_model_loader() {
        if (ctx_gguf) {
            gguf_free(ctx_gguf);
        }
        if (ctx_meta) {
            ggml_free(ctx_meta);
        }
    }

    const char * get_tensor_name(int i) const {
        return gguf_get_tensor_name(ctx_gguf, i);
    }

    struct ggml_tensor * get_tensor_meta(int i) const {
        return ggml_get_tensor(ctx_meta, get_tensor_name(i));
    }

    void calc_sizes(size_t & ctx_size_p, size_t & mmapped_size_p) const {
        ctx_size_p     = 0;
        mmapped_size_p = 0;

        for (int i = 0; i < n_tensors; i++) {
            struct ggml_tensor * meta = get_tensor_meta(i);
            ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            (use_mmap ? mmapped_size_p : ctx_size_p) += ggml_nbytes_pad(meta);
        }
    }

    struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, struct ggml_tensor * meta, ggml_backend backend) {
        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ctx, true);
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(ctx, meta);
        tensor->backend = backend; // TODO: ggml_set_backend
        ggml_set_name(tensor, ggml_get_name(meta));

        if (backend != GGML_BACKEND_CPU) {
            ggml_set_no_alloc(ctx, use_mmap);
        }

        n_created++;

        return tensor;
    }

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<uint32_t> & ne, ggml_backend backend) {
        struct ggml_tensor * cur = ggml_get_tensor(ctx_meta, name.c_str());

        if (cur == NULL) {
            throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
        }

        {
            bool is_ok = true;
            for (size_t i = 0; i < ne.size(); ++i) {
                if (ne[i] != cur->ne[i]) {
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok) {
                throw std::runtime_error(
                        format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                            __func__, name.c_str(),
                            llama_format_tensor_shape(ne).c_str(),
                            llama_format_tensor_shape(cur).c_str()));
            }
        }

        return create_tensor_for(ctx, cur, backend);
    }

    void done_getting_tensors() const {
        if (n_created != n_tensors) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
    }

    size_t file_offset(const char * name) const {
        const int idx = gguf_find_tensor(ctx_gguf, name);

        if (idx < 0) {
            throw std::runtime_error(format("%s: tensor '%s' not found in the file", __func__, name));
        }

        return gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
    }

    void load_data_for(struct ggml_tensor * cur) const {
        const size_t offs = file_offset(ggml_get_name(cur));

        if (use_mmap) {
            cur->data = (uint8_t *) mapping->addr + offs;
        } else {
            file.seek(offs, SEEK_SET);
            file.read_raw(cur->data, ggml_nbytes(cur));
        }
    }

    void load_all_data(struct ggml_context * ctx, llama_progress_callback progress_callback, void * progress_callback_user_data, llama_mlock * lmlock) {
        size_t size_data = 0;
        size_t size_lock = 0;
        size_t size_pref = 0; // prefetch

        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
            struct ggml_tensor * cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
            size_data += ggml_nbytes(cur);
            if (cur->backend == GGML_BACKEND_CPU) {
                size_pref += ggml_nbytes(cur);
            }
        }

        if (use_mmap) {
            mapping.reset(new llama_mmap(&file, size_pref, ggml_is_numa()));
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
            struct ggml_tensor * cur = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_gguf, i));
            GGML_ASSERT(cur); // unused tensors should have been caught by load_data already

            if (progress_callback) {
                progress_callback((float) done_size / size_data, progress_callback_user_data);
            }

            // allocate temp buffer if not using mmap
            if (!use_mmap && cur->data == NULL) {
                GGML_ASSERT(cur->backend != GGML_BACKEND_CPU);
                cur->data = malloc(ggml_nbytes(cur));
            }

            load_data_for(cur);

            switch (cur->backend) {
                case GGML_BACKEND_CPU:
                    if (use_mmap && lmlock) {
                        size_lock += ggml_nbytes(cur);
                        lmlock->grow_to(size_lock);
                    }
                    break;
#if defined(GGML_USE_CUBLAS)
                case GGML_BACKEND_GPU:
                case GGML_BACKEND_GPU_SPLIT:
                    // old code:
                    //ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);

                    // TODO: test if this works !!
                    ggml_cuda_transform_tensor(cur->data, cur);
                    if (!use_mmap) {
                        free(cur->data);
                    }
                    break;
#elif defined(GGML_USE_CLBLAST)
                case GGML_BACKEND_GPU:
                    ggml_cl_transform_tensor(cur->data, cur);
                    if (!use_mmap) {
                        free(cur->data);
                    }
                    break;
#endif
                default:
                    continue;
            }

            done_size += ggml_nbytes(cur);
        }
    }
};

//
// load LLaMA models
//

const char * llama_model_ftype_name(enum llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:     return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:  return "mostly F16";
        case LLAMA_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case LLAMA_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K:   return "mostly Q2_K";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S: return "mostly Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M: return "mostly Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return "mostly Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S: return "mostly Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "mostly Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S: return "mostly Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return "mostly Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:   return "mostly Q6_K";

        default: return "unknown, may not work";
    }
}

static const char * llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_3B:  return "3B";
        case MODEL_7B:  return "7B";
        case MODEL_13B: return "13B";
        case MODEL_30B: return "30B";
        case MODEL_65B: return "65B";
        case MODEL_70B: return "70B";
        default: GGML_ASSERT(false);
    }
}

static void llama_model_load_internal(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    model.t_start_us = ggml_time_us();

    std::unique_ptr<llama_model_loader> ml(new llama_model_loader(fname, use_mmap));

    model.n_gpu_layers = n_gpu_layers;

    auto & hparams = model.hparams;

    std::string general_name = "n/a";
    std::string general_arch = "n/a";

    // read hparams
    {
        struct gguf_context * ctx = ml->ctx_gguf;

#define GGUF_GET(dst, func, type, req, key) \
        { \
            const int kid = gguf_find_key(ctx, key); \
            if (kid >= 0) { \
                enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
                if (ktype != (type)) { \
                    throw std::runtime_error(format("key %s has wrong type: %s", key, gguf_type_name(ktype))); \
                } \
                (dst) = func(ctx, kid); \
            } else if (req) { \
                throw std::runtime_error(format("key not found in model: %s", key)); \
            } \
        }

        std::string tokenizer_name;
        GGUF_GET(tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, "tokenizer.ggml.model");

        if (tokenizer_name == "llama") {
            vocab.type = LLAMA_VOCAB_TYPE_SPM;
        } else if (tokenizer_name == "gpt2") {
            vocab.type = LLAMA_VOCAB_TYPE_BPE;
        } else {
            LLAMA_LOG_WARN("%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
            LLAMA_LOG_WARN("%s: using default tokenizer: 'llama'", __func__);
            vocab.type = LLAMA_VOCAB_TYPE_SPM;
        }

        // get hparams kv
        GGUF_GET(hparams.n_vocab,        gguf_get_arr_n,   GGUF_TYPE_ARRAY,   true, "tokenizer.ggml.tokens");
        GGUF_GET(hparams.n_ctx_train,    gguf_get_val_u32, GGUF_TYPE_UINT32,  true, "llama.context_length");
        GGUF_GET(hparams.n_embd,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, "llama.embedding_length");
        GGUF_GET(hparams.n_ff,           gguf_get_val_u32, GGUF_TYPE_UINT32,  true, "llama.feed_forward_length");
        GGUF_GET(hparams.n_head,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, "llama.attention.head_count");
        GGUF_GET(hparams.n_layer,        gguf_get_val_u32, GGUF_TYPE_UINT32,  true, "llama.block_count");
        GGUF_GET(hparams.n_rot,          gguf_get_val_u32, GGUF_TYPE_UINT32,  true, "llama.rope.dimension_count");
        GGUF_GET(hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, true, "llama.attention.layer_norm_rms_epsilon");

        // n_head_kv is optional, default to n_head
        hparams.n_head_kv = hparams.n_head;
        GGUF_GET(hparams.n_head_kv, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "llama.attention.head_count_kv");

        // TODO: manually setting rope scale should override this
        // rope_freq_scale (inverse of the kv) is optional
        float ropescale = 1.0f;
        GGUF_GET(ropescale, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, "llama.rope.scale_linear");
        if (ropescale != 1.0f) {
            rope_freq_scale = 1.0f/ropescale;
        }

        // get general kv
        GGUF_GET(general_name, gguf_get_val_str, GGUF_TYPE_STRING, false, "general.name");
        GGUF_GET(general_arch, gguf_get_val_str, GGUF_TYPE_STRING, false, "general.architecture");

        // special tokens
        GGUF_GET(vocab.special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "tokenizer.ggml.bos_token_id");
        GGUF_GET(vocab.special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "tokenizer.ggml.eos_token_id");
        GGUF_GET(vocab.special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "tokenizer.ggml.unknown_token_id");
        GGUF_GET(vocab.special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "tokenizer.ggml.separator_token_id");
        GGUF_GET(vocab.special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, "tokenizer.ggml.padding_token_id");

#undef GGUF_GET

        switch (hparams.n_layer) {
            case 26: model.type = e_model::MODEL_3B; break;
            case 32: model.type = e_model::MODEL_7B; break;
            case 40: model.type = e_model::MODEL_13B; break;
            case 60: model.type = e_model::MODEL_30B; break;
            case 80: model.type = e_model::MODEL_65B; break;
            default:
                {
                    if (hparams.n_layer < 32) {
                        model.type = e_model::MODEL_7B;
                    }
                } break;
        }

        model.ftype = ml->ftype;

        hparams.n_ctx = n_ctx;

        // LLaMAv2
        // TODO: probably not needed
        {
            const auto n_gqa = hparams.n_gqa();

            if (model.type == e_model::MODEL_65B && n_gqa == 8) {
                LLAMA_LOG_WARN("%s: assuming 70B model based on GQA == %d\n", __func__, n_gqa);
                model.type = e_model::MODEL_70B;
            }
        }

        hparams.rope_freq_base  = rope_freq_base;
        hparams.rope_freq_scale = rope_freq_scale;
    }

    // read vocab
    {
        struct gguf_context * ctx = ml->ctx_gguf;

        vocab.id_to_token.resize(hparams.n_vocab);

        const int token_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
        if (token_idx == -1) {
            throw std::runtime_error("cannot find tokenizer vocab in model file\n");
        }

        const int score_idx = gguf_find_key(ctx, "tokenizer.ggml.scores");
        if (score_idx == -1) {
            throw std::runtime_error("cannot find tokenizer scores in model file\n");
        }

        const float * scores = (const float * ) gguf_get_arr_data(ctx, score_idx);

        const int toktype_idx = gguf_find_key(ctx, "tokenizer.ggml.token_type");
        if (toktype_idx == -1) {
            throw std::runtime_error("cannot find token type list in GGUF file\n");
        }

        const int * toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {
            std::string word = gguf_get_arr_str(ctx, token_idx, i);

            vocab.token_to_id[word] = i;

            auto & token_data = vocab.id_to_token[i];
            token_data.text  = std::move(word);
            token_data.score = scores[i];
            token_data.type  = (llama_token_type) toktypes[i];

            // determine the newline token: 0x0A == 10 == '\n'
            if (token_data.text == "<0x0A>") {
                vocab.linefeed_id = i;
            }
        }
    }

    {
        // hparams
        LLAMA_LOG_INFO("%s: format       = %s\n",     __func__, llama_file_version_name(ml->fver));
        LLAMA_LOG_INFO("%s: arch         = %s\n",     __func__, general_arch.c_str());
        LLAMA_LOG_INFO("%s: vocab type   = %s\n",     __func__, vocab.type == LLAMA_VOCAB_TYPE_SPM ? "SPM" : "BPE"); // TODO: fix
        LLAMA_LOG_INFO("%s: n_vocab      = %u\n",     __func__, hparams.n_vocab);
        LLAMA_LOG_INFO("%s: n_ctx_train  = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_ctx        = %u\n",     __func__, hparams.n_ctx);
        LLAMA_LOG_INFO("%s: n_embd       = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_head       = %u\n",     __func__, hparams.n_head);
        LLAMA_LOG_INFO("%s: n_head_kv    = %u\n",     __func__, hparams.n_head_kv);
        LLAMA_LOG_INFO("%s: n_layer      = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_rot        = %u\n",     __func__, hparams.n_rot); // a.k.a. n_embd_head, n_head_dim
        LLAMA_LOG_INFO("%s: n_gqa        = %u\n",     __func__, hparams.n_gqa());
        LLAMA_LOG_INFO("%s: f_norm_eps   = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: n_ff         = %u\n",     __func__, hparams.n_ff);
        LLAMA_LOG_INFO("%s: freq_base    = %.1f\n",   __func__, hparams.rope_freq_base);
        LLAMA_LOG_INFO("%s: freq_scale   = %g\n",     __func__, hparams.rope_freq_scale);
        LLAMA_LOG_INFO("%s: model type   = %s\n",     __func__, llama_model_type_name(model.type));
        LLAMA_LOG_INFO("%s: model ftype  = %s\n",     __func__, llama_model_ftype_name(model.ftype));
        LLAMA_LOG_INFO("%s: model size   = %.2f B\n", __func__, ml->n_elements*1e-9);

        // general kv
        LLAMA_LOG_INFO("%s: general.name = %s\n",    __func__, general_name.c_str());

        // special tokens
        if (vocab.special_bos_id != -1) { LLAMA_LOG_INFO( "%s: BOS token = %d '%s'\n", __func__, vocab.special_bos_id, vocab.id_to_token[vocab.special_bos_id].text.c_str() ); }
        if (vocab.special_eos_id != -1) { LLAMA_LOG_INFO( "%s: EOS token = %d '%s'\n", __func__, vocab.special_eos_id, vocab.id_to_token[vocab.special_eos_id].text.c_str() ); }
        if (vocab.special_unk_id != -1) { LLAMA_LOG_INFO( "%s: UNK token = %d '%s'\n", __func__, vocab.special_unk_id, vocab.id_to_token[vocab.special_unk_id].text.c_str() ); }
        if (vocab.special_sep_id != -1) { LLAMA_LOG_INFO( "%s: SEP token = %d '%s'\n", __func__, vocab.special_sep_id, vocab.id_to_token[vocab.special_sep_id].text.c_str() ); }
        if (vocab.special_pad_id != -1) { LLAMA_LOG_INFO( "%s: PAD token = %d '%s'\n", __func__, vocab.special_pad_id, vocab.id_to_token[vocab.special_pad_id].text.c_str() ); }
        if (vocab.linefeed_id    != -1) { LLAMA_LOG_INFO( "%s: LF token  = %d '%s'\n", __func__, vocab.linefeed_id,    vocab.id_to_token[vocab.linefeed_id].text.c_str() );    }
    }

    if (vocab_only) {
        LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    size_t mmapped_size;

    ml->calc_sizes(ctx_size, mmapped_size);

    LLAMA_LOG_INFO("%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/1024.0/1024.0);

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init   (model.buf.data);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.data,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw std::runtime_error(format("ggml_init() failed"));
        }
    }

    (void) main_gpu;
    (void) mul_mat_q;
#if defined(GGML_USE_CUBLAS)
    LLAMA_LOG_INFO("%s: using CUDA for GPU acceleration\n", __func__);
    ggml_cuda_set_main_device(main_gpu);
    ggml_cuda_set_mul_mat_q(mul_mat_q);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU_SPLIT
#elif defined(GGML_USE_CLBLAST)
    LLAMA_LOG_INFO("%s: using OpenCL for GPU acceleration\n", __func__);
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU
#else
#define LLAMA_BACKEND_OFFLOAD       GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU
#endif

    // prepare memory for the weights
    size_t vram_weights = 0;
    size_t vram_scratch = 0;
    {
        const uint32_t n_embd     = hparams.n_embd;
        const uint32_t n_embd_gqa = hparams.n_embd_gqa();
        const uint32_t n_layer    = hparams.n_layer;
        const uint32_t n_vocab    = hparams.n_vocab;

        model.tok_embeddings = ml->create_tensor(ctx, TN_TOKEN_EMBD, {n_embd, n_vocab}, GGML_BACKEND_CPU);

        // "output" tensor
        {
            ggml_backend backend_norm;
            ggml_backend backend_output;
            if (n_gpu_layers > int(n_layer)) { // NOLINT
                // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
                // on Windows however this is detrimental unless everything is on the GPU
#ifndef _WIN32
                backend_norm = low_vram ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#else
                backend_norm = low_vram || n_gpu_layers <= (int) n_layer + 2 ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
#endif // _WIN32

                backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
            } else {
                backend_norm = GGML_BACKEND_CPU;
                backend_output = GGML_BACKEND_CPU;
            }

            model.norm   = ml->create_tensor(ctx, TN_OUTPUT_NORM, {n_embd},          backend_norm);
            model.output = ml->create_tensor(ctx, TN_OUTPUT,      {n_embd, n_vocab}, backend_output);
            if (backend_norm == GGML_BACKEND_GPU) {
                vram_weights += ggml_nbytes(model.norm);
            }
            if (backend_output == GGML_BACKEND_GPU_SPLIT) {
                vram_weights += ggml_nbytes(model.output);
            }
        }

        const uint32_t n_ff = hparams.n_ff;

        const int i_gpu_start = n_layer - n_gpu_layers;

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            const ggml_backend backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD; // NOLINT
            const ggml_backend backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT; // NOLINT

            auto & layer = model.layers[i];
            layer.attention_norm = ml->create_tensor(ctx, format(TN_ATTN_NORM, i), {n_embd}, backend);

            layer.wq = ml->create_tensor(ctx, format(TN_ATTN_Q, i),      {n_embd, n_embd},     backend_split);
            layer.wk = ml->create_tensor(ctx, format(TN_ATTN_K, i),      {n_embd, n_embd_gqa}, backend_split);
            layer.wv = ml->create_tensor(ctx, format(TN_ATTN_V, i),      {n_embd, n_embd_gqa}, backend_split);
            layer.wo = ml->create_tensor(ctx, format(TN_ATTN_OUTPUT, i), {n_embd, n_embd},     backend_split);

            layer.ffn_norm = ml->create_tensor(ctx, format(TN_FFN_NORM, i), {n_embd}, backend);

            layer.w1 = ml->create_tensor(ctx, format(TN_FFN_GATE, i), {n_embd,   n_ff}, backend_split);
            layer.w2 = ml->create_tensor(ctx, format(TN_FFN_DOWN, i), {  n_ff, n_embd}, backend_split);
            layer.w3 = ml->create_tensor(ctx, format(TN_FFN_UP, i),   {n_embd,   n_ff}, backend_split);

            if (backend == GGML_BACKEND_GPU) {
                vram_weights +=
                    ggml_nbytes(layer.attention_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk)             +
                    ggml_nbytes(layer.wv)             + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                    ggml_nbytes(layer.w1)             + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
            }
        }
    }

    ml->done_getting_tensors();

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        size_t mem_required =
            ctx_size +
            mmapped_size - vram_weights; // weights in VRAM not in memory

#ifndef LLAMA_USE_ALLOCATOR
        mem_required +=
            MEM_REQ_SCRATCH0(hparams.n_ctx).at(model.type) +
            MEM_REQ_SCRATCH1().at(model.type) +
            MEM_REQ_EVAL().at(model.type);
#endif

        // this is the memory required by one llama_state
        const size_t mem_required_state =
            scale*hparams.kv_size();

        LLAMA_LOG_INFO("%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);

        (void) vram_scratch;
        (void) n_batch;
#ifdef GGML_USE_CUBLAS
        if (low_vram) {
            LLAMA_LOG_INFO("%s: not allocating a VRAM scratch buffer due to low VRAM option\n", __func__);
            ggml_cuda_set_scratch_size(0); // disable scratch
        } else {
            const size_t vram_scratch_base = VRAM_REQ_SCRATCH_BASE().at(model.type);
            const size_t vram_scratch_per_context = VRAM_REQ_SCRATCH_PER_CONTEXT().at(model.type);
            vram_scratch = n_batch * (vram_scratch_base + n_ctx * vram_scratch_per_context);
            ggml_cuda_set_scratch_size(vram_scratch);
            if (n_gpu_layers > 0) {
                LLAMA_LOG_INFO("%s: allocating batch_size x (%zd kB + n_ctx x %zd B) = %zd MB VRAM for the scratch buffer\n",
                        __func__, vram_scratch_base / kB, vram_scratch_per_context,
                        (vram_scratch + MB - 1) / MB); // round up
            }
        }
#endif // GGML_USE_CUBLAS

#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }
        size_t vram_kv_cache = 0;

#ifdef GGML_USE_CUBLAS
        const int max_backend_supported_layers = hparams.n_layer + 3;
        const int max_offloadable_layers = low_vram ? hparams.n_layer + 1 : hparams.n_layer + 3;
        if (n_gpu_layers > (int) hparams.n_layer + 1) {
            if (low_vram) {
                LLAMA_LOG_INFO("%s: cannot offload v cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: offloading v cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
        if (n_gpu_layers > (int) hparams.n_layer + 2) {
            if (low_vram) {
                LLAMA_LOG_WARN("%s: cannot offload k cache to GPU due to low VRAM option\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: offloading k cache to GPU\n", __func__);
                vram_kv_cache += hparams.kv_size() / 2;
            }
        }
#elif defined(GGML_USE_CLBLAST)
        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers = hparams.n_layer + 1;
#endif // GGML_USE_CUBLAS

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n",
                __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
        LLAMA_LOG_INFO("%s: total VRAM used: %zu MB\n",
                __func__, (vram_weights + vram_scratch + vram_kv_cache + MB - 1) / MB); // round up
#else
        (void) n_gpu_layers;
#endif // defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    }

    // populate `tensors_by_name`
    for (int i = 0; i < ml->n_tensors; ++i) {
        struct ggml_tensor * cur = ggml_get_tensor(ctx, ml->get_tensor_name(i));
        model.tensors_by_name.emplace_back(ggml_get_name(cur), cur);
    }

    (void) tensor_split;
#if defined(GGML_USE_CUBLAS)
    {
        ggml_cuda_set_tensor_split(tensor_split);
    }
#endif

    ml->load_all_data(ctx, progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    if (progress_callback) {
        progress_callback(1.0f, progress_callback_user_data);
    }

    model.mapping = std::move(ml->mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_time_us() - model.t_start_us;
}

static bool llama_model_load(
        const std::string & fname,
        llama_model & model,
        llama_vocab & vocab,
        int n_ctx,
        int n_batch,
        int n_gpu_layers,
        int main_gpu,
        const float * tensor_split,
        const bool mul_mat_q,
        float rope_freq_base,
        float rope_freq_scale,
        bool low_vram,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        llama_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        llama_model_load_internal(fname, model, vocab, n_ctx, n_batch, n_gpu_layers,
                                  main_gpu, tensor_split, mul_mat_q, rope_freq_base, rope_freq_scale, low_vram, memory_type,
                                  use_mmap, use_mlock, vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading model: %s\n", err.what());
        return false;
    }
}

static struct ggml_cgraph * llama_build_graph(
         llama_context & lctx,
     const llama_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past) {

    GGML_ASSERT((!tokens && embd) || (tokens && !embd)); // NOLINT

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = hparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base    = hparams.rope_freq_base;
    const float freq_scale   = hparams.rope_freq_scale;
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

#ifdef LLAMA_USE_ALLOCATOR
    params.no_alloc = true;
#endif

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (tokens) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);

#ifdef LLAMA_USE_ALLOCATOR
        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, tokens, N*ggml_element_size(inp_tokens));
        }
#else
        memcpy(inp_tokens->data, tokens, N*ggml_element_size(inp_tokens));
#endif
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);

#ifdef LLAMA_USE_ALLOCATOR
        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, embd, N * n_embd * ggml_element_size(inpL));
        }
#else
        memcpy(inpL->data, embd, N * n_embd * ggml_element_size(inpL));
#endif
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    //
    // with the low VRAM option VRAM scratch is disabled in llama_load_model_internal
    // in that case ggml_cuda_assign_buffers has no effect
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers;
    }
#endif // GGML_USE_CUBLAS

    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
#ifdef LLAMA_USE_ALLOCATOR
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }
#else
    ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
#endif
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");

    for (int il = 0; il < n_layer; ++il) {
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers;
        }
#endif // GGML_USE_CUBLAS

        struct ggml_tensor * inpSA = inpL;

        lctx.use_buf(ctx0, 0);

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attention_norm(broadcasted)
            cur = ggml_mul(ctx0, cur, model.layers[il].attention_norm);
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur = ggml_rope_custom_inplace(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, N), n_past, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            struct ggml_tensor * Qcur = ggml_rope_custom_inplace(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, N),    n_past, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, N));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + n_past));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + n_past*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_past + N, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

#if 1
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_tensor * V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd_head, n_head));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        lctx.use_buf(ctx0, 1);

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, norm_rms_eps);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    lctx.use_buf(ctx0, 0);

    // norm
    {
        cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
        offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model.norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //cur = ggml_soft_max_inplace(ctx0, cur);

    ggml_build_forward_expand(gf, cur);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

#if 0
    LLAMA_LOG_INFO("\n%s: used_mem: eval ctx %.3f MB, scratch %.3f MB %.3f MB, work buf %.3f MB, n_past = %d, N = %d\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0,
            lctx.work_buffer.size()/1024.0/1024.0,
            n_past, N);
#endif

    ggml_free(ctx0);

    return gf;
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - embd       embeddings input
//   - n_tokens   number of tokens
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool llama_eval_internal(
         llama_context & lctx,
     const llama_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past,
                   int   n_threads,
            const char * cgraph_fname) {

    GGML_ASSERT((!tokens && embd) || (tokens && !embd)); // NOLINT

    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(n_past >= 0);
    GGML_ASSERT(n_threads > 0);
    // TODO: keep the values of n_batch and n_ctx
    // GGML_ASSERT(n_tokens <= n_batch);
    // GGML_ASSERT(n_past + n_tokens <= n_ctx);

    const int64_t t_start_us = ggml_time_us();

#ifdef GGML_USE_MPI
    ggml_mpi_eval_init(lctx.ctx_mpi, &n_tokens, &n_past, &n_threads);
#endif

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_vocab     = hparams.n_vocab;

#ifdef LLAMA_USE_ALLOCATOR
    ggml_allocr_reset(lctx.alloc);
#endif

    ggml_cgraph * gf = llama_build_graph(lctx, tokens, embd, n_tokens, n_past);

#ifdef LLAMA_USE_ALLOCATOR
    ggml_allocr_alloc_graph(lctx.alloc, gf);
#endif

    // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    n_threads = N >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_gpublas() ? 1 : n_threads;

    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 2];

    GGML_ASSERT(strcmp(res->name, "result_output") == 0);
    GGML_ASSERT(strcmp(embeddings->name, "result_norm") == 0);

#if GGML_USE_MPI
    const int64_t n_layer = hparams.n_layer;
    ggml_mpi_graph_compute_pre(lctx.ctx_mpi, gf, n_layer);
#endif

#ifdef GGML_USE_METAL
    if (lctx.ctx_metal) {
        ggml_metal_set_n_cb     (lctx.ctx_metal, n_threads);
        ggml_metal_graph_compute(lctx.ctx_metal, gf);
        ggml_metal_get_tensor   (lctx.ctx_metal, res);
        if (!lctx.embedding.empty()) {
            ggml_metal_get_tensor(lctx.ctx_metal, embeddings);
        }
    } else {
        ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
    }
#else
    ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
#endif

#if GGML_USE_MPI
    ggml_mpi_graph_compute_post(lctx.ctx_mpi, gf, n_layer);
#endif

    // update kv token count
    lctx.kv_self.n = n_past + N;

    if (cgraph_fname) {
        ggml_graph_export(gf, cgraph_fname);
    }

#ifdef GGML_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
    //}

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(res), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(res) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (!lctx.embedding.empty()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (N > 1) {
        lctx.t_p_eval_us += ggml_time_us() - t_start_us;
        lctx.n_p_eval += N;
    }

    return true;
}

//
// tokenizer
//

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab & vocab) {
    return vocab.type;
}

static bool llama_is_normal_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_NORMAL;
}

static bool llama_is_unknown_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_UNKNOWN;
}

static bool llama_is_control_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_CONTROL;
}

static bool llama_is_user_defined_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_USER_DEFINED;
}

static bool llama_is_unused_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_UNUSED;
}

static bool llama_is_byte_token(const llama_vocab & vocab, llama_token id) {
    return vocab.id_to_token[id].type == LLAMA_TOKEN_TYPE_BYTE;
}

static bool llama_is_bos_token(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(llama_is_control_token(vocab, id));
    return id == vocab.special_bos_id;
}

static bool llama_is_eos_token(const llama_vocab & vocab, llama_token id ) {
    GGML_ASSERT(llama_is_control_token(vocab, id));
    return id == vocab.special_eos_id;
}

static bool llama_is_pad_token(const llama_vocab & vocab, llama_token id ) {
    GGML_ASSERT(id < 0 || llama_is_control_token(vocab, id));
    return id == vocab.special_pad_id;
}

static uint8_t llama_token_to_byte(const llama_vocab & vocab, llama_token id) {
    GGML_ASSERT(llama_is_byte_token(vocab, id));
    const auto& token_data = vocab.id_to_token.at(id);
    auto buf = token_data.text.substr(3, 2);
    return strtol(buf.c_str(), NULL, 16);
}

static llama_token llama_byte_to_token(const llama_vocab & vocab, uint8_t ch) {
    char buf[7];
    int result = snprintf(buf, sizeof(buf), "<0x%02X>", ch);
    GGML_ASSERT(0 <= result && result < 7);
    return vocab.token_to_id.at(buf);
}

static std::string llama_escape_whitespace(const std::string& text) {
    std::string result;
    bool escaping = false;
    result += "\xe2\x96\x81";
    for (size_t offs = 0; offs < text.length(); ++offs) {
        if (text[offs] == ' ') {
            if (!escaping) {
                result += "\xe2\x96\x81";
                escaping = true;
            }
        }
        else {
            escaping = false;
            result += text[offs];
        }
    }
    return result;
}

static std::string llama_unescape_whitespace(const std::string& word) {
    if (word.length() >= 3 && word.substr(0, 3) == "\xe2\x96\x81") {
        return std::string(" ") + word.substr(3);
    }
    return word;
}

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llama_sp_symbol>::value, "llama_sp_symbol is not trivially copyable");

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t len = utf8_len(text[offs]);
            GGML_ASSERT(offs + len <= text.size());
            sym.text = text.c_str() + offs;
            sym.n = len;
            offs += len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llama_sp_symbol &symbol, std::vector<llama_vocab::id> &output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab_.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab_.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token(vocab_, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols_[p->second.first], output);
        resegment(symbols_[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_data = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size = text.size();
        work_queue_.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
    std::map<std::string, std::pair<int, int> > rev_merge;
};

static std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab, const std::string & raw_text, bool bos, bool escape) {
    llama_tokenizer tokenizer(vocab);
    std::vector<llama_vocab::id> output;

    if (raw_text.empty()) {
        return output;
    }

    if (bos) {
        output.push_back(vocab.special_bos_id);
    }

    std::string text;
    if (escape) {
        text = llama_escape_whitespace(raw_text);
    } else {
        text = raw_text;
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// grammar - internal
//

struct llama_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct llama_grammar {
    const std::vector<std::vector<llama_grammar_element>>   rules;
    std::vector<std::vector<const llama_grammar_element *>> stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8                                      partial_utf8;
};

struct llama_grammar_candidate {
    size_t               index;
    const uint32_t     * code_points;
    llama_partial_utf8   partial_utf8;
};

// Decodes a UTF-8 string which may end in an incomplete sequence. Adds a terminating 0 for use as
// pointer. If an invalid sequence is encountered, returns `llama_partial_utf8.n_remain == -1`.
std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const char         * src,
        llama_partial_utf8   partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src;
    std::vector<uint32_t> code_points;
    uint32_t              value    = partial_start.value;
    int                   n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t  first_byte = static_cast<uint8_t>(*pos);
        uint8_t  highbits   = first_byte >> 4;
                 n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, n_remain });
        }

        uint8_t  mask       = (1 << (7 - n_remain)) - 1;
                 value      = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_partial_utf8{ value, n_remain });
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;  // NOLINT
        case LLAMA_GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {

    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;

    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool llama_grammar_match_partial_char(
        const llama_grammar_element * pos,
        const llama_partial_utf8      partial_utf8) {

    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR;
    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

    uint32_t partial_value = partial_utf8.value;
    int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);

    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return !is_positive_char;
}


// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_grammar_advance_stack(
        const std::vector<std::vector<llama_grammar_element>>   & rules,
        const std::vector<const llama_grammar_element *>        & stack,
        std::vector<std::vector<const llama_grammar_element *>> & new_stacks) {

    if (stack.empty()) {
        new_stacks.push_back(stack);
        return;
    }

    const llama_grammar_element * pos = stack.back();

    switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                llama_grammar_advance_stack(rules, new_stack, new_stacks);
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
            new_stacks.push_back(stack);
            break;
        default:
            // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
            // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ASSERT(false);
    }
}

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `llama_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
static std::vector<std::vector<const llama_grammar_element *>> llama_grammar_accept(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const uint32_t                                                  chr) {

    std::vector<std::vector<const llama_grammar_element *>> new_stacks;

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = llama_grammar_match_char(stack.back(), chr);
        if (match.first) {
            const llama_grammar_element * pos = match.second;

            // update top of stack to next element, if any
            std::vector<const llama_grammar_element *> new_stack(stack.begin(), stack.end() - 1);
            if (!llama_grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            llama_grammar_advance_stack(rules, new_stack, new_stacks);
        }
    }

    return new_stacks;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates);

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates_for_stack(
        const std::vector<std::vector<llama_grammar_element>> & rules,
        const std::vector<const llama_grammar_element *>      & stack,
        const std::vector<llama_grammar_candidate>            & candidates) {

    std::vector<llama_grammar_candidate> rejects;

    if (stack.empty()) {
        for (auto tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const llama_grammar_element * stack_pos = stack.back();

    std::vector<llama_grammar_candidate> next_candidates;
    for (auto tok : candidates) {
        if (*tok.code_points == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial_utf8 });
        } else {
            rejects.push_back(tok);
        }
    }

    const auto * stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    std::vector<const llama_grammar_element *> stack_after(stack.begin(), stack.end() - 1);
    if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    std::vector<std::vector<const llama_grammar_element *>> next_stacks;
    llama_grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = llama_grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (auto tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.partial_utf8 });
    }

    return rejects;
}

static std::vector<llama_grammar_candidate> llama_grammar_reject_candidates(
        const std::vector<std::vector<llama_grammar_element>>         & rules,
        const std::vector<std::vector<const llama_grammar_element *>> & stacks,
        const std::vector<llama_grammar_candidate>                    & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return std::vector<llama_grammar_candidate>();
    }

    auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }
    return rejects;
}

//
// grammar - external
//

struct llama_grammar * llama_grammar_init(
            const llama_grammar_element ** rules,
                                 size_t    n_rules,
                                 size_t    start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    std::vector<std::vector<llama_grammar_element>> vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // loop over alternates of start rule to build initial stacks
    std::vector<std::vector<const llama_grammar_element *>> stacks;
    pos = rules[start_rule_index];
    do {
        std::vector<const llama_grammar_element *> stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    return new llama_grammar{ std::move(vec_rules), std::move(stacks), {} };
}

void llama_grammar_free(struct llama_grammar * grammar) {
    delete grammar;
}

//
// sampling
//

void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates) {
    GGML_ASSERT(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sample_softmax(ctx, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax(nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax(nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        const auto * token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens_p, size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_grammar(struct llama_context * ctx, llama_token_data_array * candidates, const struct llama_grammar * grammar) {
    GGML_ASSERT(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    bool allow_eos = false;
    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            allow_eos = true;
            break;
        }
    }

    const llama_token eos = llama_token_eos(ctx);

    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> candidates_decoded;
    std::vector<llama_grammar_candidate>                              candidates_grammar;

    for (size_t i = 0; i < candidates->size; ++i) {
        const llama_token id   = candidates->data[i].id;
        const std::string text = llama_token_to_text(ctx, id);
        if (id == eos) {
            if (!allow_eos) {
                candidates->data[i].logit = -INFINITY;
            }
        } else if (text.empty()) {
            candidates->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(text.c_str(), grammar->partial_utf8));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second });
        }
    }

    const auto rejects = llama_grammar_reject_candidates(grammar->rules, grammar->stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        candidates->data[reject.index].logit = -INFINITY;
    }

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

static void llama_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_sample_classifier_free_guidance(
          struct llama_context * ctx,
        llama_token_data_array * candidates,
          struct llama_context * guidance_ctx,
                         float   scale) {
    int64_t t_start_sample_us = ggml_time_us();

    GGML_ASSERT(ctx);

    auto n_vocab = llama_n_vocab(ctx);

    GGML_ASSERT(n_vocab == (int)candidates->size);
    GGML_ASSERT(!candidates->sorted);

    std::vector<float> logits_base;
    logits_base.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        logits_base.push_back(candidates->data[i].logit);
    }
    llama_log_softmax(logits_base.data(), candidates->size);

    float* logits_guidance = llama_get_logits(guidance_ctx);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
        float logit_guidance = logits_guidance[i];
        float logit_base = logits_base[i];
        candidates->data[i].logit = scale * (logit_base - logit_guidance) + logit_guidance;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    GGML_ASSERT(ctx);

    auto N = float(llama_n_vocab(ctx));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates) {
    GGML_ASSERT(ctx);

    const int64_t t_start_sample_us = ggml_time_us();
    llama_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    llama_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

void llama_grammar_accept_token(struct llama_context * ctx, struct llama_grammar * grammar, llama_token token) {
    const int64_t t_start_sample_us = ggml_time_us();

    if (token == llama_token_eos(ctx)) {
        for (const auto & stack : grammar->stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ASSERT(false);
    }

    const std::string text = llama_token_to_text(ctx, token);

    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(text.c_str(), grammar->partial_utf8);
    const auto & code_points = decoded.first;
    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        grammar->stacks = llama_grammar_accept(grammar->rules, grammar->stacks, *it);
    }
    grammar->partial_utf8 = decoded.second;
    GGML_ASSERT(!grammar->stacks.empty());

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
}

//
// quantization
//

static void llama_convert_tensor_internal(struct ggml_tensor * tensor, std::vector<float> & output, const size_t nelements, const int nthread) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    ggml_type_traits_t qtype;
    if (ggml_is_quantized(tensor->type)) {
        qtype = ggml_internal_get_type_traits(tensor->type);
        if (qtype.to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype.to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ASSERT(false); // unreachable
        }
        return;
    }

    auto block_size = tensor->type == GGML_TYPE_F16 ? 1 : (size_t)ggml_blck_size(tensor->type);
    auto block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    auto nblocks = nelements / block_size;
    auto blocks_per_thread = nblocks / nthread;
    auto spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    std::vector<std::thread> workers;
    for (auto tnum = 0, in_buff_offs = 0, out_buff_offs = 0; tnum < nthread; tnum++) {
        auto thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        auto thr_elems = thr_blocks * block_size; // number of elements for this thread
        auto thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else {
                qtype.to_float(inbuf, outbuf, nels);
            }
        };
        workers.push_back(std::thread(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems));
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & worker : workers) {
        worker.join();
    }
}

static void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type quantized_type;
    llama_ftype ftype = params->ftype;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  quantized_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_ALL_F32:     quantized_type = GGML_TYPE_F32;  break;

#ifdef GGML_USE_K_QUANTS
        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K:   quantized_type = GGML_TYPE_Q2_K; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: quantized_type = GGML_TYPE_Q3_K; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: quantized_type = GGML_TYPE_Q4_K; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: quantized_type = GGML_TYPE_Q5_K; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:   quantized_type = GGML_TYPE_Q6_K; break;
#endif
        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    std::unique_ptr<llama_model_loader> model_loader(new llama_model_loader(fname_inp, /*use_mmap*/ false));

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    struct gguf_context * ctx_out = gguf_init_empty();

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out, model_loader->ctx_gguf);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);

#ifdef GGML_USE_K_QUANTS
    int n_attention_wv    = 0;
    int n_feed_forward_w2 = 0;

    for (int i = 0; i < model_loader->n_tensors; ++i) {
        struct ggml_tensor * meta = model_loader->get_tensor_meta(i);

        const std::string name = ggml_get_name(meta);

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight") != std::string::npos) {
            ++n_attention_wv;
        }
        else if (name.find("ffn_down.weight") != std::string::npos) {
            ++n_feed_forward_w2;
        }
    }

    int i_attention_wv = 0;
    int i_feed_forward_w2 = 0;
#endif

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    std::mutex mutex;

    auto use_more_bits = [] (int i_layer, int num_layers) -> bool {
        return i_layer < num_layers/8 || i_layer >= 7*num_layers/8 || (i_layer - num_layers/8)%3 == 2;
    };

    int idx = 0;

    std::vector<uint8_t> read_data;
    std::vector<uint8_t> work;

    // populate the original tensors so we get an initial meta data
    for (int i = 0; i < model_loader->n_tensors; ++i) {
        struct ggml_tensor * meta = model_loader->get_tensor_meta(i);
        gguf_add_tensor(ctx_out, meta);
    }

    std::ofstream fout(fname_out, std::ios::binary);

    const size_t meta_size = gguf_get_meta_size(ctx_out);

    LLAMA_LOG_INFO("%s: meta size = %zu bytes\n", __func__, meta_size);

    // placeholder for the meta data
    ::zeros(fout, meta_size);

    for (int i = 0; i < model_loader->n_tensors; ++i) {
        struct ggml_tensor * tensor = model_loader->get_tensor_meta(i);

        const std::string name = ggml_get_name(tensor);

        read_data.resize(ggml_nbytes(tensor));
        tensor->data = read_data.data();
        model_loader->load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, model_loader->n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor->n_dims == 2);
        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= quantized_type != tensor->type;

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            new_type = quantized_type;
#ifdef GGML_USE_K_QUANTS
            // TODO: avoid hardcoded tensor names - use the TN_* constants
            if (name == TN_OUTPUT) {
                int nx = tensor->ne[0];
                int ny = tensor->ne[1];
                if (nx % QK_K == 0 && ny % QK_K == 0) {
                    new_type = GGML_TYPE_Q6_K;
                }
            } else if (name.find("attn_v.weight") != std::string::npos) {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
                else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                        use_more_bits(i_attention_wv, n_attention_wv)) new_type = GGML_TYPE_Q6_K;
                else if (QK_K == 64 && (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S) &&
                        (i_attention_wv < n_attention_wv/8 || i_attention_wv >= 7*n_attention_wv/8)) new_type = GGML_TYPE_Q6_K;
                ++i_attention_wv;
            } else if (name.find("ffn_down.weight") != std::string::npos) {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
                else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                         use_more_bits(i_feed_forward_w2, n_feed_forward_w2)) new_type = GGML_TYPE_Q6_K;
                //else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && i_feed_forward_w2 < n_feed_forward_w2/8) new_type = GGML_TYPE_Q6_K;
                ++i_feed_forward_w2;
            } else if (name.find("attn_output.weight") != std::string::npos) {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
            }
            bool convert_incompatible_tensor = false;
            if (new_type == GGML_TYPE_Q2_K || new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K ||
                new_type == GGML_TYPE_Q5_K || new_type == GGML_TYPE_Q6_K) {
                int nx = tensor->ne[0];
                int ny = tensor->ne[1];
                if (nx % QK_K != 0 || ny % QK_K != 0) {
                    LLAMA_LOG_INFO("\n\nTensor sizes %d x %d are not divisible by %d, required for k-quants.\n",nx,ny,QK_K);
                    convert_incompatible_tensor = true;
                }
            }
            if (convert_incompatible_tensor) {
                if (name == TN_OUTPUT) {
                    new_type = GGML_TYPE_F16; //fall back to F16 instead of just failing.
                    LLAMA_LOG_WARN("F16 will be used for this tensor instead.\n");
                } else if (name == TN_TOKEN_EMBD) {
                    new_type = GGML_TYPE_Q4_0; //fall back to Q4_0 instead of just failing.
                    LLAMA_LOG_WARN("Q4_0 will be used for this tensor instead.\n");
                } else {
                    throw std::runtime_error("Unsupported tensor size encountered\n");
                }
            }
#endif

            const size_t nelements = ggml_nelements(tensor);

            float * f32_data;
            std::vector<float> f32_conv_buf;

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_convert_tensor_internal(tensor, f32_conv_buf, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            LLAMA_LOG_INFO("quantizing to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            work.resize(nelements * 4); // upper bound on size
            new_data = work.data();
            std::vector<int64_t> hist_cur(1 << 4, 0);

            static const int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements]() {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) {
                                    hist_cur[j] += local_hist[j];
                                }
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) {
                            local_hist.resize(hist_cur.size(), 0);
                        }
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if ((int) workers.size() < nthread_use - 1) {
                    workers.resize(nthread_use - 1);
                }
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it] = std::thread(compute);
                }
                compute();
                for (int it = 0; it < nthread_use - 1; ++it) {
                    workers[it].join();
                }
            }

            LLAMA_LOG_INFO("size = %8.2f MB -> %8.2f MB | hist: ", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
            int64_t tot_count = 0;
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
                tot_count += hist_cur[i];
            }

            if (tot_count > 0) {
                for (size_t i = 0; i < hist_cur.size(); i++) {
                    LLAMA_LOG_INFO("%5.3f ", hist_cur[i] / float(nelements));
                }
            }
            LLAMA_LOG_INFO("\n");
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out, name.c_str(), new_data, new_size);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }

    // go back to beginning of file and write the updated meta data
    {
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *) data.data(), data.size());
    }

    fout.close();

    gguf_free(ctx_out);

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

    // print histogram for all tensors
    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        if (sum_all > 0) {
            LLAMA_LOG_INFO("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); i++) {
                LLAMA_LOG_INFO("%5.3f ", hist_all[i] / float(sum_all));
            }
            LLAMA_LOG_INFO("\n");
        }
    }
}

// TODO: after the GGUF PR, this likely won't work and needs to be updated
int llama_apply_lora_from_file_internal(const struct llama_model & model, const char * path_lora, const char * path_base_model, int n_threads) {
    LLAMA_LOG_INFO("%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    const int64_t t_start_lora_us = ggml_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        LLAMA_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            LLAMA_LOG_ERROR("%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = (float)lora_alpha / (float)lora_r;

    LLAMA_LOG_INFO("%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_context * lora_ctx = ggml_init(params);
    std::unordered_map<std::string, struct ggml_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_tensor*> model_tensors;
    for (const auto & kv : model.tensors_by_name) {
        model_tensors.insert(kv);
    }

    // load base model
    std::unique_ptr<llama_model_loader> model_loader;
    ggml_context * base_ctx = NULL;
    std::vector<uint8_t> base_buf;
    if (path_base_model) {
        LLAMA_LOG_INFO("%s: loading base model from '%s'\n", __func__, path_base_model);
        model_loader.reset(new llama_model_loader(path_base_model, /*use_mmap*/ true));

        size_t ctx_size;
        size_t mmapped_size;
        model_loader->calc_sizes(ctx_size, mmapped_size);
        base_buf.resize(ctx_size);

        ggml_init_params base_params;
        base_params.mem_size   = base_buf.size();
        base_params.mem_buffer = base_buf.data();
        base_params.no_alloc   = model_loader->use_mmap;

        base_ctx = ggml_init(base_params);

        // maybe this should in llama_model_loader
        if (model_loader->use_mmap) {
            model_loader->mapping.reset(new llama_mmap(&model_loader->file, /* prefetch */ 0, ggml_is_numa()));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;

    std::vector<uint8_t> work_buffer;

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

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name;
        {
            char buf[1024];
            fin.read(buf, length);
            name = std::string(buf, length);
        }

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            LLAMA_LOG_ERROR("%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // LLAMA_LOG_INFO("%s: %s => %s (lora type %s) \n", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name) == model_tensors.end()) {
            LLAMA_LOG_ERROR("%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            default:
                    {
                        LLAMA_LOG_ERROR("%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_tensor * lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            LLAMA_LOG_ERROR("%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }
        ggml_set_name(lora_tensor, "lora_tensor");

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_tensor * dest_t = model_tensors[base_name];

            offload_func_t offload_func = llama_nop;
            offload_func_t offload_func_force_inplace = llama_nop;

#ifdef GGML_USE_CUBLAS
            if (dest_t->backend == GGML_BACKEND_GPU || dest_t->backend == GGML_BACKEND_GPU_SPLIT) {
                if (dest_t->type != GGML_TYPE_F16) {
                    throw std::runtime_error(format(
                        "%s: error: the simultaneous use of LoRAs and GPU acceleration is only supported for f16 models", __func__));
                }
                offload_func = ggml_cuda_assign_buffers;
                offload_func_force_inplace = ggml_cuda_assign_buffers_force_inplace;
            }
#endif // GGML_USE_CUBLAS

            ggml_tensor * base_t;
            if (model_loader) {
                struct gguf_context * ctx_gguf = model_loader->ctx_gguf;

                // load from base model
                if (gguf_find_tensor(ctx_gguf, base_name.c_str()) < 0) {
                    LLAMA_LOG_ERROR("%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }

                // TODO: not tested!! maybe not working!
                base_t = model_loader->create_tensor(base_ctx, base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] }, GGML_BACKEND_CPU);
                model_loader->load_data_for(base_t);
            } else {
                base_t = dest_t;
            }

            if (ggml_is_quantized(base_t->type)) {
                if (!warned) {
                    LLAMA_LOG_WARN("%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                   "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_tensor * loraA = lora_tensors[base_name + ".loraA"];
            GGML_ASSERT(loraA->type == GGML_TYPE_F32);
            ggml_set_name(loraA, "loraA");

            ggml_tensor * loraB = lora_tensors[base_name + ".loraB"];
            GGML_ASSERT(loraB->type == GGML_TYPE_F32);
            ggml_set_name(loraB, "loraB");

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                LLAMA_LOG_ERROR("%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                                " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_tensor * BA = ggml_mul_mat(lora_ctx, loraA, loraB);
            offload_func(BA);
            ggml_set_name(BA, "BA");

            if (scaling != 1.0f) {
                ggml_tensor * scale_tensor = ggml_new_f32(lora_ctx, scaling);
                ggml_set_name(scale_tensor, "scale_tensor");

                BA = ggml_scale_inplace(lora_ctx, BA, scale_tensor);
                offload_func(BA);
                ggml_set_name(BA, "BA_scaled");
            }

            ggml_tensor * r;
            if (base_t == dest_t) {
                r = ggml_add_inplace(lora_ctx, dest_t, BA);
                offload_func_force_inplace(r);
                ggml_set_name(r, "r_add_inplace");
            }
            else {
                r = ggml_add(lora_ctx, base_t, BA);
                offload_func(r);
                ggml_set_name(r, "r_add");

                r = ggml_cpy(lora_ctx, r, dest_t);
                offload_func(r);
                ggml_set_name(r, "r_cpy");
            }

            struct ggml_cgraph gf = ggml_build_forward(r);

            ggml_graph_compute_helper(work_buffer, &gf, n_threads);

            // we won't need these tensors again, reset the context to save memory
            ggml_free(lora_ctx);
            lora_ctx = ggml_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0) {
                LLAMA_LOG_INFO(".");
            }
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_free(lora_ctx);
    if (base_ctx) {
        ggml_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
    LLAMA_LOG_INFO(" done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

//
// interface implementation
//

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.seed                        =*/ LLAMA_DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 512,
        /*.gpu_layers                  =*/ 0,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.rope_freq_base              =*/ 10000.0f,
        /*.rope_freq_scale             =*/ 1.0f,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.low_vram                    =*/ false,
        /*.mul_mat_q                   =*/ false,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
    };

    return result;
}

struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
    };

    return result;
}

int llama_max_devices(void) {
    return LLAMA_MAX_DEVICES;
}

bool llama_mmap_supported(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_mlock_supported(void) {
    return llama_mlock::SUPPORTED;
}

void llama_backend_init(bool numa) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if (numa) {
        ggml_numa_init();
    }

#ifdef GGML_USE_MPI
    ggml_mpi_backend_init();
#endif
}

void llama_backend_free(void) {
#ifdef GGML_USE_MPI
    ggml_mpi_backend_free();
#endif
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

struct llama_model * llama_load_model_from_file(
                             const char * path_model,
            struct llama_context_params   params) {
    ggml_time_init();

    llama_model * model = new llama_model;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!llama_model_load(path_model, *model, model->vocab, params.n_ctx, params.n_batch, params.n_gpu_layers,
                params.main_gpu, params.tensor_split, params.mul_mat_q, params.rope_freq_base, params.rope_freq_scale,
                params.low_vram, memory_type, params.use_mmap, params.use_mlock, params.vocab_only,
                params.progress_callback, params.progress_callback_user_data)) {
        LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        delete model;
        return nullptr;
    }

    return model;
}

void llama_free_model(struct llama_model * model) {
    delete model;
}

struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {

    if (!model) {
        return nullptr;
    }

    llama_context * ctx = new llama_context(*model);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_INFO(".");
                if (percentage >= 100) {
                    LLAMA_LOG_INFO("\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!llama_kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, ctx->model.hparams.n_ctx, params.n_gpu_layers)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->kv_self.k) + ggml_nbytes(ctx->kv_self.v);
            LLAMA_LOG_INFO("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_vocab);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

#ifdef LLAMA_USE_ALLOCATOR
        {
            static const size_t tensor_alignment = 32;
            // the compute buffer is used to store the tensor and graph structs, while the allocator buffer is used for the tensor data
            ctx->buf_compute.resize(ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead());

            // create measure allocator
            ctx->alloc = ggml_allocr_new_measure(tensor_alignment);

            // build worst-case graph
            int n_tokens = std::min((int)hparams.n_ctx, params.n_batch);
            int n_past = hparams.n_ctx - n_tokens;
            llama_token token = llama_token_bos(ctx); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            ggml_cgraph * gf = llama_build_graph(*ctx, &token, NULL, n_tokens, n_past);
#ifdef GGML_USE_METAL
            if (params.n_gpu_layers > 0) {
                ctx->ctx_metal = ggml_metal_init(1);
                if (!ctx->ctx_metal) {
                    LLAMA_LOG_ERROR("%s: ggml_metal_init() failed\n", __func__);
                    llama_free(ctx);
                    return NULL;
                }
                ggml_metal_graph_find_concurrency(ctx->ctx_metal, gf, false);
                ggml_allocr_set_parse_seq(ctx->alloc, ggml_metal_get_concur_list(ctx->ctx_metal), ggml_metal_if_optimized(ctx->ctx_metal));
            }
#endif
            // measure memory requirements for the graph
            size_t alloc_size = ggml_allocr_alloc_graph(ctx->alloc, gf) + tensor_alignment;

            LLAMA_LOG_INFO("%s: compute buffer total size = %7.2f MB\n", __func__, (ctx->buf_compute.size + alloc_size) / 1024.0 / 1024.0);

            // debug - for comparison with scratch buffer
            //size_t prev_req =
            //    MEM_REQ_SCRATCH0(hparams.n_ctx).at(ctx->model.type) +
            //    MEM_REQ_SCRATCH1().at(ctx->model.type) +
            //    MEM_REQ_EVAL().at(ctx->model.type);
            //LLAMA_LOG_INFO("%s: (debug) equivalent with scratch buffer = %7.2f MB\n", __func__, prev_req / 1024.0 / 1024.0);

            // recreate allocator with exact memory requirements
            ggml_allocr_free(ctx->alloc);

            ctx->buf_alloc.resize(alloc_size);
            ctx->alloc = ggml_allocr_new(ctx->buf_alloc.data, ctx->buf_alloc.size, tensor_alignment);
#ifdef GGML_USE_METAL
            if (ctx->ctx_metal) {
                ggml_allocr_set_parse_seq(ctx->alloc, ggml_metal_get_concur_list(ctx->ctx_metal), ggml_metal_if_optimized(ctx->ctx_metal));
            }
#endif
        }
#else
        ctx->buf_compute.resize(MEM_REQ_EVAL().at(ctx->model.type) + ggml_graph_overhead());
#endif

#ifdef LLAMA_USE_SCRATCH
        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0(hparams.n_ctx).at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1().at(ctx->model.type));
#endif
    }

#ifdef GGML_USE_METAL
    if (params.n_gpu_layers > 0) {
        // this allocates all Metal resources and memory buffers

        void * data_ptr  = NULL;
        size_t data_size = 0;

        if (params.use_mmap) {
            data_ptr  = ctx->model.mapping->addr;
            data_size = ctx->model.mapping->size;
        } else {
            data_ptr  = ggml_get_mem_buffer(ctx->model.ctx);
            data_size = ggml_get_mem_size  (ctx->model.ctx);
        }

        const size_t max_size = ggml_get_max_tensor_size(ctx->model.ctx);

        LLAMA_LOG_INFO("%s: max tensor size = %8.2f MB\n", __func__, max_size/1024.0/1024.0);

#define LLAMA_METAL_CHECK_BUF(result)                            \
    if (!(result)) {                                             \
        LLAMA_LOG_ERROR("%s: failed to add buffer\n", __func__); \
        llama_free(ctx);                                         \
        return NULL;                                             \
    }

        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "data", data_ptr, data_size, max_size));

        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "eval", ctx->buf_compute.data, ctx->buf_compute.size, 0));
        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "kv",   ctx->kv_self.buf.data, ctx->kv_self.buf.size, 0));

        LLAMA_METAL_CHECK_BUF(ggml_metal_add_buffer(ctx->ctx_metal, "alloc", ctx->buf_alloc.data, ctx->buf_alloc.size, 0));
#undef LLAMA_METAL_CHECK_BUF
    }
#endif

#ifdef GGML_USE_MPI
    ctx->ctx_mpi = ggml_mpi_init();

    if (ggml_mpi_rank(ctx->ctx_mpi) > 0) {
        // Enter a blocking eval loop with dummy input, letting rank=0 drive the process
        const std::vector<llama_token> tmp(ctx->model.hparams.n_ctx, llama_token_bos(ctx));
        while (!llama_eval(ctx, tmp.data(), tmp.size(), 0, 0)) {};
        llama_backend_free();
        exit(1);
    }
#endif

    return ctx;
}

struct llama_context * llama_init_from_file(
                             const char * path_model,
            struct llama_context_params   params) {

    struct llama_model * model = llama_load_model_from_file(path_model, params);
    if (!model) {
        return nullptr;
    }
    struct llama_context * ctx = llama_new_context_with_model(model, params);
    ctx->model_owner = true;
    return ctx;
}

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

int llama_n_vocab(const struct llama_context * ctx) {
    return ctx->model.vocab.id_to_token.size();
}

int llama_n_ctx(const struct llama_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int llama_n_embd(const struct llama_context * ctx) {
    return ctx->model.hparams.n_embd;
}

int llama_model_n_vocab(const struct llama_model * model) {
    return model->vocab.id_to_token.size();
}

int llama_model_n_ctx(const struct llama_model * model) {
    return model->hparams.n_ctx;
}

int llama_model_n_embd(const struct llama_model * model) {
    return model->hparams.n_embd;
}

int llama_model_type(const struct llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "LLaMA %s %s", llama_model_type_name(model->type), llama_model_ftype_name(model->ftype));
}

int llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_internal(fname_inp, fname_out, params);
        return 0;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_apply_lora_from_file(struct llama_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_apply_lora_from_file_internal(ctx->model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_model_apply_lora_from_file(const struct llama_model * model, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return llama_apply_lora_from_file_internal(*model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int llama_get_kv_cache_token_count(const struct llama_context * ctx) {
    return ctx->kv_self.n;
}

#define LLAMA_MAX_RNG_STATE (64*1024)

void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t llama_get_state_size(const struct llama_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = LLAMA_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_kv_size         = sizeof(size_t);
    const size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv              = ctx->kv_self.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

// llama_context_data
struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t * ptr;
    size_t size_written = 0;

    llama_data_buffer_context(uint8_t * p) : ptr(p) {}

    void write(const void * src, size_t size) override {
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_file_context : llama_data_context {
    llama_file * file;
    size_t size_written = 0;

    llama_data_file_context(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_file_context data_ctx(&file);
 * llama_copy_state_data(ctx, &data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_buffer_context data_ctx(&buf.data());
 * llama_copy_state_data(ctx, &data_ctx);
 *
*/
void llama_copy_state_data_internal(struct llama_context * ctx, llama_data_context * data_ctx) {
    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[LLAMA_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, LLAMA_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        data_ctx->write(&rng_size,   sizeof(rng_size));
        data_ctx->write(&rng_buf[0], LLAMA_MAX_RNG_STATE);
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        data_ctx->write(&logits_cap,  sizeof(logits_cap));
        data_ctx->write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            data_ctx->write(ctx->logits.data(), logits_size * sizeof(float));
        }

        // If there is a gap between the size and the capacity, write padding
        size_t padding_size = (logits_cap - logits_size) * sizeof(float);
        if (padding_size > 0) {
            std::vector<uint8_t> padding(padding_size, 0); // Create a buffer filled with zeros
            data_ctx->write(padding.data(), padding_size);
        }
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        data_ctx->write(&embedding_size, sizeof(embedding_size));

        if (embedding_size) {
            data_ctx->write(ctx->embedding.data(), embedding_size * sizeof(float));
        }
    }

    // copy kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = hparams.n_ctx;

        const size_t kv_size = kv_self.buf.size;
        const int    kv_ntok = llama_get_kv_cache_token_count(ctx);

        data_ctx->write(&kv_size, sizeof(kv_size));
        data_ctx->write(&kv_ntok, sizeof(kv_ntok));

        if (kv_size) {
            const size_t elt_size = ggml_element_size(kv_self.k);

            ggml_context * cpy_ctx = ggml_init({ 4096, NULL, /* no_alloc */ true });
            ggml_cgraph gf{};

            ggml_tensor * kout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            std::vector<uint8_t> kout3d_data(ggml_nbytes(kout3d), 0);
            kout3d->data = kout3d_data.data();

            ggml_tensor * vout3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            std::vector<uint8_t> vout3d_data(ggml_nbytes(vout3d), 0);
            vout3d->data = vout3d_data.data();

            ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, k3d, kout3d));
            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, v3d, vout3d));
            ggml_graph_compute_helper(ctx->work_buffer, &gf, /*n_threads*/ 1);

            ggml_free(cpy_ctx);

            // our data is now in the kout3d_data and vout3d_data buffers
            // write them to file
            data_ctx->write(kout3d_data.data(), kout3d_data.size());
            data_ctx->write(vout3d_data.data(), vout3d_data.size());
        }
    }
}

size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    llama_data_buffer_context data_ctx(dst);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return data_ctx.get_size_written();
}

// Sets the state reading from the specified source address
size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src) {
    uint8_t * inp = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[LLAMA_MAX_RNG_STATE];

        memcpy(&rng_size,   inp, sizeof(rng_size));    inp += sizeof(rng_size);
        memcpy(&rng_buf[0], inp, LLAMA_MAX_RNG_STATE); inp += LLAMA_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        GGML_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  inp, sizeof(logits_cap));  inp += sizeof(logits_cap);
        memcpy(&logits_size, inp, sizeof(logits_size)); inp += sizeof(logits_size);

        GGML_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), inp, logits_size * sizeof(float));
        }

        inp += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, inp, sizeof(embedding_size)); inp += sizeof(embedding_size);

        GGML_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), inp, embedding_size * sizeof(float));
            inp += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        const auto & kv_self = ctx->kv_self;
        const auto & hparams = ctx->model.hparams;
        const int    n_layer = hparams.n_layer;
        const int    n_embd  = hparams.n_embd_gqa();
        const int    n_ctx   = hparams.n_ctx;

        size_t kv_size;
        int kv_ntok;

        memcpy(&kv_size, inp, sizeof(kv_size)); inp += sizeof(kv_size);
        memcpy(&kv_ntok, inp, sizeof(kv_ntok)); inp += sizeof(kv_ntok);

        if (kv_size) {
            GGML_ASSERT(kv_self.buf.size == kv_size);

            const size_t elt_size = ggml_element_size(kv_self.k);

            ggml_context * cpy_ctx = ggml_init({ 4096, NULL, /* no_alloc */ true });
            ggml_cgraph gf{};

            ggml_tensor * kin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer);
            kin3d->data = (void *) inp;
            inp += ggml_nbytes(kin3d);

            ggml_tensor * vin3d = ggml_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer);
            vin3d->data = (void *) inp;
            inp += ggml_nbytes(vin3d);

            ggml_tensor * k3d = ggml_view_3d(cpy_ctx, kv_self.k,
                n_embd, kv_ntok, n_layer,
                elt_size*n_embd, elt_size*n_embd*n_ctx, 0);

            ggml_tensor * v3d = ggml_view_3d(cpy_ctx, kv_self.v,
                kv_ntok, n_embd, n_layer,
                elt_size*n_ctx, elt_size*n_ctx*n_embd, 0);

            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, kin3d, k3d));
            ggml_build_forward_expand(&gf, ggml_cpy(cpy_ctx, vin3d, v3d));
            ggml_graph_compute_helper(ctx->work_buffer, &gf, /*n_threads*/ 1);

            ggml_free(cpy_ctx);
        }

        ctx->kv_self.n = kv_ntok;
    }

    const size_t nread    = inp - src;
    const size_t max_size = llama_get_state_size(ctx);

    GGML_ASSERT(nread <= max_size);

    return nread;
}

static bool llama_load_session_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }

        llama_hparams session_hparams;
        file.read_raw(&session_hparams, sizeof(llama_hparams));

        if (session_hparams != ctx->model.hparams) {
            LLAMA_LOG_INFO("%s : model hparams didn't match from session file!\n", __func__);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size - file.tell();
        const size_t n_state_size_max = llama_get_state_size(ctx);

        if (n_state_size_cur > n_state_size_max) {
            LLAMA_LOG_ERROR("%s : the state size in session file is too big! max %zu, got %zu\n", __func__, n_state_size_max, n_state_size_cur);
            return false;
        }

        std::vector<uint8_t> state_data(n_state_size_max);
        file.read_raw(state_data.data(), n_state_size_cur);

        llama_set_state_data(ctx, state_data.data());
    }

    return true;
}

bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_load_session_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("error loading session file: %s\n", err.what());
        return false;
    }
}

bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    file.write_raw(&ctx->model.hparams, sizeof(llama_hparams));

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_file_context data_ctx(&file);
    llama_copy_state_data_internal(ctx, &data_ctx);

    return true;
}

int llama_eval(
        struct llama_context * ctx,
           const llama_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!llama_eval_internal(*ctx, tokens, nullptr, n_tokens, n_past, n_threads, nullptr)) {
        LLAMA_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}

int llama_eval_embd(
            struct llama_context * ctx,
                     const float * embd,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads) {
    if (!llama_eval_internal(*ctx, nullptr, embd, n_tokens, n_past, n_threads, nullptr)) {
        LLAMA_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    // get a more accurate load time, upon first eval
    // TODO: fix this
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    return 0;
}

int llama_eval_export(struct llama_context * ctx, const char * fname) {
    const int n_batch = 1;
    const int n_ctx   = 512 - n_batch;

    const std::vector<llama_token> tmp(n_batch, llama_token_bos(ctx));

    if (!llama_eval_internal(*ctx, tmp.data(), nullptr, tmp.size(), n_ctx, 1, fname)) {
        LLAMA_LOG_ERROR("%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

float * llama_get_logits(struct llama_context * ctx) {
    return ctx->logits.data();
}

float * llama_get_embeddings(struct llama_context * ctx) {
    return ctx->embedding.data();
}

const char * llama_token_get_text(const struct llama_context * ctx, llama_token token) {
    return ctx->model.vocab.id_to_token[token].text.c_str();
}

float llama_token_get_score(const struct llama_context * ctx, llama_token token) {
    return ctx->model.vocab.id_to_token[token].score;
}

llama_token_type llama_token_get_type(const struct llama_context * ctx, llama_token token) {
    return ctx->model.vocab.id_to_token[token].type;
}

llama_token llama_token_bos(const struct llama_context * ctx) {
    return ctx->model.vocab.special_bos_id;
}

llama_token llama_token_eos(const struct llama_context * ctx) {
    return ctx->model.vocab.special_eos_id;
}

llama_token llama_token_nl(const struct llama_context * ctx) {
    return ctx->model.vocab.linefeed_id;
}

int llama_tokenize(
        struct llama_context * ctx,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    return llama_tokenize_with_model(&ctx->model, text, tokens, n_max_tokens, add_bos);
}

int llama_tokenize_bpe(
        struct llama_context * ctx,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = llama_tokenize_internal(ctx->model.vocab, text, add_bos, false);

    if (n_max_tokens < (int) res.size()) {
        LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_tokenize_with_model(
    const struct llama_model * model,
                  const char * text,
                 llama_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto escape = llama_vocab_get_type(model->vocab) == LLAMA_VOCAB_TYPE_SPM;
    auto res = llama_tokenize_internal(model->vocab, text, add_bos, escape);

    if (n_max_tokens < (int) res.size()) {
        LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int llama_token_to_str(const struct llama_context * ctx, llama_token token, char * buf, int length) {
    return llama_token_to_str_with_model(&ctx->model, token, buf, length);
}

int llama_token_to_str_bpe(const struct llama_context * ctx, llama_token token, char * buf, int length) {
    if (0 <= token && token < llama_model_n_vocab(&ctx->model)) {
        std::string result = ctx->model.vocab.id_to_token[token].text;
        if (length < (int) result.length()) {
            return -result.length();
        }
        memcpy(buf, result.c_str(), result.length());
        return result.length();
    }
    return 0;
}

// does not write null-terminator to str
int llama_token_to_str_with_model(const struct llama_model * model, llama_token token, char * buf, int length) {
    if (0 <= token && token < llama_model_n_vocab(model)) {
        if (llama_is_normal_token(model->vocab, token)) {
            std::string result = model->vocab.id_to_token[token].text;
            if (llama_vocab_get_type(model->vocab) == LLAMA_VOCAB_TYPE_SPM) {
                result = llama_unescape_whitespace(result);
            }
            if (length < (int) result.length()) {
                return -result.length();
            }
            memcpy(buf, result.c_str(), result.length());
            return result.length();
        } else if (llama_is_unknown_token(model->vocab, token)) { // NOLINT
            if (length < 3) {
                return -3;
            }
            buf[0] = '\xe2';
            buf[1] = '\x96';
            buf[2] = '\x85';
            return 3;
        } else if (llama_is_control_token(model->vocab, token)) {
            ;
        } else if (llama_is_byte_token(model->vocab, token)) {
            if (length < 1) {
                return -1;
            }
            buf[0] = llama_token_to_byte(model->vocab, token);
            return 1;
        }
    }
    return 0;
}

struct llama_timings llama_get_timings(struct llama_context * ctx) {
    struct llama_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->n_sample),
        /*.n_p_eval =*/ std::max(1, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    return result;
}

void llama_print_timings(struct llama_context * ctx) {
    const llama_timings timings = llama_get_timings(ctx);

    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("%s:        load time = %8.2f ms\n", __func__, timings.t_load_ms);
    LLAMA_LOG_INFO("%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    LLAMA_LOG_INFO("%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %8.2f ms\n", __func__, (timings.t_end_ms - timings.t_start_ms));
}

void llama_reset_timings(struct llama_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>>& llama_internal_get_tensor_map(struct llama_context * ctx) {
    return ctx->model.tensors_by_name;
}

void llama_log_set(llama_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_state.log_callback_user_data = user_data;
}

#if defined(_MSC_VER) && !defined(vsnprintf)
#define vsnprintf _vsnprintf
#endif

static void llama_log_internal_v(llama_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

static void llama_log_internal(llama_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

static void llama_log_callback_default(llama_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}
