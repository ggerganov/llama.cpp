// Internal header to be included only by llama.cpp.
// Contains wrappers around OS interfaces.

#ifndef LLAMA_UTIL_H
#define LLAMA_UTIL_H

#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <climits>

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <stdexcept>

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

#define LLAMA_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "LLAMA_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifdef __GNUC__
#ifdef __MINGW32__
__attribute__((format(gnu_printf, 1, 2)))
#else
__attribute__((format(printf, 1, 2)))
#endif
#endif
static std::string format(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    LLAMA_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    LLAMA_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

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
        LLAMA_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        LLAMA_ASSERT(ret == 0); // same
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

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
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

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

// llama_context_data
struct llama_data_context {
    virtual void write(const void * src, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_context() = default;
};

struct llama_data_buffer_context : llama_data_context {
    uint8_t* ptr;
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
    llama_file* file;
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
        if (prefetch >= file->size) { flags |= MAP_POPULATE; }
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

    llama_mmap(struct llama_file *, bool prefetch = true, bool numa = false) {
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
        LLAMA_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        LLAMA_ASSERT(addr);
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

    size_t lock_granularity() {
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

    bool raw_lock(const void * addr, size_t size) {
        if (!mlock(addr, size)) {
            return true;
        } else {
            char* errmsg = std::strerror(errno);
            bool suggest = (errno == ENOMEM);

            // Check if the resource limit is fine after all
            struct rlimit lock_limit;
            if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit))
                suggest = false;
            if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size))
                suggest = false;

            fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                    size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
            return false;
        }
    }

    #undef MLOCK_SUGGESTION

    void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) {
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

    void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) {
        fprintf(stderr, "warning: mlock not supported on this system\n");
        return false;
    }

    void raw_unlock(const void * addr, size_t len) {}
#endif
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct llama_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    llama_buffer() = default;

    void resize(size_t len) {
#ifdef GGML_USE_METAL
        free(addr);
        int result = posix_memalign((void **) &addr, getpagesize(), len);
        if (result == 0) {
            memset(addr, 0, len);
        }
        else {
            addr = NULL;
        }
#else
        delete[] addr;
        addr = new uint8_t[len];
#endif
        size = len;
    }

    ~llama_buffer() {
#ifdef GGML_USE_METAL
        free(addr);
#else
        delete[] addr;
#endif
        addr = NULL;
    }

    // disable copy and move
    llama_buffer(const llama_buffer&) = delete;
    llama_buffer(llama_buffer&&) = delete;
    llama_buffer& operator=(const llama_buffer&) = delete;
    llama_buffer& operator=(llama_buffer&&) = delete;
};

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
struct llama_ctx_buffer {
    uint8_t * addr = NULL;
    bool is_cuda;
    size_t size = 0;

    llama_ctx_buffer() = default;

    void resize(size_t size) {
        free();

        addr = (uint8_t *) ggml_cuda_host_malloc(size);
        if (addr) {
            is_cuda = true;
        }
        else {
            // fall back to pageable memory
            addr = new uint8_t[size];
            is_cuda = false;
        }
        this->size = size;
    }

    void free() {
        if (addr) {
            if (is_cuda) {
                ggml_cuda_host_free(addr);
            }
            else {
                delete[] addr;
            }
        }
        addr = NULL;
    }

    ~llama_ctx_buffer() {
        free();
    }

    // disable copy and move
    llama_ctx_buffer(const llama_ctx_buffer&) = delete;
    llama_ctx_buffer(llama_ctx_buffer&&) = delete;
    llama_ctx_buffer& operator=(const llama_ctx_buffer&) = delete;
    llama_ctx_buffer& operator=(llama_ctx_buffer&&) = delete;
};
#else
typedef llama_buffer llama_ctx_buffer;
#endif

struct llama_trie_node {
    llama_trie_node(): is_terminator(false) {}
    
    std::unordered_map<char, llama_trie_node*> children;
    bool is_terminator;
};

// Trie in C++. Creates a Trie out of a list of words. The trie is used to split on multiple delimiters in one pass
// Ported from: https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/tokenization_utils.py#L52
struct llama_trie {
public:
    llama_trie(): root_(new llama_trie_node()) {}

    void add(const std::string & word) {
        if (word.empty()) {
            return;
        }
            
        llama_trie_node *ref = root_;
        for (char c : word) {
            if (ref->children.find(c) == ref->children.end()) {
                ref->children[c] = new llama_trie_node();
            }
            ref = ref->children[c];
        }
        ref->is_terminator = true;
    }

    // Will look for the words added to the trie within `text`. Output is the boundaries of the words found.
    // Note that this trie will match the longest possible word first!
    std::vector<int> split(const std::string & text) const {
        std::map<int, llama_trie_node*> states;
        std::vector<int> offsets{0};

        int skip = 0;
        for (int current = 0; current < text.size(); current++) {
            char current_char = text[current];
            if (skip > 0 && current < skip) {
                // Prevents the lookahead for matching twice
                // like extra_id_100 and id_100
                continue;
            }

            // Whenever we found a match, we need to drop everything
            // this is a greedy algorithm, it will match on the first found token
            bool reset = false;

            // In this case, we already have partial matches (But unfinished)
            for (auto state = states.begin(); state != states.end(); ) {
                int start = state->first;
                llama_trie_node *trie_pointer = state->second;
                if (trie_pointer->is_terminator) {
                    // This is a final match, we need to reset and
                    // store the results in `offsets`.

                    // Lookahead to match longest first
                    // Important in case of extra_id_1 vs extra_id_100
                    // Here we are also actively looking for other earlier partial
                    // matches
                    // "[CLS]", "L", we need to match CLS even if L is special
                    int end = 0;
                    for (const auto & look : states) {
                        int lookstart = look.first;
                        llama_trie_node *looktrie_pointer = look.second;
                        int lookahead_index = 0;
                        if (lookstart > start) {
                            // This partial match is later, we can stop looking
                            break;
                        }
                        if (lookstart < start) {
                            // This partial match is earlier, the trie pointer
                            // was already updated, so index is + 1
                            lookahead_index = current + 1;
                            end = current + 1;
                        } else {
                            // Here lookstart == start and
                            //      looktrie_pointer == trie_pointer
                            // It wasn't updated yet so indices are current ones
                            lookahead_index = current;
                            end = current;
                        }
                        char next_char = lookahead_index < text.size() ? text[lookahead_index] : '\0';
                        if (looktrie_pointer->is_terminator) {
                            start = lookstart;
                            end = lookahead_index;
                            skip = lookahead_index;
                        }
                        
                        auto looktrie_pointer_it = looktrie_pointer->children.find(next_char);
                        while (looktrie_pointer_it != looktrie_pointer->children.end()) {
                            looktrie_pointer = looktrie_pointer_it->second;
                            lookahead_index++;
                            if (looktrie_pointer->is_terminator) {
                                start = lookstart;
                                end = lookahead_index;
                                skip = lookahead_index;
                            }
                            
                            if (lookahead_index == text.size()) {
                                // End of string
                                break;
                            }
                            next_char = text[lookahead_index];
                            looktrie_pointer_it = looktrie_pointer->children.find(next_char);
                        }
                    }
                    
                    offsets.push_back(start);
                    offsets.push_back(end);
                    reset = true;
                    break;
                } 
                
                auto trie_pointer_it = trie_pointer->children.find(current_char);
                if (trie_pointer_it != trie_pointer->children.end()) {
                    // The current character being looked at has a match within the trie
                    // update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer_it->second;
                    states[start] = trie_pointer;
                    ++state;
                } else {
                    // The new character has not match in the trie, we need
                    // to stop keeping track of this partial match.
                    state = states.erase(state);
                }
            }
            
            if (reset) {
                // Clear the full start (we found a real match)
                states.clear();
            }
            
            // If this character is a starting character within the trie
            // start keeping track of this partial match.
            auto children_it = root_->children.find(current_char);
            if (current >= skip && children_it != root_->children.end()) {
                states[current] = children_it->second;
            }
        }
        
        // We have a cut at the end with states.
        for (const auto & state : states) {
            int start = state.first;
            llama_trie_node *trie_pointer = state.second;
            if (trie_pointer->is_terminator) {
                // This is a final match, we need to reset and
                // store the results in `offsets`.
                int end = text.size();
                offsets.push_back(start);
                offsets.push_back(end);
                break;
            }
        }
        
        offsets.push_back(text.size());
        return offsets;
    }

private:
    llama_trie_node *root_;
};

#endif
