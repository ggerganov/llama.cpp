// GGUF counterpart of llama-util.h.
// we may consider making it a part of ggml.c once GGUF work is complete.
// this will require extra work to migrate this to pure C.
// Contains wrappers around OS interfaces.

#ifndef GGUF_UTIL_H
#define GGUF_UTIL_H
#include "ggml.h"
#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <climits>

#include <string>
#include <sstream>
#include <vector>
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
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}


template<typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

// TODO: can we merge this one and gguf_context?
struct gguf_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    gguf_file(const char * fname, const char * mode) {
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

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    
    void write_str(const std::string & val) {
        const int32_t n = val.size();
        fwrite((const char *) &n, sizeof(n), 1, fp);
        fwrite(val.c_str(), n, 1, fp);
    }

    void write_i32(int32_t val) {
        fwrite((const char *) &val, sizeof(val), 1, fp);
    }

    void write_u64(size_t val) {
        fwrite((const char *) &val, sizeof(val), 1, fp);
    }

    template<typename T>
    void write_val(const std::string & key, enum gguf_type type, const T & val) {
        write_str(key);
        fwrite((const char *) &type, sizeof(type), 1, fp);
        fwrite((const char *) &val, sizeof(val), 1, fp);
    }

    template<typename T>
    void write_arr(const std::string & key, enum gguf_type type, const std::vector<T> & val) {
        write_str(key);
        {
            const enum gguf_type tarr = GGUF_TYPE_ARRAY;
            fwrite((const char *) &tarr, sizeof(tarr), 1, fp);
        }

        const int32_t n = val.size();
        fwrite((const char *) &type, sizeof(type), 1, fp);
        fwrite((const char *) &n, sizeof(n), 1, fp);
        fwrite(val.data(), sizeof(T), n, fp);
    }
};

#if defined(_WIN32)
static std::string gguf_format_win_err(DWORD err) {
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

struct gguf_mmap {
    void * addr;
    size_t size;

    gguf_mmap(const gguf_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    gguf_mmap(struct gguf_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false) {
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

    ~gguf_mmap() {
        munmap(addr, size);
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    gguf_mmap(struct llama_file * file, bool prefetch = true, bool numa = false) {
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
                        gguf_format_win_err(GetLastError()).c_str());
            }
        }
        #else
        #pragma message("warning: You are building for pre-Windows 8; prefetch not supported")
        #endif // _WIN32_WINNT >= _WIN32_WINNT_WIN8
    }

    ~gguf_mmap() {
        if (!UnmapViewOfFile(addr)) {
            fprintf(stderr, "warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    gguf_mmap(struct llama_file *, bool prefetch = true, bool numa = false) {
        (void) prefetch;
        (void) numa;

        throw std::runtime_error(std::string("mmap not supported"));
    }
#endif
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct gguf_mlock {
    void * addr = NULL;
    size_t size = 0;
    bool failed_already = false;

    gguf_mlock() {}
    gguf_mlock(const gguf_mlock &) = delete;

    ~gguf_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
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
                        gguf_format_win_err(GetLastError()).c_str());
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
                        gguf_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            fprintf(stderr, "warning: failed to VirtualUnlock buffer: %s\n",
                    gguf_format_win_err(GetLastError()).c_str());
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
struct gguf_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    gguf_buffer() = default;

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

    ~gguf_buffer() {
#ifdef GGML_USE_METAL
        free(addr);
#else
        delete[] addr;
#endif
        addr = NULL;
    }

    // disable copy and move
    gguf_buffer(const gguf_buffer&) = delete;
    gguf_buffer(gguf_buffer&&) = delete;
    gguf_buffer& operator=(const gguf_buffer&) = delete;
    gguf_buffer& operator=(gguf_buffer&&) = delete;
};

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
struct gguf_ctx_buffer {
    uint8_t * addr = NULL;
    bool is_cuda;
    size_t size = 0;

    gguf_ctx_buffer() = default;

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

    ~gguf_ctx_buffer() {
        free();
    }

    // disable copy and move
    gguf_ctx_buffer(const gguf_ctx_buffer&) = delete;
    gguf_ctx_buffer(gguf_ctx_buffer&&) = delete;
    gguf_ctx_buffer& operator=(const gguf_ctx_buffer&) = delete;
    gguf_ctx_buffer& operator=(gguf_ctx_buffer&&) = delete;
};
#else
typedef gguf_buffer gguf_ctx_buffer;
#endif

#endif
