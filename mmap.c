// Lightweight Portable mmap() Polyfill
//
// 1. Supports POSIX.1
//
//    The baseline POSIX standard doesn't specify MAP_ANONYMOUS. This
//    library makes sure, on the hypothetical UNIX systems that don't
//    have it, or on the mainstream UNIX platforms where the user has
//    chosen to define _POSIX_C_SOURCE that cause headers to undefine
//    it, this implementation will fallback to creating a secure temp
//    file, for each anonymous mapping.
//
// 2. Supports Windows w/ Visual Studio
//
//    On Windows Vista and later an API exists that's almost as good as
//    mmap(). However code that uses this library should conform to the
//    subset of behaviors Microsoft accommodates.
//
// Caveats
//
// - You should just assume the page size is 64kb. That's how it is on
//   Windows and it usually goes faster to assume that elsewhere too.
//
// - Not designed to support mprotect() at the moment. In order to
//   support this, we'd need to consider _open(O_ACCMODE) on Windows
//   and then have mmap() be more greedy about permissions.
//
// - There's limited support for being clever with memory intervals.
//   For example, you can't punch a hole in a memory map on Windows.
//   This abstraction does aim to offer more flexibility than WIN32.
//   There should also be good error reporting for unsupported uses.

#include "mmap.h"

#ifdef NEED_POSIX_MMAP
#include <stdlib.h>

void *PosixMmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    int tfd;
    void* res;
    char path[] = "/tmp/llama.dat.XXXXXX";
    if (~flags & MAP_ANONYMOUS) {
        res = mmap(addr, length, prot, flags, fd, offset);
    } else if ((tfd = mkstemp(path)) != -1) {
        unlink(path);
        if (!ftruncate(tfd, length)) {
            res = mmap(addr, length, prot, flags & ~MAP_ANONYMOUS, tfd, 0);
        } else {
            res = MAP_FAILED;
        }
        close(tfd);
    } else {
        res = MAP_FAILED;
    }
    return res;
}

#elif defined(NEED_WIN32_MMAP)
#include <errno.h>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>

struct WinMap {        // O(n) no ordering no overlaps
    HANDLE hand;       // zero means array slots empty
    HANDLE fand;       // for the original file, or -1
    uintptr_t addr;    // base address (64 kb aligned)
    uintptr_t length;  // byte size (>0, rounded 64kb)
};

struct WinMaps {
    int n;
    struct WinMap *p;
    volatile long lock;
};

static struct WinMaps g_winmaps;

static inline uintptr_t Min(uintptr_t x, uintptr_t y) {
    return y > x ? x : y;
}

static inline uintptr_t Max(uintptr_t x, uintptr_t y) {
    return y < x ? x : y;
}

static inline uintptr_t Roundup(uintptr_t x, intptr_t a) {
    assert(a > 0);
    assert(!(a & (a - 1)));
    return (x + (a - 1)) & -a;
}

static inline void Lock(void) {
    long x;
    for (;;) {
        x = InterlockedExchange(&g_winmaps.lock, 1);
        if (!x) break;
        assert(x == 1);
    }
}

static inline void Unlock(void) {
    assert(g_winmaps.lock == 1);
    g_winmaps.lock = 0;
}

static int WinStrerror(int err, char *buf, int size) {
    return FormatMessageA(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        buf, size, NULL);
}

#ifdef NDEBUG
#define LogError(thing) (void)0
#else
static void LogError(const char* file, int line, const char* thing) {
#define LogError(thing) LogError(__FILE__, __LINE__, thing)
    fprintf(stderr, "%s:%d: error: %s\n", file, line, thing);
}
#endif

#ifdef NDEBUG
#define LogWindowsError(thing) (void)0
#else
static void LogWindowsError(const char* file, int line, const char* thing) {
#define LogWindowsError(thing) LogWindowsError(__FILE__, __LINE__, thing)
    char s[256];
    int e = GetLastError();
    WinStrerror(e, s, sizeof(s));
    fprintf(stderr, "%s:%d: error[%#x]: %s failed: %s\n", file, line, e, thing, s);
}
#endif

static void *Recalloc(void *ptr, uint64_t newSize) {
    HANDLE heap = GetProcessHeap();
    if (!ptr) {
        return HeapAlloc(heap, HEAP_ZERO_MEMORY, newSize);
    }
    if (!newSize) {
        HeapFree(heap, 0, ptr);
        return 0;
    }
    return HeapReAlloc(heap, HEAP_ZERO_MEMORY, ptr, newSize);
}

uint64_t WinSeek(int fd, uint64_t offset, int whence) {
    HANDLE hFile;
    DWORD winwhence;
    LARGE_INTEGER distanceToMove;
    LARGE_INTEGER newFilePointer;
    distanceToMove.QuadPart = offset;
    switch (whence) {
    case SEEK_SET:
        winwhence = FILE_BEGIN;
        break;
    case SEEK_CUR:
        winwhence = FILE_CURRENT;
        break;
    case SEEK_END:
        winwhence = FILE_END;
        break;
    default:
        LogError("bad lseek() whence");
        errno = EINVAL;
        return -1;
    }
    hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) {
        LogWindowsError("_get_osfhandle");
        errno = EBADF;
        return -1;
    }
    if (GetFileType(hFile) != FILE_TYPE_DISK) {
        LogError("bad file type for lseek()");
        errno = ESPIPE;
        return -1;
    }
    if (!SetFilePointerEx(hFile, distanceToMove, &newFilePointer, winwhence)) {
        LogWindowsError("SetFilePointerEx");
        errno = EPERM;
        return -1;
    }
    return newFilePointer.QuadPart;
}

int WinFtruncate(int fd, uint64_t length) {
    HANDLE hFile;
    LARGE_INTEGER old, neu;
    hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) {
        LogWindowsError("_get_osfhandle");
        errno = EBADF;
        return -1;
    }
    // save current file position
    old.QuadPart = 0;
    neu.QuadPart = 0;
    if (!SetFilePointerEx(hFile, neu, &old, FILE_CURRENT)) {
        LogWindowsError("SetFilePointerEx#1");
        return -1;
    }
    // set current position to new file size
    neu.QuadPart = length;
    if (!SetFilePointerEx(hFile, neu, NULL, FILE_BEGIN)) {
        LogWindowsError("SetFilePointerEx#2");
        return -1;
    }
    // change the file size
    if (!SetEndOfFile(hFile)) {
        LogWindowsError("SetEndOfFile");
        SetFilePointerEx(hFile, old, NULL, FILE_BEGIN);
        return -1;
    }
    // restore the original file position
    // win32 allows this to exceed the end of file
    if (!SetFilePointerEx(hFile, old, NULL, FILE_BEGIN)) {
        LogWindowsError("SetFilePointerEx>3");
        return -1;
    }
    return 0;
}

int WinMadvise(void *addr, uintptr_t length, int advice) {
    switch (advice) {
    case MADV_NORMAL:
    case MADV_DONTNEED:
    case MADV_SEQUENTIAL:
        return 0;
    case MADV_RANDOM:
    case MADV_WILLNEED: {
        HANDLE proc;
        WIN32_MEMORY_RANGE_ENTRY entry;
        proc = GetCurrentProcess();
        entry.VirtualAddress = addr;
        entry.NumberOfBytes = length;
        if (!PrefetchVirtualMemory(proc, 1, &entry, 0)) {
            LogWindowsError("PrefetchVirtualMemory");
            errno = ENOMEM;
            return -1;
        }
        return 0;
    }
    default:
        errno = EINVAL;
        return -1;
    }
}

int WinUnmap(void *addr, uintptr_t length) {
    void *view;
    HANDLE hand;
    HANDLE fand;
    int i, err = 0;
    uintptr_t a, b;
    uintptr_t x, y;
    // compute the requested interval
    // 1. length can't be zero
    // 2. length is rounded up to the page size
    // 3. addr must be aligned to page boundary
    a = (uintptr_t)addr;
    b = a + Roundup(length, 65536);
    if (!length) {
        LogError("tried to munmap zero bytes");
        errno = EINVAL;
        return -1;
    }
    if (a & 65535) {
        LogError("tried to munmap an address that's not 64kb aligned");
        errno = EINVAL;
        return -1;
    }
    // 1. we permit unmapping multiple maps in one call
    // 2. we don't care if the matched mappings aren't contiguous
    // 3. it's an error if a matched mapping only partially overlaps
    // 4. similar to close() we release all resources possible on error
    Lock();
    for (i = 0; i < g_winmaps.n; ++i) {
        if (!g_winmaps.p[i].hand) {
            // this array slot is empty
            continue;
        }
        // compute overlap between known mapping and requested interval
        x = Max(a, g_winmaps.p[i].addr);
        y = Min(b, g_winmaps.p[i].addr + g_winmaps.p[i].length);
        if (x >= y) {
            // there isn't any overlap
            continue;
        }
        if (y - x != g_winmaps.p[i].length) {
            // requested interval partially overlapped this mapping
            // therefore we can't unmap it and must report an error
            LogError("tried to partially unmap a mapping");
            err = ENOMEM;
            continue;
        }
        // save the information we care about
        view = (void *)g_winmaps.p[i].addr;
        hand = g_winmaps.p[i].hand;
        fand = g_winmaps.p[i].fand;
        // delete this mapping from the global array
        g_winmaps.p[i].hand = 0;
        // perform the systems operations
        // safe to release lock since g_winmaps.n is monotonic
        Unlock();
        if (!UnmapViewOfFile(view)) {
            LogWindowsError("UnmapViewOfFile");
        }
        if (!CloseHandle(hand)) {
            LogWindowsError("CloseHandle#1");
        }
        if (fand != INVALID_HANDLE_VALUE) {
            if (!CloseHandle(fand)) {
                LogWindowsError("CloseHandle#2");
            }
        }
        Lock();
    }
    Unlock();
    if (err) {
        errno = err;
        return -1;
    }
    return 0;
}

void* WinMap(void *addr, uintptr_t length, int prot, int flags, int fd, uint64_t offset) {
    int i;
    LPVOID res;
    HANDLE hand;
    HANDLE hFile;
    DWORD access;
    DWORD wiprot;
    uintptr_t fsize;
    if (!length) {
        LogError("mmap(length) was zero");
        errno = EINVAL;
        return MAP_FAILED;
    }
    length = Roundup(length, 65536);
    if ((uintptr_t)addr & 65535) {
        if (~flags & MAP_FIXED) {
            addr = 0;
        } else {
            LogError("MAP_FIXED used with address that's not 64kb aligned");
            errno = EINVAL;
            return MAP_FAILED;
        }
    }
    // these are the logical flag equivalents for creating mappings.  please
    // note that any subsequent virtualprotect calls must be a subset of the
    // permissions we're using here.  that's not a supported use case for us
    if (flags & MAP_PRIVATE) {
        // private mapping
        if (prot & PROT_EXEC) {
            if (prot & PROT_WRITE) {
                if (flags & MAP_ANONYMOUS) {
                    wiprot = PAGE_EXECUTE_READWRITE;
                    access = FILE_MAP_READ | FILE_MAP_WRITE | FILE_MAP_EXECUTE;
                } else {
                    wiprot = PAGE_EXECUTE_WRITECOPY;
                    access = FILE_MAP_COPY | FILE_MAP_EXECUTE;
                }
            } else {
                wiprot = PAGE_EXECUTE_READ;
                access = FILE_MAP_READ | FILE_MAP_EXECUTE;
            }
        } else if (prot & PROT_WRITE) {
            if (flags & MAP_ANONYMOUS) {
                wiprot = PAGE_READWRITE;
                access = FILE_MAP_READ | FILE_MAP_WRITE;
            } else {
                wiprot = PAGE_WRITECOPY;
                access = FILE_MAP_COPY;
            }
        } else {
            wiprot = PAGE_READONLY;
            access = FILE_MAP_READ;
        }
    } else {
        // shared mapping
        if (prot & PROT_EXEC) {
            if (prot & PROT_WRITE) {
                wiprot = PAGE_EXECUTE_READWRITE;
                access = FILE_MAP_READ | FILE_MAP_WRITE | FILE_MAP_EXECUTE;
            } else {
                wiprot = PAGE_EXECUTE_READ;
                access = FILE_MAP_READ | FILE_MAP_EXECUTE;
            }
        } else if (prot & PROT_WRITE) {
            wiprot = PAGE_READWRITE;
            access = FILE_MAP_READ | FILE_MAP_WRITE;
        } else {
            wiprot = PAGE_READONLY;
            access = FILE_MAP_READ;
        }
    }
    if (flags & MAP_ANONYMOUS) {
        hFile = INVALID_HANDLE_VALUE;
        fsize = length;
        offset = 0;
    } else {
        fsize = 0;
        hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile == INVALID_HANDLE_VALUE) {
            LogWindowsError("_get_osfhandle");
            errno = EBADF;
            return MAP_FAILED;
        }
        if (!DuplicateHandle(GetCurrentProcess(), hFile,
                             GetCurrentProcess(), &hFile,
                             0, FALSE, DUPLICATE_SAME_ACCESS)) {
            LogWindowsError("DuplicateHandle");
            errno = EBADF;
            return MAP_FAILED;
        }
    }
    if (flags & MAP_FIXED) {
        if (!addr) {
            // zero chance of microsoft letting us map the null page
            if (hFile != INVALID_HANDLE_VALUE) {
                CloseHandle(hFile);
            }
            errno = EINVAL;
            return MAP_FAILED;
        } else {
            // blow away any existing mappings on requested interval
            if (WinUnmap(addr, length) == -1) {
                // can only happen if we partially overlap an existing mapping
                assert(errno == ENOMEM);
                if (hFile != INVALID_HANDLE_VALUE) {
                    CloseHandle(hFile);
                }
                return MAP_FAILED;
            }
        }
    }
    hand = CreateFileMapping(hFile, 0, wiprot,
                             (DWORD)(fsize >> 32),
                             (DWORD)fsize,
                             0);
    if (!hand) {
        LogWindowsError("CreateFileMapping");
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
        errno = EPERM;
        return MAP_FAILED;
    }
    res = MapViewOfFileEx(hand, access,
                          (DWORD)(offset >> 32),
                          (DWORD)offset,
                          length, addr);
    if (!res) {
        LogWindowsError("MapViewOfFileEx");
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
        CloseHandle(hand);
        errno = EPERM;
        return MAP_FAILED;
    }
    if (flags & MAP_FIXED) {
        // this assertion could legitimately fail if two threads engage in a
        // race to create a MAP_FIXED mapping at the same address and that's
        // certainly not the kind of use case we're designed to support here
        assert(res == addr);
    }
    // record our new mapping in the global array
    Lock();
    for (i = 0; i < g_winmaps.n; ++i) {
        if (!g_winmaps.p[i].hand) {
            // we found an empty slot
            break;
        }
    }
    if (i == g_winmaps.n) {
        // we need to grow the array
        // it's important to use kernel32 memory
        // our malloc implementation depends on this
        int n2;
        struct WinMap *p2;
        p2 = g_winmaps.p;
        n2 = g_winmaps.n;
        if (n2) {
            n2 += n2 >> 1;
        } else {
            n2 = 7;
        }
        if ((p2 = (struct WinMap*)Recalloc(p2, n2 * sizeof(*p2)))) {
            g_winmaps.p = p2;
            g_winmaps.n = n2;
        } else {
            Unlock();
            LogError("recalloc failed");
            UnmapViewOfFile(res);
            CloseHandle(hand);
            if (hFile != INVALID_HANDLE_VALUE) {
                CloseHandle(hFile);
            }
            errno = ENOMEM;
            return MAP_FAILED;
        }
    }
    g_winmaps.p[i].hand = hand;
    g_winmaps.p[i].fand = hFile;
    g_winmaps.p[i].addr = (uintptr_t)res;
    g_winmaps.p[i].length = length;
    Unlock();
    return res;
}

int WinMsync(void *addr, uintptr_t length, int flags) {
    int i, err;
    HANDLE hand;
    uintptr_t x, y;
    if (flags & ~(MS_ASYNC | MS_INVALIDATE | MS_SYNC)) {
        LogError("bad msync flags");
        errno = EINVAL;
        return -1;
    }
    // 1. we do nothing if length is zero (unlike win32 api)
    // 2. the requested interval may envelop multiple known mappings
    // 3. we don't care if those mappings aren't contiguous or a hole exists
    // 4. the requested interval may specify a subrange of any given mapping
    Lock();
    for (err = i = 0; i < g_winmaps.n; ++i) {
        if (!g_winmaps.p[i].hand) {
            // this array slot is empty
            continue;
        }
        // compute overlap between known mapping and requested interval
        x = Max((uintptr_t)addr, g_winmaps.p[i].addr);
        y = Min((uintptr_t)addr + length, g_winmaps.p[i].addr + g_winmaps.p[i].length);
        if (x >= y) {
            // there isn't any overlap
            continue;
        }
        // it's safe to release lock temporarily, since g_winmaps.n is monotonic
        // any race conditions in handle being deleted should be caught by win32
        hand = g_winmaps.p[i].fand;
        Unlock();
        // ensure coherency and that filesystem flush *will* happen
        if (!FlushViewOfFile((void*)x, y - x)) {
            LogWindowsError("FlushViewOfFile");
            err = EPERM;
        }
        if (flags & MS_SYNC) {
            // ensure that filesystem flush *has* happened
            if (!FlushFileBuffers(hand)) {
                LogWindowsError("FlushFileBuffers");
                err = EPERM;
            }
        }
        Lock();
    }
    Unlock();
    if (err) {
        errno = err;
        return -1;
    }
    return 0;
}

#else // NEED_*_MAP

// this is a normal unix platform
// add some content to this object so the apple linker doesn't whine
int justine_mmap_module;

#endif // NEED_*_MMAP
