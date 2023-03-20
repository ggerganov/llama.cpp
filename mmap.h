#pragma once

// portable mmap() implementation
//
// - supports win32 (needs vista+)
// - supports posix.1 (no map_anonymous)
//
// notes on windows
//
// - no errno support
// - not designed to support mprotect()
// - very poor support for memory intervals

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>

#if defined(_MSC_VER) || defined(__MINGW32__)
#ifndef __MINGW32__
#include <Windows.h>
#include <strsafe.h>
#else
#include <windows.h>
#endif
#include <io.h>
#include <atomic>
#include <map>

#ifndef PROT_READ
#define PROT_READ 1
#endif
#ifndef PROT_WRITE
#define PROT_WRITE 2
#endif
#ifndef PROT_EXEC
#define PROT_EXEC 4
#endif

#ifndef MAP_SHARED
#define MAP_SHARED 1
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 2
#endif
#ifndef MAP_FIXED
#define MAP_FIXED 16
#endif
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 32
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void*)-1)
#endif

#ifndef O_RDONLY
#define O_RDONLY _O_RDWR  // intentional smudge for mmap()
#endif
#ifndef O_WRONLY
#define O_WRONLY _O_WRONLY
#endif
#ifndef O_RDWR
#define O_RDWR _O_RDWR
#endif
#ifndef O_CREAT
#define O_CREAT _O_CREAT
#endif
#ifndef O_TRUNC
#define O_TRUNC _O_TRUNC
#endif
#ifndef O_EXCL
#define O_EXCL _O_EXCL
#endif

#ifndef MADV_NORMAL
#define MADV_NORMAL 0
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif
#ifndef MADV_RANDOM
#define MADV_RANDOM 1
#endif
#ifndef MADV_SEQUENTIAL
#define MADV_SEQUENTIAL 2
#endif
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif

#ifndef mmap
#define mmap WinMap
#endif
#ifndef munmap
#define munmap WinUnmap
#endif
#ifndef open
#define open _open
#endif
#ifndef close
#define close _close
#endif
#ifndef fstat
#define fstat _fstati64
#endif
#ifndef madvise
#define madvise WinMadvise
#endif
#ifndef ftruncate
#define ftruncate WinFtruncate
#endif

static std::atomic<unsigned> g_winlock;
static std::map<LPVOID, HANDLE> g_winmap;


static void WinLock(void) {
    while (!g_winlock.exchange(1, std::memory_order_acquire));
}

static void WunLock(void) {
    g_winlock.store(0, std::memory_order_release);
}

static int WinMadvise(char* fd, size_t length, int flags) {
    auto p_handle = GetCurrentProcess();
    struct _WIN32_MEMORY_RANGE_ENTRY entry((void*)fd, length);
    bool success = PrefetchVirtualMemory(p_handle, 1, &entry, 0);
    if (!success) {
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD error_code = GetLastError();
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);
        lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 256) * sizeof(TCHAR));
        StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("%s failed with error %d: %s"),
            error_code, lpMsgBuf);
    }
    return 0;
}

static int WinFtruncate(int fd, uint64_t length) {
    return _chsize_s(fd, length) ? -1 : 0;
}

static int WinUnmap(void *addr, size_t length) {
    HANDLE hand;
    WinLock();
    hand = g_winmap[addr];
    g_winmap[addr] = 0;
    WunLock();
    if (hand) {
        UnmapViewOfFile(addr);
        CloseHandle(hand);
        return 0;
    } else {
        return -1;
    }
}

static void *WinMap(void *addr, size_t length, int prot,
                    int flags, int fd, uint64_t offset) {
    HANDLE hFile;
    DWORD winprot;
    DWORD access = 0;
    HANDLE hand = NULL;
    LPVOID res = NULL;
    if (prot & PROT_READ) {
        access |= FILE_MAP_READ;
    }
    if (prot & PROT_WRITE) {
        access |= FILE_MAP_WRITE;
    }
    if (prot & PROT_EXEC) {
        access |= FILE_MAP_EXECUTE;
    }
    if (flags & MAP_PRIVATE) {
        // private mapping
        if (prot & PROT_EXEC) {
            if (prot & PROT_WRITE) {
                if (flags & MAP_ANONYMOUS) {
                    winprot = PAGE_EXECUTE_READWRITE;
                } else {
                    winprot = PAGE_EXECUTE_WRITECOPY;
                }
            } else {
                winprot = PAGE_EXECUTE_READ;
            }
        } else if (prot & PROT_WRITE) {
            if (flags & MAP_ANONYMOUS) {
                winprot = PAGE_READWRITE;
            } else {
                winprot = PAGE_WRITECOPY;
            }
        } else {
            winprot = PAGE_READONLY;
        }
    } else {
        // shared mapping
        if (prot & PROT_EXEC) {
            if (prot & PROT_WRITE) {
                winprot = PAGE_EXECUTE_READWRITE;
            } else {
                winprot = PAGE_EXECUTE_READ;
            }
        } else if (prot & PROT_WRITE) {
            winprot = PAGE_READWRITE;
        } else {
            winprot = PAGE_READONLY;
        }
    }
    if (flags & MAP_ANONYMOUS) {
        hFile = INVALID_HANDLE_VALUE;
        offset = 0;
    } else {
        hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile == INVALID_HANDLE_VALUE) {
            return MAP_FAILED;
        }
    }
    if (flags & MAP_FIXED) {
        if (!addr) {
            return MAP_FAILED;
        } else {
            WinUnmap(addr, length);
        }
    }
    hand = CreateFileMapping(hFile, 0, winprot,
                             (offset + length) >> 32,
                             (offset + length), 0);
    if (!hand) {
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD error_code = GetLastError();
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);
        lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 256) * sizeof(TCHAR));
        StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("%s failed with error %d: %s"),
            error_code, lpMsgBuf);
        return MAP_FAILED;
    }
    if (winprot == PAGE_WRITECOPY) {
        access = FILE_MAP_COPY;
    }

    res = MapViewOfFileEx(hand, access, offset >> 32,
                          offset, length, addr);
    if (!res) {
        LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;
        DWORD error_code = GetLastError();
        FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            (LPTSTR)&lpMsgBuf,
            0, NULL);
        lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
            (lstrlen((LPCTSTR)lpMsgBuf) + 40) * sizeof(TCHAR));
        StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf) / sizeof(TCHAR),
            TEXT("failed with error %d: %s"),
            error_code, lpMsgBuf);
        fprintf(stderr, (char*)lpDisplayBuf);
        CloseHandle(hand);
        return MAP_FAILED;
    }
    WinLock();
    g_winmap[res] = hand;
    WunLock();
    return res;
}

#else
#include <unistd.h>
#include <sys/mman.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x10000000

static void *PosixMmap(void *addr, size_t length, int prot,
                       int flags, int fd, uint64_t offset) {
    int tfd;
    void *res;
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

#ifndef mmap
#define mmap PosixMmap
#endif
#endif // MAP_ANONYMOUS
#endif // _MSC_VER
