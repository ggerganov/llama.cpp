#pragma once

#include <stddef.h>
#include <stdint.h>
#include <fcntl.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define NEED_WIN32_MMAP
#include <Windows.h>
#include <io.h>

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

#ifndef MS_ASYNC
#define MS_ASYNC 1
#endif
#ifndef MS_INVALIDATE
#define MS_INVALIDATE 2
#endif
#ifndef MS_SYNC
#define MS_SYNC 4
#endif

#ifndef SEEK_SET
#define SEEK_SET 0
#endif
#ifndef SEEK_CUR
#define SEEK_CUR 1
#endif
#ifndef SEEK_END
#define SEEK_END 2
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
#ifndef lseek
#define lseek WinSeek
#endif
#ifndef msync
#define msync WinMsync
#endif
#ifndef madvise
#define madvise WinMadvise
#endif
#ifndef ftruncate
#define ftruncate WinFtruncate
#endif

uint64_t WinSeek(int, uint64_t, int);
int WinMsync(void *, uintptr_t, int);
int WinMadvise(void *, uintptr_t, int);
int WinFtruncate(int, uint64_t);
int WinUnmap(void *, uintptr_t);
void *WinMap(void *, uintptr_t, int, int, int, uint64_t);

#else // _MSC_VER

#include <unistd.h>
#include <sys/mman.h>

#ifndef MAP_ANONYMOUS
#define NEED_POSIX_MMAP
#define mmap PosixMmap
#define MAP_ANONYMOUS 0x10000000
void *PosixMmap(void*, size_t, int, int, int, off_t);
#endif // MAP_ANONYMOUS

#endif // _MSC_VER

#ifdef __cplusplus
}
#endif
