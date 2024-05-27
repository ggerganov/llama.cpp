#include <stdint.h>
#include <llama.h>

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#ifdef  __cplusplus
extern "C" {
#endif

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;

extern void atomic_store(atomic_int* ptr, LONG val);
extern LONG atomic_load(atomic_int* ptr);
extern LONG atomic_fetch_add(atomic_int* ptr, LONG inc);
extern LONG atomic_fetch_sub(atomic_int* ptr, LONG dec);

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;

extern int pthread_create(pthread_t* out, void* unused, thread_ret_t(*func)(void*), void* arg);
extern int pthread_join(pthread_t thread, void* unused);

extern int sched_yield(void);

#else
#include <pthread.h>
#include <stdatomic.h>

typedef void* thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

// barrier via spin lock
extern void ggml_critical_section_start(void);
extern void ggml_critical_section_end(void);

extern void ggml_numa_zero();

extern void ggml_numa_init(enum ggml_numa_strategy numa_flag);


//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//

#ifdef __APPLE__

//#include <os/lock.h>
//
//typedef os_unfair_lock ggml_lock_t;
//
//#define ggml_lock_init(x)    UNUSED(x)
//#define ggml_lock_destroy(x) UNUSED(x)
//#define ggml_lock_lock       os_unfair_lock_lock
//#define ggml_lock_unlock     os_unfair_lock_unlock
//
//#define GGML_LOCK_INITIALIZER OS_UNFAIR_LOCK_INIT

typedef int ggml_lock_t;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x)    UNUSED(x)
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

//typedef pthread_spinlock_t ggml_lock_t;

//#define ggml_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
//#define ggml_lock_destroy pthread_spin_destroy
//#define ggml_lock_lock    pthread_spin_lock
//#define ggml_lock_unlock  pthread_spin_unlock

typedef int ggml_lock_t;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

extern void set_numa_thread_affinity(int thread_n);
extern void clear_numa_thread_affinity(void);

#ifdef  __cplusplus
}
#endif
