/* GCC 5 cannot parse C11 _Atomic with -fopenmp. These GCC atomics
 * preserve the exact memory-order contract without changing scheduling. */
#pragma once
#include <stdbool.h>
typedef int atomic_int;
typedef bool atomic_bool;
typedef unsigned char atomic_flag;
#define ATOMIC_FLAG_INIT 0
#define memory_order_relaxed __ATOMIC_RELAXED
#define memory_order_consume __ATOMIC_CONSUME
#define memory_order_acquire __ATOMIC_ACQUIRE
#define memory_order_release __ATOMIC_RELEASE
#define memory_order_acq_rel __ATOMIC_ACQ_REL
#define memory_order_seq_cst __ATOMIC_SEQ_CST
#define atomic_load_explicit(p,o) __atomic_load_n((p),(o))
#define atomic_store_explicit(p,v,o) __atomic_store_n((p),(v),(o))
#define atomic_fetch_add_explicit(p,v,o) __atomic_fetch_add((p),(v),(o))
#define atomic_load(p) atomic_load_explicit((p),memory_order_seq_cst)
#define atomic_store(p,v) atomic_store_explicit((p),(v),memory_order_seq_cst)
#define atomic_fetch_add(p,v) atomic_fetch_add_explicit((p),(v),memory_order_seq_cst)
#define atomic_thread_fence(o) __atomic_thread_fence(o)
#define atomic_flag_test_and_set(p) __atomic_test_and_set((p),memory_order_seq_cst)
#define atomic_flag_clear(p) __atomic_clear((p),memory_order_seq_cst)
