/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef _CUDA_AWBARRIER_HELPERS_H_
# define _CUDA_AWBARRIER_HELPERS_H_

# define _CUDA_AWBARRIER_NAMESPACE       nvcuda::experimental
# define _CUDA_AWBARRIER_BEGIN_NAMESPACE namespace nvcuda { namespace experimental {
# define _CUDA_AWBARRIER_END_NAMESPACE   } }

# define _CUDA_AWBARRIER_INTERNAL_NAMESPACE       _CUDA_AWBARRIER_NAMESPACE::__awbarrier_internal
# define _CUDA_AWBARRIER_BEGIN_INTERNAL_NAMESPACE _CUDA_AWBARRIER_BEGIN_NAMESPACE namespace __awbarrier_internal {
# define _CUDA_AWBARRIER_END_INTERNAL_NAMESPACE   } _CUDA_AWBARRIER_END_NAMESPACE

# if !defined(_CUDA_AWBARRIER_QUALIFIER)
#  define _CUDA_AWBARRIER_QUALIFIER inline __device__
# endif
# if !defined(_CUDA_AWBARRIER_STATIC_QUALIFIER)
#  define _CUDA_AWBARRIER_STATIC_QUALIFIER static inline __device__
# endif

# if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
#  define _CUDA_AWBARRIER_ARCH_700_OR_LATER
# endif

# define _CUDA_AWBARRIER_MAX_COUNT ((1 << 14) - 1)

# if (__CUDA_ARCH__ >= 800)
#  define _CUDA_AWBARRIER_HAS_HW_MBARRIER 1
# else
#  define _CUDA_AWBARRIER_HAS_HW_MBARRIER 0
# endif

# if defined(__cplusplus) && ((__cplusplus >= 201103L) || (defined(_MSC_VER) && (_MSC_VER >= 1900)))
#  define _CUDA_AWBARRIER_CPLUSPLUS_11_OR_LATER
# endif

# if !defined(_CUDA_AWBARRIER_DEBUG)
#  if defined(__CUDACC_DEBUG__)
#   define _CUDA_AWBARRIER_DEBUG 1
#  else
#   define _CUDA_AWBARRIER_DEBUG 0
#  endif
# endif

# if defined(_CUDA_AWBARRIER_DEBUG) && (_CUDA_AWBARRIER_DEBUG == 1) && !defined(NDEBUG)
#  if !defined(__CUDACC_RTC__)
#   include <cassert>
#  endif
#  define _CUDA_AWBARRIER_ASSERT(x) assert((x));
#  define _CUDA_AWBARRIER_ABORT() assert(0);
# else
#  define _CUDA_AWBARRIER_ASSERT(x)
#  define _CUDA_AWBARRIER_ABORT() __trap();
# endif


# if defined(_MSC_VER) && !defined(_WIN64)
#  define _CUDA_AWBARRIER_ASM_PTR_CONSTRAINT "r"
# else
#  define _CUDA_AWBARRIER_ASM_PTR_CONSTRAINT "l"
# endif

# if defined(__CUDACC_RTC__)
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef uint64_t           uintptr_t;
# else
#  include <stdint.h>
# endif

# if defined(_CUDA_AWBARRIER_ARCH_700_OR_LATER)

_CUDA_AWBARRIER_BEGIN_INTERNAL_NAMESPACE

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

template<bool UseHWAtomicArrive>
struct ImplementationChooser;

template<>
struct ImplementationChooser<true> {
    _CUDA_AWBARRIER_STATIC_QUALIFIER
    void awbarrier_init(uint64_t* barrier, uint32_t expected_count)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));
        _CUDA_AWBARRIER_ASSERT(expected_count > 0 && expected_count < (1 << 29));

        asm volatile ("mbarrier.init.shared.b64 [%0], %1;"
            :
            : "r"(__nvvm_get_smem_pointer(barrier)), "r"(expected_count)
            : "memory");
    }

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    void awbarrier_inval(uint64_t* barrier)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));

        asm volatile ("mbarrier.inval.shared.b64 [%0];"
            :
            : "r"(__nvvm_get_smem_pointer(barrier))
            : "memory");
    }

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    uint32_t awbarrier_token_pending_count(uint64_t token)
    {
        uint32_t pending_count;

        asm ("mbarrier.pending_count.b64 %0, %1;"
            : "=r"(pending_count)
            : "l"(token));
        return pending_count;
    }

    template<bool Drop>
    _CUDA_AWBARRIER_STATIC_QUALIFIER
    uint64_t awbarrier_arrive_drop(uint64_t* barrier)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));

        uint64_t token;

        if (Drop) {
            asm volatile ("mbarrier.arrive_drop.shared.b64 %0, [%1];"
                : "=l"(token)
                : "r"(__nvvm_get_smem_pointer(barrier))
                          : "memory");
        } else {
            asm volatile ("mbarrier.arrive.shared.b64 %0, [%1];"
                : "=l"(token)
                : "r"(__nvvm_get_smem_pointer(barrier))
                : "memory");
        }

        return token;
    }

    template<bool Drop>
    _CUDA_AWBARRIER_STATIC_QUALIFIER
    uint64_t awbarrier_arrive_drop_no_complete(uint64_t* barrier, uint32_t count)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));
        _CUDA_AWBARRIER_ASSERT(count > 0 && count < (1 << 29));

        uint64_t token;

        if (Drop) {
            asm volatile ("mbarrier.arrive_drop.noComplete.shared.b64 %0, [%1], %2;"
                : "=l"(token)
                : "r"(__nvvm_get_smem_pointer(barrier)), "r"(count)
                : "memory");
        } else {
            asm volatile ("mbarrier.arrive.noComplete.shared.b64 %0, [%1], %2;"
                : "=l"(token)
                : "r"(__nvvm_get_smem_pointer(barrier)), "r"(count)
                : "memory");
        }

        return token;
    }

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    bool awbarrier_test_wait(uint64_t* barrier, uint64_t token)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));

        uint16_t wait_complete;

        asm volatile ("{"
                      "    .reg .pred %%p;"
                      "    mbarrier.test_wait.shared.b64 %%p, [%1], %2;"
                      "    selp.u16 %0, 1, 0, %%p;"
                      "}"
            : "=h"(wait_complete)
            : "r"(__nvvm_get_smem_pointer(barrier)), "l"(token)
            : "memory");
        return bool(wait_complete);
    }
};

template<>
struct ImplementationChooser<false> {
    union AWBarrier {
        struct {
            uint32_t expected;
            uint32_t pending;
        } split;
        uint64_t raw;
    };

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    void awbarrier_init(uint64_t* barrier, uint32_t expected_count)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));
        _CUDA_AWBARRIER_ASSERT(expected_count > 0 && expected_count < (1 << 29));

        AWBarrier* awbarrier = reinterpret_cast<AWBarrier*>(barrier);

        awbarrier->split.expected = 0x40000000 - expected_count;
        awbarrier->split.pending = 0x80000000 - expected_count;
    }

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    void awbarrier_inval(uint64_t* barrier)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));
    }

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    uint32_t awbarrier_token_pending_count(uint64_t token)
    {
        const uint32_t pending = token >> 32;
        return 0x80000000 - (pending & 0x7fffffff);
    }

    template<bool Drop>
    _CUDA_AWBARRIER_STATIC_QUALIFIER
    uint64_t awbarrier_arrive_drop(uint64_t* barrier)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));

        AWBarrier* awbarrier = reinterpret_cast<AWBarrier*>(barrier);

        while ((*reinterpret_cast<volatile uint32_t*>(&awbarrier->split.pending) & 0x7fffffff) == 0);

        if (Drop) {
            (void)atomicAdd_block(&awbarrier->split.expected, 1);
        }

        __threadfence_block();

        const uint32_t old_pending = atomicAdd_block(&awbarrier->split.pending, 1);
        const uint32_t new_pending = old_pending + 1;
        const bool reset = (old_pending ^ new_pending) & 0x80000000;

        if (reset) {
            __threadfence_block();

            uint32_t new_expected = *reinterpret_cast<volatile uint32_t*>(&awbarrier->split.expected);
            new_expected &= ~0x40000000;
            if (new_expected & 0x20000000) {
                new_expected |= 0x40000000;
            }
            atomicAdd_block(&awbarrier->split.pending, new_expected);
        }

        return static_cast<uint64_t>(old_pending) << 32;
    }

    template<bool Drop>
    _CUDA_AWBARRIER_STATIC_QUALIFIER
    uint64_t awbarrier_arrive_drop_no_complete(uint64_t* barrier, uint32_t count)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));
        _CUDA_AWBARRIER_ASSERT(count > 0 && count < (1 << 29));

        AWBarrier* awbarrier = reinterpret_cast<AWBarrier*>(barrier);

        while ((*reinterpret_cast<volatile uint32_t*>(&awbarrier->split.pending) & 0x7fffffff) == 0);

        if (Drop) {
            (void)atomicAdd_block(&awbarrier->split.expected, count);
        }

        return static_cast<uint64_t>(atomicAdd_block(&awbarrier->split.pending, count)) << 32;
    }

    _CUDA_AWBARRIER_STATIC_QUALIFIER
    bool awbarrier_test_wait(uint64_t* barrier, uint64_t token)
    {
        _CUDA_AWBARRIER_ASSERT(__isShared(barrier));

        volatile AWBarrier* awbarrier = reinterpret_cast<volatile AWBarrier*>(barrier);

        return ((token >> 32) ^ awbarrier->split.pending) & 0x80000000;
    }
};

_CUDA_AWBARRIER_QUALIFIER
void awbarrier_init(uint64_t* barrier, uint32_t expected_count)
{
    ImplementationChooser<_CUDA_AWBARRIER_HAS_HW_MBARRIER>::awbarrier_init(barrier, expected_count);
}

_CUDA_AWBARRIER_QUALIFIER
void awbarrier_inval(uint64_t* barrier)
{
    ImplementationChooser<_CUDA_AWBARRIER_HAS_HW_MBARRIER>::awbarrier_inval(barrier);
}

_CUDA_AWBARRIER_QUALIFIER
uint32_t awbarrier_token_pending_count(uint64_t token)
{
    return ImplementationChooser<_CUDA_AWBARRIER_HAS_HW_MBARRIER>::awbarrier_token_pending_count(token);
}

template<bool Drop>
_CUDA_AWBARRIER_QUALIFIER
uint64_t awbarrier_arrive_drop_no_complete(uint64_t* barrier, uint32_t arrive_count)
{
    return ImplementationChooser<_CUDA_AWBARRIER_HAS_HW_MBARRIER>::awbarrier_arrive_drop_no_complete<Drop>(barrier, arrive_count);
}

template<bool Drop>
_CUDA_AWBARRIER_QUALIFIER
uint64_t awbarrier_arrive_drop(uint64_t* barrier)
{
    return ImplementationChooser<_CUDA_AWBARRIER_HAS_HW_MBARRIER>::awbarrier_arrive_drop<Drop>(barrier);
}

_CUDA_AWBARRIER_QUALIFIER
bool awbarrier_test_wait(uint64_t* barrier, uint64_t token)
{
    return ImplementationChooser<_CUDA_AWBARRIER_HAS_HW_MBARRIER>::awbarrier_test_wait(barrier, token);
}

_CUDA_AWBARRIER_END_INTERNAL_NAMESPACE

# endif /* !_CUDA_AWBARRIER_ARCH_700_OR_LATER */

#endif /* !_CUDA_AWBARRIER_HELPERS_H_ */
