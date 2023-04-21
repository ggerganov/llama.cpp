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

#ifndef _CUDA_PIPELINE_HELPERS_H_
# define _CUDA_PIPELINE_HELPERS_H_

# define _CUDA_PIPELINE_NAMESPACE       nvcuda::experimental
# define _CUDA_PIPELINE_BEGIN_NAMESPACE namespace nvcuda { namespace experimental {
# define _CUDA_PIPELINE_END_NAMESPACE   } }

# define _CUDA_PIPELINE_INTERNAL_NAMESPACE       _CUDA_PIPELINE_NAMESPACE::__pipeline_internal
# define _CUDA_PIPELINE_BEGIN_INTERNAL_NAMESPACE _CUDA_PIPELINE_BEGIN_NAMESPACE namespace __pipeline_internal {
# define _CUDA_PIPELINE_END_INTERNAL_NAMESPACE   } _CUDA_PIPELINE_END_NAMESPACE

# if !defined(_CUDA_PIPELINE_QUALIFIER)
#  define _CUDA_PIPELINE_QUALIFIER inline __device__
# endif
# if !defined(_CUDA_PIPELINE_STATIC_QUALIFIER)
#  define _CUDA_PIPELINE_STATIC_QUALIFIER static inline __device__
# endif

# if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
#  define _CUDA_PIPELINE_ARCH_700_OR_LATER
# endif

# if (__CUDA_ARCH__ >= 800)
#  define _CUDA_PIPELINE_HAS_ASYNC_COPY 1
# else
#  define _CUDA_PIPELINE_HAS_ASYNC_COPY 0
# endif

# if !defined(_CUDA_PIPELINE_MAX_STAGES)
#  define _CUDA_PIPELINE_MAX_STAGES 8
# endif

# if defined(__cplusplus) && ((__cplusplus >= 201103L) || (defined(_MSC_VER) && (_MSC_VER >= 1900)))
#  define _CUDA_PIPELINE_CPLUSPLUS_11_OR_LATER
# endif

# if !defined(_CUDA_PIPELINE_DEBUG)
#  if defined(__CUDACC_DEBUG__)
#   define _CUDA_PIPELINE_DEBUG 1
#  else
#   define _CUDA_PIPELINE_DEBUG 0
#  endif
# endif

# if defined(_CUDA_PIPELINE_DEBUG) && (_CUDA_PIPELINE_DEBUG == 1) && !defined(NDEBUG)
#  if !defined(__CUDACC_RTC__)
#   include <cassert>
#  endif
#  define _CUDA_PIPELINE_ASSERT(x) assert((x));
#  define _CUDA_PIPELINE_ABORT() assert(0);
# else
#  define _CUDA_PIPELINE_ASSERT(x)
#  define _CUDA_PIPELINE_ABORT() __trap();
# endif

# if defined(_CUDA_PIPELINE_CPLUSPLUS_11_OR_LATER)
#  define _CUDA_PIPELINE_STATIC_ASSERT(c, m) static_assert(c, m)
# else
#  define _CUDA_PIPELINE_STATIC_ASSERT(c, m)
# endif

# if (defined(_MSC_VER) && !defined(_WIN64)) || defined(__arm__)
#  define _CUDA_PIPELINE_ASM_PTR_CONSTRAINT "r"
# else
#  define _CUDA_PIPELINE_ASM_PTR_CONSTRAINT "l"
# endif

# if defined(__CUDACC_RTC__)
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef uint64_t           uintptr_t;
# else
#  include <stdint.h>
# endif

_CUDA_PIPELINE_BEGIN_INTERNAL_NAMESPACE

_CUDA_PIPELINE_STATIC_ASSERT(sizeof(short) ==  2, "Size mismatch for type 'short'");
_CUDA_PIPELINE_STATIC_ASSERT(sizeof(int)   ==  4, "Size mismatch for type 'int'");
_CUDA_PIPELINE_STATIC_ASSERT(sizeof(int2)  ==  8, "Size mismatch for type 'int2'");
_CUDA_PIPELINE_STATIC_ASSERT(sizeof(int4)  == 16, "Size mismatch for type 'int4'");

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *);

template<size_t CopySize, size_t SourceSize>
_CUDA_PIPELINE_QUALIFIER
void pipeline_memcpy_sync(void* __restrict__ dst, const void* __restrict__ src)
{
    _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
    _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size");
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));

    char* const d = reinterpret_cast<char*>(dst);
    const char* const s = reinterpret_cast<const char*>(src);

    size_t copy_step_size;
    if (SourceSize == 0) {
        copy_step_size = CopySize;
    } else if (SourceSize == 2 || SourceSize == 4 || SourceSize == 8 || SourceSize == 16) {
        copy_step_size = SourceSize;
    } else {
        copy_step_size = 1;
    }

    for (size_t i = 0; i < CopySize; i += copy_step_size) {
        const bool copy_source = SourceSize && (i < SourceSize);

        switch (copy_step_size) {
        case 1:
            d[i] = copy_source ? s[i] : char();
            break;
        case 2:
            *reinterpret_cast<short*>(d + i) = copy_source ? *reinterpret_cast<const short*>(s + i) : short();
            break;
        case 4:
            *reinterpret_cast<int*>(d + i) = copy_source ? *reinterpret_cast<const int*>(s + i) : int();
            break;
        case 8:
            *reinterpret_cast<int2*>(d + i) = copy_source ? *reinterpret_cast<const int2*>(s + i) : int2();
            break;
        case 16:
            *reinterpret_cast<int4*>(d + i) = copy_source ? *reinterpret_cast<const int4*>(s + i) : int4();
            break;
        }
    }
}

template<bool UseHwAsyncCopy>
struct ImplementationChooser;

template<>
struct ImplementationChooser<true> {
    template<size_t CopySize, size_t SourceSize>
    struct CpAsyncChooser {
        _CUDA_PIPELINE_STATIC_QUALIFIER
        void cp_async(void* __restrict__ dst, const void* __restrict__ src)
        {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], %2, %3;"
                :
                : "r"(__nvvm_get_smem_pointer(dst)), _CUDA_PIPELINE_ASM_PTR_CONSTRAINT(src), "n"(CopySize),
                  "n"(SourceSize)
                : "memory");
        }
    };

    template<size_t SourceSize>
    struct CpAsyncChooser<16, SourceSize> {
        _CUDA_PIPELINE_STATIC_QUALIFIER
        void cp_async(void* __restrict__ dst, const void* __restrict__ src)
        {
            asm volatile ("cp.async.cg.shared.global [%0], [%1], %2, %3;"
                :
                : "r"(__nvvm_get_smem_pointer(dst)), _CUDA_PIPELINE_ASM_PTR_CONSTRAINT(src), "n"(16), "n"(SourceSize)
                : "memory");
        }
    };

    template<size_t CopySize, size_t SourceSize>
    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_memcpy_async(void* __restrict__ dst, const void* __restrict__ src)
    {
        _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
        _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size");
        _CUDA_PIPELINE_ASSERT(__isShared(dst));
        _CUDA_PIPELINE_ASSERT(__isGlobal(src));
        _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));
        _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));

        CpAsyncChooser<CopySize, SourceSize>::cp_async(dst, src);
    }

    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_commit()
    {
        asm volatile ("cp.async.commit_group;");
    }

    template<unsigned N>
    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_wait_prior()
    {
        asm volatile ("cp.async.wait_group %0;"
            :
            : "n"(N < _CUDA_PIPELINE_MAX_STAGES ? N : _CUDA_PIPELINE_MAX_STAGES));
    }

    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_arrive_on(uint64_t* barrier)
    {
        _CUDA_PIPELINE_ASSERT(__isShared(barrier));

        asm volatile ("cp.async.mbarrier.arrive.shared.b64 [%0];"
            :
            : "r"(__nvvm_get_smem_pointer(barrier)));
    }
};

template<>
struct ImplementationChooser<false> {
    template<size_t CopySize, size_t SourceSize>
    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_memcpy_async(void* __restrict__ dst, const void* __restrict__ src)
    {
        _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
        _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size");
        _CUDA_PIPELINE_ASSERT(__isShared(dst));
        _CUDA_PIPELINE_ASSERT(__isGlobal(src));
        _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));
        _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));

        pipeline_memcpy_sync<CopySize, SourceSize>(dst, src);
    }

    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_commit()
    {
    }

    template<unsigned N>
    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_wait_prior()
    {
    }

    _CUDA_PIPELINE_STATIC_QUALIFIER
    void pipeline_arrive_on(uint64_t* barrier)
    {
    }
};

template<size_t CopySize, size_t SourceSize>
_CUDA_PIPELINE_QUALIFIER
void pipeline_memcpy_async(void* __restrict__ dst, const void* __restrict__ src)
{
    _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
    _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size");
    _CUDA_PIPELINE_ASSERT(__isShared(dst));
    _CUDA_PIPELINE_ASSERT(__isGlobal(src));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));

    ImplementationChooser<_CUDA_PIPELINE_HAS_ASYNC_COPY>::pipeline_memcpy_async<CopySize, SourceSize>(dst, src);
}

_CUDA_PIPELINE_QUALIFIER
void pipeline_commit()
{
    ImplementationChooser<_CUDA_PIPELINE_HAS_ASYNC_COPY>::pipeline_commit();
}

template<unsigned N>
_CUDA_PIPELINE_QUALIFIER
void pipeline_wait_prior()
{
    ImplementationChooser<_CUDA_PIPELINE_HAS_ASYNC_COPY>::pipeline_wait_prior<N>();
}

_CUDA_PIPELINE_QUALIFIER
void pipeline_arrive_on(uint64_t* barrier)
{
    ImplementationChooser<_CUDA_PIPELINE_HAS_ASYNC_COPY>::pipeline_arrive_on(barrier);
}

template<size_t CopySize, size_t SourceSize>
_CUDA_PIPELINE_QUALIFIER
void pipeline_copy_strict(void* __restrict__ dst, const void* __restrict__ src)
{
    _CUDA_PIPELINE_STATIC_ASSERT(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
    _CUDA_PIPELINE_STATIC_ASSERT(SourceSize <= CopySize, "Source size must be less than or equal to copy size.");
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (CopySize - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (CopySize - 1)));

    if (__isGlobal(src) && __isShared(dst)) {
        pipeline_memcpy_async<CopySize, SourceSize>(dst, src);
    } else {
        pipeline_memcpy_sync<CopySize, SourceSize>(dst, src);
    }
}

template<size_t CopySize, size_t Align>
_CUDA_PIPELINE_QUALIFIER
void pipeline_copy_relaxed(void* __restrict__ dst, const void* __restrict__ src)
{
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (Align - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (Align - 1)));

    const char* s = reinterpret_cast<const char*>(src);
    char* d = reinterpret_cast<char*>(dst);
    size_t remaining = CopySize;

    while (remaining) {
        if ((Align >= 16) && (remaining >= 16)) {
            pipeline_copy_strict<16, 16>(dst, src);
            d += 16;
            s += 16;
            remaining -= 16;
        } else if ((Align >= 8) && (remaining >= 8)) {
            pipeline_copy_strict<8, 8>(dst, src);
            d += 8;
            s += 8;
            remaining -= 8;
        } else if ((Align >= 4) && (remaining >= 4)) {
            pipeline_copy_strict<4, 4>(dst, src);
            d += 4;
            s += 4;
            remaining -= 4;
        } else if ((Align >= 2) && (remaining >= 2)) {
            *reinterpret_cast<short*>(d) = *reinterpret_cast<const short*>(s);
            d += 2;
            s += 2;
            remaining -= 2;
        } else {
            *d = *s;
            d += 1;
            s += 1;
            remaining -= 1;
        }
    }
}

_CUDA_PIPELINE_END_INTERNAL_NAMESPACE

#endif /* !_CUDA_PIPELINE_HELPERS_H_ */
