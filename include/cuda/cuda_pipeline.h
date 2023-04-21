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

#ifndef _CUDA_PIPELINE_H_
# define _CUDA_PIPELINE_H_

# include "cuda_pipeline_primitives.h"

# if !defined(_CUDA_PIPELINE_CPLUSPLUS_11_OR_LATER)
#  error This file requires compiler support for the ISO C++ 2011 standard. This support must be enabled with the \
         -std=c++11 compiler option.
# endif

# if defined(_CUDA_PIPELINE_ARCH_700_OR_LATER)
#  include "cuda_awbarrier.h"
# endif

// Integration with libcu++'s cuda::barrier<cuda::thread_scope_block>.

# if defined(_CUDA_PIPELINE_ARCH_700_OR_LATER)
#  if defined(_LIBCUDACXX_CUDA_ABI_VERSION)
#   define _LIBCUDACXX_PIPELINE_ASSUMED_ABI_VERSION _LIBCUDACXX_CUDA_ABI_VERSION
#  else
#   define _LIBCUDACXX_PIPELINE_ASSUMED_ABI_VERSION 4
#  endif

#  define _LIBCUDACXX_PIPELINE_CONCAT(X, Y) X ## Y
#  define _LIBCUDACXX_PIPELINE_CONCAT2(X, Y) _LIBCUDACXX_PIPELINE_CONCAT(X, Y)
#  define _LIBCUDACXX_PIPELINE_INLINE_NAMESPACE _LIBCUDACXX_PIPELINE_CONCAT2(__, _LIBCUDACXX_PIPELINE_ASSUMED_ABI_VERSION)

namespace cuda { inline namespace _LIBCUDACXX_PIPELINE_INLINE_NAMESPACE {
    struct __block_scope_barrier_base;
}}

# endif

_CUDA_PIPELINE_BEGIN_NAMESPACE

template<size_t N, typename T>
_CUDA_PIPELINE_QUALIFIER
auto segment(T* ptr) -> T(*)[N];

class pipeline {
public:
    pipeline(const pipeline&) = delete;
    pipeline(pipeline&&) = delete;
    pipeline& operator=(const pipeline&) = delete;
    pipeline& operator=(pipeline&&) = delete;

    _CUDA_PIPELINE_QUALIFIER pipeline();
    _CUDA_PIPELINE_QUALIFIER size_t commit();
    _CUDA_PIPELINE_QUALIFIER void commit_and_wait();
    _CUDA_PIPELINE_QUALIFIER void wait(size_t batch);
    template<unsigned N>
    _CUDA_PIPELINE_QUALIFIER void wait_prior();

# if defined(_CUDA_PIPELINE_ARCH_700_OR_LATER)
    _CUDA_PIPELINE_QUALIFIER void arrive_on(awbarrier& barrier);
    _CUDA_PIPELINE_QUALIFIER void arrive_on(cuda::__block_scope_barrier_base& barrier);
# endif

private:
    size_t current_batch;
};

template<class T>
_CUDA_PIPELINE_QUALIFIER
void memcpy_async(T& dst, const T& src, pipeline& pipe);

template<class T, size_t DstN, size_t SrcN>
_CUDA_PIPELINE_QUALIFIER
void memcpy_async(T(*dst)[DstN], const T(*src)[SrcN], pipeline& pipe);

template<size_t N, typename T>
_CUDA_PIPELINE_QUALIFIER
auto segment(T* ptr) -> T(*)[N]
{
    return (T(*)[N])ptr;
}

_CUDA_PIPELINE_QUALIFIER
pipeline::pipeline()
    : current_batch(0)
{
}

_CUDA_PIPELINE_QUALIFIER
size_t pipeline::commit()
{
    _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_commit();
    return this->current_batch++;
}

_CUDA_PIPELINE_QUALIFIER
void pipeline::commit_and_wait()
{
    (void)pipeline::commit();
    pipeline::wait_prior<0>();
}

_CUDA_PIPELINE_QUALIFIER
void pipeline::wait(size_t batch)
{
    const size_t prior = this->current_batch > batch ? this->current_batch - batch : 0;

    switch (prior) {
    case  0 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<0>(); break;
    case  1 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<1>(); break;
    case  2 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<2>(); break;
    case  3 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<3>(); break;
    case  4 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<4>(); break;
    case  5 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<5>(); break;
    case  6 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<6>(); break;
    case  7 : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<7>(); break;
    default : _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<8>(); break;
    }
}

template<unsigned N>
_CUDA_PIPELINE_QUALIFIER
void pipeline::wait_prior()
{
    _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_wait_prior<N>();
}

# if defined(_CUDA_PIPELINE_ARCH_700_OR_LATER)
_CUDA_PIPELINE_QUALIFIER
void pipeline::arrive_on(awbarrier& barrier)
{
    _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_arrive_on(&barrier.barrier);
}

_CUDA_PIPELINE_QUALIFIER
void pipeline::arrive_on(cuda::__block_scope_barrier_base & barrier)
{
    _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_arrive_on(reinterpret_cast<uint64_t *>(&barrier));
}
# endif

template<class T>
_CUDA_PIPELINE_QUALIFIER
void memcpy_async(T& dst, const T& src, pipeline& pipe)
{
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(&src) & (alignof(T) - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(&dst) & (alignof(T) - 1)));

    if (__is_trivially_copyable(T)) {
        _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_copy_relaxed<sizeof(T), alignof(T)>(
                reinterpret_cast<void*>(&dst), reinterpret_cast<const void*>(&src));
    } else {
        dst = src;
    }
}

template<class T, size_t DstN, size_t SrcN>
_CUDA_PIPELINE_QUALIFIER
void memcpy_async(T(*dst)[DstN], const T(*src)[SrcN], pipeline& pipe)
{
    constexpr size_t dst_size = sizeof(*dst);
    constexpr size_t src_size = sizeof(*src);
    static_assert(dst_size == 4 || dst_size == 8 || dst_size == 16, "Unsupported copy size.");
    static_assert(src_size <= dst_size, "Source size must be less than or equal to destination size.");
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(src) & (dst_size - 1)));
    _CUDA_PIPELINE_ASSERT(!(reinterpret_cast<uintptr_t>(dst) & (dst_size - 1)));

    if (__is_trivially_copyable(T)) {
        _CUDA_PIPELINE_INTERNAL_NAMESPACE::pipeline_copy_strict<sizeof(*dst), sizeof(*src)>(
                reinterpret_cast<void*>(*dst), reinterpret_cast<const void*>(*src));
    } else {
        for (size_t i = 0; i < DstN; ++i) {
            (*dst)[i] = (i < SrcN) ? (*src)[i] : T();
        }
    }
}

_CUDA_PIPELINE_END_NAMESPACE

#endif /* !_CUDA_PIPELINE_H_ */
