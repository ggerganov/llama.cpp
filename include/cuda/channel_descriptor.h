/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CHANNEL_DESCRIPTOR_H__)
#define __CHANNEL_DESCRIPTOR_H__

#if defined(__cplusplus)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * \addtogroup CUDART_HIGHLEVEL
 *
 * @{
 */

/**
 * \brief \hl Returns a channel descriptor using the specified format
 *
 * Returns a channel descriptor with format \p f and number of bits of each
 * component \p x, \p y, \p z, and \p w.  The ::cudaChannelFormatDesc is
 * defined as:
 * \code
  struct cudaChannelFormatDesc {
    int x, y, z, w;
    enum cudaChannelFormatKind f;
  };
 * \endcode
 *
 * where ::cudaChannelFormatKind is one of ::cudaChannelFormatKindSigned,
 * ::cudaChannelFormatKindUnsigned, cudaChannelFormatKindFloat,
 * ::cudaChannelFormatKindSignedNormalized8X1, ::cudaChannelFormatKindSignedNormalized8X2,
 * ::cudaChannelFormatKindSignedNormalized8X4,
 * ::cudaChannelFormatKindUnsignedNormalized8X1, ::cudaChannelFormatKindUnsignedNormalized8X2,
 * ::cudaChannelFormatKindUnsignedNormalized8X4,
 * ::cudaChannelFormatKindSignedNormalized16X1, ::cudaChannelFormatKindSignedNormalized16X2,
 * ::cudaChannelFormatKindSignedNormalized16X4,
 * ::cudaChannelFormatKindUnsignedNormalized16X1, ::cudaChannelFormatKindUnsignedNormalized16X2,
 * ::cudaChannelFormatKindUnsignedNormalized16X4
 * or ::cudaChannelFormatKindNV12.
 *
 * The format is specified by the template specialization.
 *
 * The template function specializes for the following scalar types:
 * char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, and float.
 * The template function specializes for the following vector types:
 * char{1|2|4}, uchar{1|2|4}, short{1|2|4}, ushort{1|2|4}, int{1|2|4}, uint{1|2|4}, long{1|2|4}, ulong{1|2|4}, float{1|2|4}.
 * The template function specializes for following cudaChannelFormatKind enum values:
 * ::cudaChannelFormatKind{Uns|S}ignedNormalized{8|16}X{1|2|4}, and ::cudaChannelFormatKindNV12.
 *
 * Invoking the function on a type without a specialization defaults to creating a channel format of kind ::cudaChannelFormatKindNone
 *
 * \return
 * Channel descriptor with format \p f
 *
 * \sa \ref ::cudaCreateChannelDesc(int,int,int,int,cudaChannelFormatKind) "cudaCreateChannelDesc (Low level)",
 * ::cudaGetChannelDesc, ::cudaGetTextureReference,
 * \ref ::cudaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t) "cudaBindTexture (High level)",
 * \ref ::cudaBindTexture(size_t*, const struct texture< T, dim, readMode>&, const void*, size_t) "cudaBindTexture (High level, inherited channel descriptor)",
 * \ref ::cudaBindTexture2D(size_t*, const struct texture< T, dim, readMode>&, const void*, const struct cudaChannelFormatDesc&, size_t, size_t, size_t) "cudaBindTexture2D (High level)",
 * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t, const struct cudaChannelFormatDesc&) "cudaBindTextureToArray (High level)",
 * \ref ::cudaBindTextureToArray(const struct texture< T, dim, readMode>&, cudaArray_const_t) "cudaBindTextureToArray (High level, inherited channel descriptor)",
 * \ref ::cudaUnbindTexture(const struct texture< T, dim, readMode>&) "cudaUnbindTexture (High level)",
 * \ref ::cudaGetTextureAlignmentOffset(size_t*, const struct texture< T, dim, readMode>&) "cudaGetTextureAlignmentOffset (High level)"
 */
template<class T> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
  return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

static __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDescHalf(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDescHalf1(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDescHalf2(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

static __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDescHalf4(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char>(void)
{
  int e = (int)sizeof(char) * 8;

#if defined(_CHAR_UNSIGNED) || defined(__CHAR_UNSIGNED__)
  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
#else /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
#endif /* _CHAR_UNSIGNED || __CHAR_UNSIGNED__ */
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<signed char>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned char>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char1>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar1>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char2>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar2>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<char4>(void)
{
  int e = (int)sizeof(signed char) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uchar4>(void)
{
  int e = (int)sizeof(unsigned char) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned short>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short1>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort1>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short2>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort2>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<short4>(void)
{
  int e = (int)sizeof(short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ushort4>(void)
{
  int e = (int)sizeof(unsigned short) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned int>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int1>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint1>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int2>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint2>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<int4>(void)
{
  int e = (int)sizeof(int) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<uint4>(void)
{
  int e = (int)sizeof(unsigned int) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

#if !defined(__LP64__)

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long>(void)
{
  int e = (int)sizeof(long) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<unsigned long>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long1>(void)
{
  int e = (int)sizeof(long) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong1>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long2>(void)
{
  int e = (int)sizeof(long) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong2>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<long4>(void)
{
  int e = (int)sizeof(long) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<ulong4>(void)
{
  int e = (int)sizeof(unsigned long) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned);
}

#endif /* !__LP64__ */

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float1>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float2>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<float4>(void)
{
  int e = (int)sizeof(float) * 8;

  return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

static __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDescNV12(void)
{
    int e = (int)sizeof(char) * 8;

    return cudaCreateChannelDesc(e, e, e, 0, cudaChannelFormatKindNV12);
}

template<cudaChannelFormatKind> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc(void)
{
    return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
}

/* Signed 8-bit normalized integer formats */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X1>(void)
{
    return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedNormalized8X1);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X2>(void)
{
    return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedNormalized8X2);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X4>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSignedNormalized8X4);
}

/* Unsigned 8-bit normalized integer formats */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X1>(void)
{
    return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X2>(void)
{
    return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedNormalized8X2);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4);
}

/* Signed 16-bit normalized integer formats */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X1>(void)
{
    return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSignedNormalized16X1);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X2>(void)
{
    return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSignedNormalized16X2);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X4>(void)
{
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSignedNormalized16X4);
}

/* Unsigned 16-bit normalized integer formats */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X1>(void)
{
    return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized16X1);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X2>(void)
{
    return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsignedNormalized16X2);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X4>(void)
{
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsignedNormalized16X4);
}

/* NV12 format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindNV12>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindNV12);
}

/* BC1 format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1);
}

/* BC1sRGB format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1SRGB>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed1SRGB);
}

/* BC2 format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2);
}

/* BC2sRGB format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2SRGB>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed2SRGB);
}

/* BC3 format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3);
}

/* BC3sRGB format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3SRGB>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed3SRGB);
}

/* BC4 unsigned format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>(void)
{
    return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed4);
}

/* BC4 signed format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed4>(void)
{
    return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSignedBlockCompressed4);
}

/* BC5 unsigned format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>(void)
{
    return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsignedBlockCompressed5);
}

/* BC5 signed format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed5>(void)
{
    return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSignedBlockCompressed5);
}

/* BC6H unsigned format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed6H>(void)
{
    return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindUnsignedBlockCompressed6H);
}

/* BC6H signed format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed6H>(void)
{
    return cudaCreateChannelDesc(16, 16, 16, 0, cudaChannelFormatKindSignedBlockCompressed6H);
}

/* BC7 format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7);
}

/* BC7sRGB format */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7SRGB>(void)
{
    return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedBlockCompressed7SRGB);
}

#endif /* __cplusplus */

/** @} */
/** @} */ /* END CUDART_TEXTURE_HL */

#endif /* !__CHANNEL_DESCRIPTOR_H__ */
