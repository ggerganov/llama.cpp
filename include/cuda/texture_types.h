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

#if !defined(__TEXTURE_TYPES_H__)
#define __TEXTURE_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "driver_types.h"

/**
 * \addtogroup CUDART_TYPES
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#define cudaTextureType1D              0x01
#define cudaTextureType2D              0x02
#define cudaTextureType3D              0x03
#define cudaTextureTypeCubemap         0x0C
#define cudaTextureType1DLayered       0xF1
#define cudaTextureType2DLayered       0xF2
#define cudaTextureTypeCubemapLayered  0xFC

/**
 * CUDA texture address modes
 */
enum __device_builtin__ cudaTextureAddressMode
{
    cudaAddressModeWrap   = 0,    /**< Wrapping address mode */
    cudaAddressModeClamp  = 1,    /**< Clamp to edge address mode */
    cudaAddressModeMirror = 2,    /**< Mirror address mode */
    cudaAddressModeBorder = 3     /**< Border address mode */
};

/**
 * CUDA texture filter modes
 */
enum __device_builtin__ cudaTextureFilterMode
{
    cudaFilterModePoint  = 0,     /**< Point filter mode */
    cudaFilterModeLinear = 1      /**< Linear filter mode */
};

/**
 * CUDA texture read modes
 */
enum __device_builtin__ cudaTextureReadMode
{
    cudaReadModeElementType     = 0,  /**< Read texture as specified element type */
    cudaReadModeNormalizedFloat = 1   /**< Read texture as normalized float */
};

/**
 * CUDA texture reference
 */
struct __device_builtin__ textureReference
{
    /**
     * Indicates whether texture reads are normalized or not
     */
    int                          normalized;
    /**
     * Texture filter mode
     */
    enum cudaTextureFilterMode   filterMode;
    /**
     * Texture address mode for up to 3 dimensions
     */
    enum cudaTextureAddressMode  addressMode[3];
    /**
     * Channel descriptor for the texture reference
     */
    struct cudaChannelFormatDesc channelDesc;
    /**
     * Perform sRGB->linear conversion during texture read
     */
    int                          sRGB;
    /**
     * Limit to the anisotropy ratio
     */
    unsigned int                 maxAnisotropy;
    /**
     * Mipmap filter mode
     */
    enum cudaTextureFilterMode   mipmapFilterMode;
    /**
     * Offset applied to the supplied mipmap level
     */
    float                        mipmapLevelBias;
    /**
     * Lower end of the mipmap level range to clamp access to
     */
    float                        minMipmapLevelClamp;
    /**
     * Upper end of the mipmap level range to clamp access to
     */
    float                        maxMipmapLevelClamp;
    /**
     * Disable any trilinear filtering optimizations.
     */
    int                          disableTrilinearOptimization;
    int                          __cudaReserved[14];
};

/**
 * CUDA texture descriptor
 */
struct __device_builtin__ cudaTextureDesc
{
    /**
     * Texture address mode for up to 3 dimensions
     */
    enum cudaTextureAddressMode addressMode[3];
    /**
     * Texture filter mode
     */
    enum cudaTextureFilterMode  filterMode;
    /**
     * Texture read mode
     */
    enum cudaTextureReadMode    readMode;
    /**
     * Perform sRGB->linear conversion during texture read
     */
    int                         sRGB;
    /**
     * Texture Border Color
     */
    float                       borderColor[4];
    /**
     * Indicates whether texture reads are normalized or not
     */
    int                         normalizedCoords;
    /**
     * Limit to the anisotropy ratio
     */
    unsigned int                maxAnisotropy;
    /**
     * Mipmap filter mode
     */
    enum cudaTextureFilterMode  mipmapFilterMode;
    /**
     * Offset applied to the supplied mipmap level
     */
    float                       mipmapLevelBias;
    /**
     * Lower end of the mipmap level range to clamp access to
     */
    float                       minMipmapLevelClamp;
    /**
     * Upper end of the mipmap level range to clamp access to
     */
    float                       maxMipmapLevelClamp;
    /**
     * Disable any trilinear filtering optimizations.
     */
    int                         disableTrilinearOptimization;
    /**
     * Enable seamless cube map filtering.
     */
    int                         seamlessCubemap;
};

/**
 * An opaque value that represents a CUDA texture object
 */
typedef __device_builtin__ unsigned long long cudaTextureObject_t;

/** @} */
/** @} */ /* END CUDART_TYPES */

#endif /* !__TEXTURE_TYPES_H__ */
