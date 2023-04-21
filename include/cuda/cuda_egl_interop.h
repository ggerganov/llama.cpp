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

#if !defined(__CUDA_EGL_INTEROP_H__)
#define __CUDA_EGL_INTEROP_H__

#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "cudart_platform.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \addtogroup CUDART_TYPES
 * @{
 */

 /**
 * Maximum number of planes per frame
 */
#define CUDA_EGL_MAX_PLANES 3

/**
 * CUDA EglFrame type - array or pointer
 */
typedef enum cudaEglFrameType_enum
{
    cudaEglFrameTypeArray = 0,  /**< Frame type CUDA array */
    cudaEglFrameTypePitch = 1,  /**< Frame type CUDA pointer */
} cudaEglFrameType;

/**
 * Resource location flags- sysmem or vidmem
 *
 * For CUDA context on iGPU, since video and system memory are equivalent -
 * these flags will not have an effect on the execution.
 *
 * For CUDA context on dGPU, applications can use the flag ::cudaEglResourceLocationFlags
 * to give a hint about the desired location.
 *
 * ::cudaEglResourceLocationSysmem - the frame data is made resident on the system memory
 * to be accessed by CUDA.
 *
 * ::cudaEglResourceLocationVidmem - the frame data is made resident on the dedicated
 * video memory to be accessed by CUDA.
 *
 * There may be an additional latency due to new allocation and data migration,
 * if the frame is produced on a different memory.
 */
typedef enum cudaEglResourceLocationFlags_enum {
    cudaEglResourceLocationSysmem   = 0x00,       /**< Resource location sysmem */
    cudaEglResourceLocationVidmem   = 0x01,       /**< Resource location vidmem */
} cudaEglResourceLocationFlags;

/**
 * CUDA EGL Color Format - The different planar and multiplanar formats currently supported for CUDA_EGL interops.
 */
typedef enum cudaEglColorFormat_enum {
    cudaEglColorFormatYUV420Planar            = 0,  /**< Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYUV420SemiPlanar        = 1,  /**< Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar. */
    cudaEglColorFormatYUV422Planar            = 2,  /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYUV422SemiPlanar        = 3,  /**< Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar. */
    cudaEglColorFormatARGB                    = 6,  /**< R/G/B/A four channels in one surface with BGRA byte ordering. */
    cudaEglColorFormatRGBA                    = 7,  /**< R/G/B/A four channels in one surface with ABGR byte ordering. */
    cudaEglColorFormatL                       = 8,  /**< single luminance channel in one surface. */
    cudaEglColorFormatR                       = 9,  /**< single color channel in one surface. */
    cudaEglColorFormatYUV444Planar            = 10, /**< Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYUV444SemiPlanar        = 11, /**< Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar. */
    cudaEglColorFormatYUYV422                 = 12, /**< Y, U, V in one surface, interleaved as UYVY in one channel. */
    cudaEglColorFormatUYVY422                 = 13, /**< Y, U, V in one surface, interleaved as YUYV in one channel. */
    cudaEglColorFormatABGR                    = 14, /**< R/G/B/A four channels in one surface with RGBA byte ordering. */
    cudaEglColorFormatBGRA                    = 15, /**< R/G/B/A four channels in one surface with ARGB byte ordering. */
    cudaEglColorFormatA                       = 16, /**< Alpha color format - one channel in one surface. */
    cudaEglColorFormatRG                      = 17, /**< R/G color format - two channels in one surface with GR byte ordering */
    cudaEglColorFormatAYUV                    = 18, /**< Y, U, V, A four channels in one surface, interleaved as VUYA. */
    cudaEglColorFormatYVU444SemiPlanar        = 19, /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYVU422SemiPlanar        = 20, /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYVU420SemiPlanar        = 21, /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY10V10U10_444SemiPlanar = 22, /**< Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatY10V10U10_420SemiPlanar = 23, /**< Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY12V12U12_444SemiPlanar = 24, /**< Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatY12V12U12_420SemiPlanar = 25, /**< Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatVYUY_ER                 = 26, /**< Extended Range Y, U, V in one surface, interleaved as YVYU in one channel. */
    cudaEglColorFormatUYVY_ER                 = 27, /**< Extended Range Y, U, V in one surface, interleaved as YUYV in one channel. */
    cudaEglColorFormatYUYV_ER                 = 28, /**< Extended Range Y, U, V in one surface, interleaved as UYVY in one channel. */
    cudaEglColorFormatYVYU_ER                 = 29, /**< Extended Range Y, U, V in one surface, interleaved as VYUY in one channel. */
    cudaEglColorFormatYUVA_ER                 = 31, /**< Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY. */
    cudaEglColorFormatAYUV_ER                 = 32, /**< Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA. */
    cudaEglColorFormatYUV444Planar_ER         = 33, /**< Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYUV422Planar_ER         = 34, /**< Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYUV420Planar_ER         = 35, /**< Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYUV444SemiPlanar_ER     = 36, /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYUV422SemiPlanar_ER     = 37, /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYUV420SemiPlanar_ER     = 38, /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYVU444Planar_ER         = 39, /**< Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYVU422Planar_ER         = 40, /**< Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYVU420Planar_ER         = 41, /**< Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYVU444SemiPlanar_ER     = 42, /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYVU422SemiPlanar_ER     = 43, /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYVU420SemiPlanar_ER     = 44, /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatBayerRGGB               = 45, /**< Bayer format - one channel in one surface with interleaved RGGB ordering. */
    cudaEglColorFormatBayerBGGR               = 46, /**< Bayer format - one channel in one surface with interleaved BGGR ordering. */
    cudaEglColorFormatBayerGRBG               = 47, /**< Bayer format - one channel in one surface with interleaved GRBG ordering. */
    cudaEglColorFormatBayerGBRG               = 48, /**< Bayer format - one channel in one surface with interleaved GBRG ordering. */
    cudaEglColorFormatBayer10RGGB             = 49, /**< Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    cudaEglColorFormatBayer10BGGR             = 50, /**< Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    cudaEglColorFormatBayer10GRBG             = 51, /**< Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    cudaEglColorFormatBayer10GBRG             = 52, /**< Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    cudaEglColorFormatBayer12RGGB             = 53, /**< Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12BGGR             = 54, /**< Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12GRBG             = 55, /**< Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12GBRG             = 56, /**< Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer14RGGB             = 57, /**< Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    cudaEglColorFormatBayer14BGGR             = 58, /**< Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    cudaEglColorFormatBayer14GRBG             = 59, /**< Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    cudaEglColorFormatBayer14GBRG             = 60, /**< Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    cudaEglColorFormatBayer20RGGB             = 61, /**< Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    cudaEglColorFormatBayer20BGGR             = 62, /**< Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    cudaEglColorFormatBayer20GRBG             = 63, /**< Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    cudaEglColorFormatBayer20GBRG             = 64, /**< Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    cudaEglColorFormatYVU444Planar            = 65, /**< Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatYVU422Planar            = 66, /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height. */
    cudaEglColorFormatYVU420Planar            = 67, /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatBayerIspRGGB            = 68, /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype. */
    cudaEglColorFormatBayerIspBGGR            = 69, /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype. */
    cudaEglColorFormatBayerIspGRBG            = 70, /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype. */
    cudaEglColorFormatBayerIspGBRG            = 71, /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype. */
    cudaEglColorFormatBayerBCCR               = 72, /**< Bayer format - one channel in one surface with interleaved BCCR ordering. */
    cudaEglColorFormatBayerRCCB               = 73, /**< Bayer format - one channel in one surface with interleaved RCCB ordering. */
    cudaEglColorFormatBayerCRBC               = 74, /**< Bayer format - one channel in one surface with interleaved CRBC ordering. */
    cudaEglColorFormatBayerCBRC               = 75, /**< Bayer format - one channel in one surface with interleaved CBRC ordering. */
    cudaEglColorFormatBayer10CCCC             = 76, /**< Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    cudaEglColorFormatBayer12BCCR             = 77, /**< Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12RCCB             = 78, /**< Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12CRBC             = 79, /**< Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12CBRC             = 80, /**< Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatBayer12CCCC             = 81, /**< Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    cudaEglColorFormatY                       = 82, /**< Color format for single Y plane. */
    cudaEglColorFormatYUV420SemiPlanar_2020   = 83, /**< Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYVU420SemiPlanar_2020   = 84, /**< Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYUV420Planar_2020       = 85, /**< Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYVU420Planar_2020       = 86, /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYUV420SemiPlanar_709    = 87, /**< Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYVU420SemiPlanar_709    = 88, /**< Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYUV420Planar_709        = 89, /**< Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatYVU420Planar_709        = 90, /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY10V10U10_420SemiPlanar_709  = 91, /**< Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY10V10U10_420SemiPlanar_2020 = 92, /**< Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY10V10U10_422SemiPlanar_2020 = 93, /**< Y10, V10U10  in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height =  Y height. */
    cudaEglColorFormatY10V10U10_422SemiPlanar      = 94, /**< Y10, V10U10  in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height =  Y height. */
    cudaEglColorFormatY10V10U10_422SemiPlanar_709  = 95, /**< Y10, V10U10  in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height =  Y height. */
    cudaEglColorFormatY_ER                         = 96, /**< Extended Range Color format for single Y plane. */
    cudaEglColorFormatY_709_ER                     = 97, /**< Extended Range Color format for single Y plane. */
    cudaEglColorFormatY10_ER                       = 98, /**< Extended Range Color format for single Y10 plane. */
    cudaEglColorFormatY10_709_ER                   = 99, /**< Extended Range Color format for single Y10 plane. */
    cudaEglColorFormatY12_ER                       = 100, /**< Extended Range Color format for single Y12 plane. */
    cudaEglColorFormatY12_709_ER                   = 101, /**< Extended Range Color format for single Y12 plane. */
    cudaEglColorFormatYUVA                         = 102, /**< Y, U, V, A four channels in one surface, interleaved as AVUY. */
    cudaEglColorFormatYVYU                         = 104, /**< Y, U, V in one surface, interleaved as YVYU in one channel. */
    cudaEglColorFormatVYUY                         = 105, /**< Y, U, V in one surface, interleaved as VYUY in one channel. */
    cudaEglColorFormatY10V10U10_420SemiPlanar_ER     = 106, /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER = 107, /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY10V10U10_444SemiPlanar_ER     = 108, /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */ 
    cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER = 109, /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatY12V12U12_420SemiPlanar_ER     = 110, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER = 111, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    cudaEglColorFormatY12V12U12_444SemiPlanar_ER     = 112, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */
    cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER = 113, /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */
} cudaEglColorFormat;

/**
 * CUDA EGL Plane Descriptor - structure defining each plane of a CUDA EGLFrame
 */
typedef struct cudaEglPlaneDesc_st {
    unsigned int width;                         /**< Width of plane */
    unsigned int height;                        /**< Height of plane */
    unsigned int depth;                         /**< Depth of plane */
    unsigned int pitch;                         /**< Pitch of plane */
    unsigned int numChannels;                   /**< Number of channels for the plane */
    struct cudaChannelFormatDesc channelDesc;   /**< Channel Format Descriptor */
    unsigned int reserved[4];                   /**< Reserved for future use */
} cudaEglPlaneDesc;

/**
 * CUDA EGLFrame Descriptor - structure defining one frame of EGL.
 *
 * Each frame may contain one or more planes depending on whether the surface is Multiplanar or not.
 * Each plane of EGLFrame is represented by ::cudaEglPlaneDesc which is defined as:
 * \code
 * typedef struct cudaEglPlaneDesc_st {
 *     unsigned int width;
 *     unsigned int height;
 *     unsigned int depth;
 *     unsigned int pitch;
 *     unsigned int numChannels;
 *     struct cudaChannelFormatDesc channelDesc;
 *     unsigned int reserved[4];
 * } cudaEglPlaneDesc;
 * \endcode

*/
typedef struct cudaEglFrame_st {
   union {
       cudaArray_t            pArray[CUDA_EGL_MAX_PLANES];     /**< Array of CUDA arrays corresponding to each plane*/
       struct cudaPitchedPtr  pPitch[CUDA_EGL_MAX_PLANES];     /**< Array of Pointers corresponding to each plane*/
   } frame;
   cudaEglPlaneDesc planeDesc[CUDA_EGL_MAX_PLANES];     /**< CUDA EGL Plane Descriptor ::cudaEglPlaneDesc*/
   unsigned int planeCount;                             /**< Number of planes */
   cudaEglFrameType frameType;                          /**< Array or Pitch */
   cudaEglColorFormat eglColorFormat;                   /**< CUDA EGL Color Format*/
} cudaEglFrame;

/**
 * CUDA EGLSream Connection
 */
typedef struct  CUeglStreamConnection_st *cudaEglStreamConnection;

/** @} */ /* END CUDART_TYPES */

/**
 * \addtogroup CUDART_EGL EGL Interoperability
 * This section describes the EGL interoperability functions of the CUDA
 * runtime application programming interface.
 *
 * @{
 */

/**
 * \brief Registers an EGL image
 *
 * Registers the EGLImageKHR specified by \p image for access by
 * CUDA. A handle to the registered object is returned as \p pCudaResource.
 * Additional Mapping/Unmapping is not required for the registered resource and
 * ::cudaGraphicsResourceGetMappedEglFrame can be directly called on the \p pCudaResource.
 *
 * The application will be responsible for synchronizing access to shared objects.
 * The application must ensure that any pending operation which access the objects have completed
 * before passing control to CUDA. This may be accomplished by issuing and waiting for
 * glFinish command on all GLcontexts (for OpenGL and likewise for other APIs).
 * The application will be also responsible for ensuring that any pending operation on the
 * registered CUDA resource has completed prior to executing subsequent commands in other APIs
 * accesing the same memory objects.
 * This can be accomplished by calling cuCtxSynchronize or cuEventSynchronize (preferably).
 *
 * The surface's intended usage is specified using \p flags, as follows:
 *
 * - ::cudaGraphicsRegisterFlagsNone: Specifies no hints about how this
 *   resource will be used. It is therefore assumed that this resource will be
 *   read from and written to by CUDA. This is the default value.
 * - ::cudaGraphicsRegisterFlagsReadOnly: Specifies that CUDA
 *   will not write to this resource.
 * - ::cudaGraphicsRegisterFlagsWriteDiscard: Specifies that
 *   CUDA will not read from this resource and will write over the
 *   entire contents of the resource, so none of the data previously
 *   stored in the resource will be preserved.
 *
 * The EGLImageKHR is an object which can be used to create EGLImage target resource. It is defined as a void pointer.
 * typedef void* EGLImageKHR
 *
 * \param pCudaResource   - Pointer to the returned object handle
 * \param image           - An EGLImageKHR image which can be used to create target resource.
 * \param flags           - Map flags
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidResourceHandle,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaGraphicsUnregisterResource,
 * ::cudaGraphicsResourceGetMappedEglFrame,
 * ::cuGraphicsEGLRegisterImage
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsEGLRegisterImage(struct cudaGraphicsResource **pCudaResource, EGLImageKHR image, unsigned int flags);

/**
 * \brief Connect CUDA to EGLStream as a consumer.
 *
 * Connect CUDA as a consumer to EGLStreamKHR specified by \p eglStream.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn              - Pointer to the returned connection handle
 * \param eglStream         - EGLStreamKHR handle
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamConsumerDisconnect,
 * ::cudaEGLStreamConsumerAcquireFrame,
 * ::cudaEGLStreamConsumerReleaseFrame,
 * ::cuEGLStreamConsumerConnect
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamConsumerConnect(cudaEglStreamConnection *conn, EGLStreamKHR eglStream);

/**
 * \brief Connect CUDA to EGLStream as a consumer with given flags.
 *
 * Connect CUDA as a consumer to EGLStreamKHR specified by \p stream with specified \p flags defined by
 * ::cudaEglResourceLocationFlags.
 *
 * The flags specify whether the consumer wants to access frames from system memory or video memory.
 * Default is ::cudaEglResourceLocationVidmem.
 *
 * \param conn              - Pointer to the returned connection handle
 * \param eglStream         - EGLStreamKHR handle
 * \param flags             - Flags denote intended location - system or video.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamConsumerDisconnect,
 * ::cudaEGLStreamConsumerAcquireFrame,
 * ::cudaEGLStreamConsumerReleaseFrame,
 * ::cuEGLStreamConsumerConnectWithFlags
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection *conn, EGLStreamKHR eglStream, unsigned int flags);

/**
 * \brief Disconnect CUDA as a consumer to EGLStream .
 *
 * Disconnect CUDA as a consumer to EGLStreamKHR.
 *
 * \param conn            - Conection to disconnect.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamConsumerConnect,
 * ::cudaEGLStreamConsumerAcquireFrame,
 * ::cudaEGLStreamConsumerReleaseFrame,
 * ::cuEGLStreamConsumerDisconnect
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection *conn);

/**
 * \brief Acquire an image frame from the EGLStream with CUDA as a consumer.
 *
 * Acquire an image frame from EGLStreamKHR.
 * ::cudaGraphicsResourceGetMappedEglFrame can be called on \p pCudaResource to get
 * ::cudaEglFrame.
 *
 * \param conn            - Connection on which to acquire
 * \param pCudaResource   - CUDA resource on which the EGLStream frame will be mapped for use.
 * \param pStream         - CUDA stream for synchronization and any data migrations
 * implied by ::cudaEglResourceLocationFlags.
 * \param timeout         - Desired timeout in usec.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * ::cudaErrorLaunchTimeout
 *
 * \sa
 * ::cudaEGLStreamConsumerConnect,
 * ::cudaEGLStreamConsumerDisconnect,
 * ::cudaEGLStreamConsumerReleaseFrame,
 * ::cuEGLStreamConsumerAcquireFrame
 */

extern __host__ cudaError_t CUDARTAPI cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection *conn,
        cudaGraphicsResource_t *pCudaResource, cudaStream_t *pStream, unsigned int timeout);
/**
 * \brief Releases the last frame acquired from the EGLStream.
 *
 * Release the acquired image frame specified by \p pCudaResource to EGLStreamKHR.
 *
 * \param conn            - Connection on which to release
 * \param pCudaResource   - CUDA resource whose corresponding frame is to be released
 * \param pStream         - CUDA stream on which release will be done.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamConsumerConnect,
 * ::cudaEGLStreamConsumerDisconnect,
 * ::cudaEGLStreamConsumerAcquireFrame,
 * ::cuEGLStreamConsumerReleaseFrame
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection *conn,
                                                  cudaGraphicsResource_t pCudaResource, cudaStream_t *pStream);

/**
 * \brief Connect CUDA to EGLStream as a producer.
 *
 * Connect CUDA as a producer to EGLStreamKHR specified by \p stream.
 *
 * The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one
 * API to another.
 *
 * \param conn   - Pointer to the returned connection handle
 * \param eglStream - EGLStreamKHR handle
 * \param width  - width of the image to be submitted to the stream
 * \param height - height of the image to be submitted to the stream
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamProducerDisconnect,
 * ::cudaEGLStreamProducerPresentFrame,
 * ::cudaEGLStreamProducerReturnFrame,
 * ::cuEGLStreamProducerConnect
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamProducerConnect(cudaEglStreamConnection *conn,
                                                EGLStreamKHR eglStream, EGLint width, EGLint height);

/**
 * \brief Disconnect CUDA as a producer  to EGLStream .
 *
 * Disconnect CUDA as a producer to EGLStreamKHR.
 *
 * \param conn            - Conection to disconnect.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamProducerConnect,
 * ::cudaEGLStreamProducerPresentFrame,
 * ::cudaEGLStreamProducerReturnFrame,
 * ::cuEGLStreamProducerDisconnect
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamProducerDisconnect(cudaEglStreamConnection *conn);

/**
 * \brief Present a CUDA eglFrame to the EGLStream with CUDA as a producer.
 *
 * The ::cudaEglFrame is defined as:
 * \code
 * typedef struct cudaEglFrame_st {
 *     union {
 *         cudaArray_t            pArray[CUDA_EGL_MAX_PLANES];
 *         struct cudaPitchedPtr  pPitch[CUDA_EGL_MAX_PLANES];
 *     } frame;
 *     cudaEglPlaneDesc planeDesc[CUDA_EGL_MAX_PLANES];
 *     unsigned int planeCount;
 *     cudaEglFrameType frameType;
 *     cudaEglColorFormat eglColorFormat;
 * } cudaEglFrame;
 * \endcode
 *
 * For ::cudaEglFrame of type ::cudaEglFrameTypePitch, the application may present sub-region of a memory
 * allocation. In that case, ::cudaPitchedPtr::ptr will specify the start address of the sub-region in
 * the allocation and ::cudaEglPlaneDesc will specify the dimensions of the sub-region.
 *
 * \param conn            - Connection on which to present the CUDA array
 * \param eglframe        - CUDA Eglstream Proucer Frame handle to be sent to the consumer over EglStream.
 * \param pStream         - CUDA stream on which to present the frame.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamProducerConnect,
 * ::cudaEGLStreamProducerDisconnect,
 * ::cudaEGLStreamProducerReturnFrame,
 * ::cuEGLStreamProducerPresentFrame
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection *conn,
                                                 cudaEglFrame eglframe, cudaStream_t *pStream);

/**
 * \brief Return the CUDA eglFrame to the EGLStream last released by the consumer.
 * 
 * This API can potentially return cudaErrorLaunchTimeout if the consumer has not 
 * returned a frame to EGL stream. If timeout is returned the application can retry.
 *
 * \param conn            - Connection on which to present the CUDA array
 * \param eglframe        - CUDA Eglstream Proucer Frame handle returned from the consumer over EglStream.
 * \param pStream         - CUDA stream on which to return the frame.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \sa
 * ::cudaEGLStreamProducerConnect,
 * ::cudaEGLStreamProducerDisconnect,
 * ::cudaEGLStreamProducerPresentFrame,
 * ::cuEGLStreamProducerReturnFrame
 */
extern __host__ cudaError_t CUDARTAPI cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection *conn,
                                                cudaEglFrame *eglframe, cudaStream_t *pStream);

/**
 * \brief Get an eglFrame through which to access a registered EGL graphics resource.
 *
 * Returns in \p *eglFrame an eglFrame pointer through which the registered graphics resource
 * \p resource may be accessed.
 * This API can only be called for EGL graphics resources.
 *
 * The ::cudaEglFrame is defined as
 * \code
 * typedef struct cudaEglFrame_st {
 *     union {
 *         cudaArray_t             pArray[CUDA_EGL_MAX_PLANES];
 *         struct cudaPitchedPtr   pPitch[CUDA_EGL_MAX_PLANES];
 *     } frame;
 *     cudaEglPlaneDesc planeDesc[CUDA_EGL_MAX_PLANES];
 *     unsigned int planeCount;
 *     cudaEglFrameType frameType;
 *     cudaEglColorFormat eglColorFormat;
 * } cudaEglFrame;
 * \endcode
 *
 *
 * \param eglFrame   - Returned eglFrame.
 * \param resource   - Registered resource to access.
 * \param index      - Index for cubemap surfaces.
 * \param mipLevel   - Mipmap level for the subresource to access.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown
 *
 * \note Note that in case of multiplanar \p *eglFrame, pitch of only first plane (unsigned int cudaEglPlaneDesc::pitch) is to be considered by the application.
 *
 * \sa
 * ::cudaGraphicsSubResourceGetMappedArray,
 * ::cudaGraphicsResourceGetMappedPointer,
 * ::cuGraphicsResourceGetMappedEglFrame
 */
extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame,
                                        cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel);

/**
 * \brief Creates an event from EGLSync object
 *
 * Creates an event *phEvent from an EGLSyncKHR eglSync with the flages specified
 * via \p flags. Valid flags include:
 * - ::cudaEventDefault: Default event creation flag.
 * - ::cudaEventBlockingSync: Specifies that the created event should use blocking
 * synchronization.  A CPU thread that uses ::cudaEventSynchronize() to wait on
 * an event created with this flag will block until the event has actually
 * been completed.
 *
 * ::cudaEventRecord and TimingData are not supported for events created from EGLSync.
 *
 * The EGLSyncKHR is an opaque handle to an EGL sync object.
 * typedef void* EGLSyncKHR
 *
 * \param phEvent - Returns newly created event
 * \param eglSync - Opaque handle to EGLSync object
 * \param flags   - Event creation flags
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorMemoryAllocation
 *
 * \sa
 * ::cudaEventQuery,
 * ::cudaEventSynchronize,
 * ::cudaEventDestroy
 */
extern __host__ cudaError_t CUDARTAPI cudaEventCreateFromEGLSync(cudaEvent_t *phEvent, EGLSyncKHR eglSync, unsigned int flags);

/** @} */ /* END CUDART_EGL */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* __CUDA_EGL_INTEROP_H__ */

