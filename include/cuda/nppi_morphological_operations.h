 /* Copyright 2009-2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved. 
  * 
  * NOTICE TO LICENSEE: 
  * 
  * The source code and/or documentation ("Licensed Deliverables") are 
  * subject to NVIDIA intellectual property rights under U.S. and 
  * international Copyright laws. 
  * 
  * The Licensed Deliverables contained herein are PROPRIETARY and 
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and 
  * conditions of a form of NVIDIA software license agreement by and 
  * between NVIDIA and Licensee ("License Agreement") or electronically 
  * accepted by Licensee.  Notwithstanding any terms or conditions to 
  * the contrary in the License Agreement, reproduction or disclosure 
  * of the Licensed Deliverables to any third party without the express 
  * written consent of NVIDIA is prohibited. 
  * 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
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
#ifndef NV_NPPI_MORPHOLOGICAL_OPERATIONS_H
#define NV_NPPI_MORPHOLOGICAL_OPERATIONS_H
 
/**
 * \file nppi_morphological_operations.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_morphological_operations Morphological Operations
 *  @ingroup nppi
 *
 * Morphological image operations. 
 *
 * Morphological operations are classified as \ref neighborhood_operations. 
 *
 * @{
 *
 * These functions can be found in the nppim library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
 *
 */

/** @defgroup image_dilate Dilation
 *
 * Dilation computes the output pixel as the maximum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero do not 
 * participate in the maximum search.
 *
 * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
 *
 * <h3><a name="CommonDilateParameters">Common parameters for nppiDilate functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus 
nppiDilate_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three-channel 8-bit unsigned integer dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus 
nppiDilate_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 8-bit unsigned integer dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus
nppiDilate_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 8-bit unsigned integer dilation, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus
nppiDilate_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single-channel 16-bit unsigned integer dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus 
nppiDilate_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three-channel 16-bit unsigned integer dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus 
nppiDilate_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 16-bit unsigned integer dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus
nppiDilate_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 16-bit unsigned integer dilation, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus
nppiDilate_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single-channel 32-bit floating-point dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus 
nppiDilate_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three-channel 32-bit floating-point dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus 
nppiDilate_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 32-bit floating-point dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus
nppiDilate_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 32-bit floating-point dilation, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateParameters">Common parameters for nppiDilate functions</a>.
 *
 */
NppStatus
nppiDilate_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/** @} image_dilate */

/** @defgroup image_dilate_border Dilation with border control
 *
 * Dilation computes the output pixel as the maximum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero do not 
 * participate in the maximum search. For gray scale dilation the mask contains signed mask values
 * which are added to the corresponding source image sample value before determining the maximun value after clamping.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image.
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiDilateBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilateBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three-channel 8-bit unsigned integer dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiDilateBorder_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilateBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus
nppiDilateBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilateBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer dilation with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus
nppiDilateBorder_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilateBorder_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 16-bit unsigned integer dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiDilateBorder_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilateBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three-channel 16-bit unsigned integer dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiDilateBorder_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilateBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus
nppiDilateBorder_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilateBorder_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer dilation with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus
nppiDilateBorder_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                              const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilateBorder_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                          const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 32-bit floating-point dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiDilateBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilateBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three-channel 32-bit floating-point dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiDilateBorder_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilateBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus
nppiDilateBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilateBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point dilation with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus
nppiDilateBorder_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                              const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilateBorder_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                          const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 8-bit unsigned integer gray scale dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiGrayDilateBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                const Npp32s * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiGrayDilateBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32s * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 32-bit floating point gray scale dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilateBorderParameters">Common parameters for nppiDilateBorder functions</a>.
 *
 */
NppStatus 
nppiGrayDilateBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                 const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiGrayDilateBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                             const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/** @} image_dilate_border */

/** @defgroup image_dilate_3x3 Dilate3x3
 *
 * Dilation using a 3x3 mask with the anchor at its center pixel.
 *
 * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
 *
 * <h3><a name="CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three-channel 8-bit unsigned integer 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 8-bit unsigned integer 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus
nppiDilate3x3_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 8-bit unsigned integer 3x3 dilation, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus
nppiDilate3x3_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Single-channel 16-bit unsigned integer 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three-channel 16-bit unsigned integer 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 16-bit unsigned integer 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus
nppiDilate3x3_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 16-bit unsigned integer 3x3 dilation, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus
nppiDilate3x3_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Single-channel 32-bit floating-point 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three-channel 32-bit floating-point 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 32-bit floating-point 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus
nppiDilate3x3_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 32-bit floating-point 3x3 dilation, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus
nppiDilate3x3_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Single-channel 64-bit floating-point 3x3 dilation.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3Parameters">Common parameters for nppiDilate3x3 functions</a>.
 *
 */
NppStatus 
nppiDilate3x3_64f_C1R_Ctx(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} image_dilate_3x3 */

/** @defgroup image_dilate_3x3_border Dilate3x3Border
 *
 * Dilation using a 3x3 mask with the anchor at its center pixel with border control.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image.
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus 
nppiDilate3x3Border_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3Border_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three-channel 8-bit unsigned integer 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus 
nppiDilate3x3Border_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3Border_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus
nppiDilate3x3Border_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3Border_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus
nppiDilate3x3Border_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3Border_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single-channel 16-bit unsigned integer 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus 
nppiDilate3x3Border_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3Border_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three-channel 16-bit unsigned integer 3x3 dilation with border control.
 * 
 */
NppStatus 
nppiDilate3x3Border_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3Border_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer 3x3 dilation with border control.
 * 
 */
NppStatus
nppiDilate3x3Border_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3Border_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.
 * 
 */
NppStatus
nppiDilate3x3Border_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3Border_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single-channel 32-bit floating-point 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus 
nppiDilate3x3Border_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3Border_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three-channel 32-bit floating-point 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus 
nppiDilate3x3Border_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiDilate3x3Border_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point 3x3 dilation with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus
nppiDilate3x3Border_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3Border_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point 3x3 dilation with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonDilate3x3BorderParameters">Common parameters for nppiDilate3x3Border functions</a>.
 *
 */
NppStatus
nppiDilate3x3Border_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiDilate3x3Border_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} image_dilate_3x3_border */

/** @defgroup image_erode Erode
 *
 * Erosion computes the output pixel as the minimum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero do not 
 * participate in the maximum search.
 *
 * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
 *
 * <h3><a name="CommonErodeParameters">Common parameters for nppiErode functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */


/**
 * Single-channel 8-bit unsigned integer erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus 
nppiErode_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three-channel 8-bit unsigned integer erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus 
nppiErode_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                     const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 8-bit unsigned integer erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus
nppiErode_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                     const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiErode_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                 const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 8-bit unsigned integer erosion, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus
nppiErode_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiErode_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single-channel 16-bit unsigned integer erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus 
nppiErode_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three-channel 16-bit unsigned integer erosion.
 * 
 */
NppStatus 
nppiErode_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 16-bit unsigned integer erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus
nppiErode_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiErode_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 16-bit unsigned integer erosion, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus
nppiErode_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiErode_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Single-channel 32-bit floating-point erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus 
nppiErode_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Three-channel 32-bit floating-point erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus 
nppiErode_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 32-bit floating-point erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus
nppiErode_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                      const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiErode_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                  const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/**
 * Four-channel 32-bit floating-point erosion, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeParameters">Common parameters for nppiErode functions</a>.
 *
 */
NppStatus
nppiErode_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx);

NppStatus
nppiErode_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                   const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor);

/** @} image_erode */

/** @defgroup image_erode_border Erosion with border control
 *
 * Erosion computes the output pixel as the minimum pixel value of the pixels
 * under the mask. Pixels who's corresponding mask values are zero do not 
 * participate in the minimum search. For gray scale erosion the mask contains signed mask values
 * which are added to the corresponding source image sample value before determining the minimum value after clamping.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image.
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiErodeBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErodeBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three-channel 8-bit unsigned integer erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiErodeBorder_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErodeBorder_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus
nppiErodeBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                           const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErodeBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                       const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer erosion with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus
nppiErodeBorder_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErodeBorder_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI,
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 16-bit unsigned integer erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiErodeBorder_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErodeBorder_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three-channel 16-bit unsigned integer erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiErodeBorder_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErodeBorder_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus
nppiErodeBorder_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErodeBorder_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer erosion with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus
nppiErodeBorder_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErodeBorder_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, NppiSize oSizeROI,
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 32-bit floating-point erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiErodeBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErodeBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Three-channel 32-bit floating-point erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiErodeBorder_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErodeBorder_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus
nppiErodeBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErodeBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                        const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point erosion with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus
nppiErodeBorder_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                             const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErodeBorder_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, NppiSize oSizeROI,
                         const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 8-bit unsigned integer gray scale erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiGrayErodeBorder_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                               const Npp32s * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiGrayErodeBorder_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                           const Npp32s * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/**
 * Single-channel 32-bit floating point gray scale erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErodeBorderParameters">Common parameters for nppiErodeBorder functions</a>.
 *
 */
NppStatus 
nppiGrayErodeBorder_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                                const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiGrayErodeBorder_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, 
                            const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType);

/** @} image_erode_border */

/** @defgroup image_erode_3x3 Erode3x3
 *
 * Erosion using a 3x3 mask with the anchor at its center pixel.
 *
 * It is the user's responsibility to avoid \ref sampling_beyond_image_boundaries.
 *
 * <h3><a name="CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */


/**
 * Single-channel 8-bit unsigned integer 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three-channel 8-bit unsigned integer 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 8-bit unsigned integer 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus
nppiErode3x3_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 8-bit unsigned integer 3x3 erosion, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus
nppiErode3x3_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Single-channel 16-bit unsigned integer 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three-channel 16-bit unsigned integer 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 16-bit unsigned integer 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus
nppiErode3x3_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 16-bit unsigned integer 3x3 erosion, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus
nppiErode3x3_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Single-channel 32-bit floating-point 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Three-channel 32-bit floating-point 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 32-bit floating-point 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus
nppiErode3x3_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Four-channel 32-bit floating-point 3x3 erosion, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus
nppiErode3x3_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/**
 * Single-channel 64-bit floating-point 3x3 erosion.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3Parameters">Common parameters for nppiErode3x3 functions</a>.
 *
 */
NppStatus 
nppiErode3x3_64f_C1R_Ctx(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI);

/** @} image_erode */

/** @defgroup image_erode_3x3_border Erode3x3Border
 *
 * Erosion using a 3x3 mask with the anchor at its center pixel with border control.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image.
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus 
nppiErode3x3Border_8u_C1R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                    Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3Border_8u_C1R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three-channel 8-bit unsigned integer 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus 
nppiErode3x3Border_8u_C3R_Ctx(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                    Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3Border_8u_C3R(const Npp8u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                Npp8u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus
nppiErode3x3Border_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                    Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3Border_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 8-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus
nppiErode3x3Border_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3Border_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single-channel 16-bit unsigned integer 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus 
nppiErode3x3Border_16u_C1R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3Border_16u_C1R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three-channel 16-bit unsigned integer 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus 
nppiErode3x3Border_16u_C3R_Ctx(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3Border_16u_C3R(const Npp16u * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp16u * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus
nppiErode3x3Border_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3Border_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 16-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus
nppiErode3x3Border_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                      Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3Border_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                  Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Single-channel 32-bit floating-point 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus 
nppiErode3x3Border_32f_C1R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3Border_32f_C1R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Three-channel 32-bit floating-point 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus 
nppiErode3x3Border_32f_C3R_Ctx(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiErode3x3Border_32f_C3R(const Npp32f * pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp32f * pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point 3x3 erosion with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus
nppiErode3x3Border_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                     Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3Border_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                 Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/**
 * Four-channel 32-bit floating-point 3x3 erosion with border control, ignoring alpha-channel.
 * 
 * For common parameter descriptions, see <a href="#CommonErode3x3BorderParameters">Common parameters for nppiErode3x3Border functions</a>.
 *
 */
NppStatus
nppiErode3x3Border_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                      Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiErode3x3Border_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, 
                                  Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppiBorderType eBorderType);

/** @} image_erode_3x3_border */

/** @defgroup image_morph ComplexImageMorphology
 * Complex image morphological operations.
 *
 * @{
 *
 */

/** @defgroup image_morph_get_buffer_size MorphGetBufferSize
 *
 * Before calling any of the MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder, MorphBlackHatBorder, or MorphGradientBorder
 * functions the application first needs to call the corresponding
 * MorphGetBufferSize to determine the amount of device memory to allocate as a working buffer.  The application allocated device memory
 * is then passed as the pBuffer parameter to the corresponding MorphXXXBorder function.
 *
 * <h3><a name="CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions include:</a></h3>
 *
 * \param oSizeROI \ref roi_specification.
 * \param hpBufferSize Required buffer size in bytes.
 *
 * @{
 *
 */

/**
 * Calculate scratch buffer size needed for 1 channel 8-bit unsigned integer MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder, 
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_8u_C1R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 3 channel 8-bit unsigned integer MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder, 
 * MorphBlackHatBorder or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_8u_C3R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 4 channel 8-bit unsigned integer MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder,
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_8u_C4R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 1 channel 16-bit unsigned integer MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder,
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_16u_C1R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 1 channel 16-bit signed integer MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder,
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_16s_C1R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 1 channel 32-bit floating point MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder,
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_32f_C1R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 3 channel 32-bit floating point MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder,
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_32f_C3R(NppiSize oSizeROI, int * hpBufferSize);

/**
 * Calculate scratch buffer size needed for 4 channel 32-bit floating point MorphCloseBorder, MorphOpenBorder, MorphTopHatBorder,
 * MorphBlackHatBorder, or MorphGradientBorder function based on destination image oSizeROI width and height.
 *
 * For common parameter descriptions, see <a href="#CommonMorphGetBufferSizeParameters">Common parameters for nppiMorphGetBufferSize functions</a>.
 *
 */
NppStatus 
nppiMorphGetBufferSize_32f_C4R(NppiSize oSizeROI, int * hpBufferSize);

/** @} image_morph_get_buffer_size */

/** @defgroup image_morph_close_border MorphCloseBorder
 * Dilation followed by Erosion with border control.
 *
 * Morphological close computes the output pixel as the maximum pixel value of the pixels
 * under the mask followed by a second pass using the result of the first pass as input which outputs 
 * the minimum pixel value of the pixels under the same mask. 
 * Pixels who's corresponding mask values are zero do not participate in the maximum or minimum search.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image. 
 * The mask is centered over the source image pixel being tested.
 *
 * Before calling any of the MorphCloseBorder functions the application first needs to call the corresponding
 * MorphGetBufferSize to determine the amount of device memory to allocate as a working buffer.  The allocated device memory
 * is then passed as the pBuffer parameter to the corresponding MorphCloseBorder function.
 *
 * Use the oSrcOffset and oSrcSize parameters to control where the border control operation is applied to the source image ROI borders.  
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param pBuffer Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned integer morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned integer morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 8-bit unsigned integer morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned integer morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed integer morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 32-bit floating point morphological close with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphCloseBorderParameters">Common parameters for nppiMorphCloseBorder functions</a>.
 *
 */
NppStatus 
nppiMorphCloseBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphCloseBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/** @} image_morph_close_border */

/** @defgroup image_morph_open_border MorphOpenBorder
 * Erosion followed by Dilation with border control.
 *
 * Morphological open computes the output pixel as the minimum pixel value of the pixels
 * under the mask followed by a second pass using the result of the first pass as input which outputs 
 * the maximum pixel value of the pixels under the same mask. 
 * Pixels who's corresponding mask values are zero do not participate in the minimum or maximum search.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image. 
 * The mask is centered over the source image pixel being tested.
 *
 * Before calling any of the MorphOpenBorder functions the application first needs to call the corresponding
 * MorphGetBufferSize to determine the amount of device memory to allocate as a working buffer.  The allocated device memory
 * is then passed as the pBuffer parameter to the corresponding MorphOpenBorder function. 
 *
 * Use the oSrcOffset and oSrcSize parameters to control where the border control operation is applied to the source image ROI borders.  
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param pBuffer Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned integer morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                           NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned integer morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                           NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 8-bit unsigned integer morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                           NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned integer morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed integer morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 32-bit floating point morphological open with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphOpenBorderParameters">Common parameters for nppiMorphOpenBorder functions</a>.
 *
 */
NppStatus 
nppiMorphOpenBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphOpenBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                            NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/** @} image_morph_open_border */

/** @defgroup image_morph_top_hat_border MorphToHatBorder
 * Source pixel minus the morphological open pixel result with border control.
 *
 * Morphological top hat computes the output pixel as the source pixel minus the morphological open result of the pixels
 * under the mask. 
 * Pixels who's corresponding mask values are zero do not participate in the maximum or minimum search.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image. 
 * The mask is centered over the source image pixel being tested.
 *
 * Before calling any of the MorphTopHatBorder functions the application first needs to call the corresponding
 * MorphGetBufferSize to determine the amount of device memory to allocate as a working buffer.  The allocated device memory
 * is then passed as the pBuffer parameter to the corresponding MorphTopHatBorder function.
 *
 * Use the oSrcOffset and oSrcSize parameters to control where the border control operation is applied to the source image ROI borders.  
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param pBuffer Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned integer morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned integer morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 8-bit unsigned integer morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                 NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                             NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned integer morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                              NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed integer morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                  NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                              NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                  NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                              NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                  NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                              NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 32-bit floating point morphological top hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphTopHatBorderParameters">Common parameters for nppiMorphTopHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphTopHatBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                  NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphTopHatBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                              NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/** @} image_morph_top_hat_border */

/** @defgroup image_morph_black_hat_border MorphBlackHatBorder
 * Morphological close pixel result minus source pixel with border control.
 *
 * Morphological black hat computes the output pixel as the morphological close pixel value of the pixels
 * under the mask minus the source pixel value. 
 * Pixels who's corresponding mask values are zero do not participate in the maximum or minimum search.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image. 
 * The mask is centered over the source image pixel being tested.
 *
 * Before calling any of the MorphBlackHatBorder functions the application first needs to call the corresponding
 * MorphGetBufferSize to determine the amount of device memory to allocate as a working buffer.  The allocated device memory
 * is then passed as the pBuffer parameter to the corresponding MorphBlackHatBorder function.
 *
 * Use the oSrcOffset and oSrcSize parameters to control where the border control operation is applied to the source image ROI borders.  
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param pBuffer Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned integer morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned integer morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus
nppiMorphBlackHatBorder_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 8-bit unsigned integer morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned integer morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed integer morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 32-bit floating point morphological black hat with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphBlackHatBorderParameters">Common parameters for nppiMorphBlackHatBorder functions</a>.
 *
 */
NppStatus 
nppiMorphBlackHatBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphBlackHatBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/** @} image_morph_black_hat_border */

/** @defgroup image_morph_gradient_border MorphGradientBorder
 * Morphological dilated pixel result minus morphological eroded pixel result with border control.
 *
 * Morphological gradient computes the output pixel as the morphological dilated pixel value of the pixels
 * under the mask minus the morphological eroded pixel value of the pixels under the mask. 
 * Pixels who's corresponding mask values are zero do not participate in the maximum or minimum search.
 *
 * If any portion of the mask overlaps the source image boundary the requested border type 
 * operation is applied to all mask pixels which fall outside of the source image. 
 * The mask is centered over the source image pixel being tested.
 *
 * Before calling any of the MorphGradientBorder functions the application first needs to call the corresponding
 * MorphGetBufferSize to determine the amount of device memory to allocate as a working buffer.  The allocated device memory
 * is then passed as the pBuffer parameter to the corresponding MorphGradientBorder function.
 *
 * Use the oSrcOffset and oSrcSize parameters to control where the border control operation is applied to the source image ROI borders.  
 *
 * Currently only the NPP_BORDER_REPLICATE border type operation is supported.
 *
 * <h3><a name="CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions include:</a></h3>
 *
 * \param pSrc  \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Source image width and height in pixels relative to pSrc.
 * \param oSrcOffset Source image starting point relative to pSrc. 
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask Pointer to the start address of the mask array
 * \param oMaskSize Width and Height mask array.
 * \param oAnchor X and Y offsets of the mask origin frame of reference
 *        w.r.t the source pixel.
 * \param pBuffer Pointer to device memory scratch buffer at least as large as value returned by the corresponding MorphGetBufferSize call.
 * \param eBorderType The border type operation to be applied at source image border boundaries.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned integer morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 8-bit unsigned integer morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 8-bit unsigned integer morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit unsigned integer morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 16-bit signed integer morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 1 channel 32-bit floating point morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                     NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 3 channel 32-bit floating point morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/**
 * 4 channel 32-bit floating point morphological gradient with border control.
 * 
 * For common parameter descriptions, see <a href="#CommonMorphGradientBorderParameters">Common parameters for nppiMorphGradientBorder functions</a>.
 *
 */
NppStatus 
nppiMorphGradientBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType, NppStreamContext nppStreamCtx);

NppStatus 
nppiMorphGradientBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, const Npp8u * pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u * pBuffer, NppiBorderType eBorderType);

/** @} image_morph_gradient_border */

/** @} image_morph */

/** @} image_morphological_operations */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPI_MORPHOLOGICAL_OPERATIONS_H */
