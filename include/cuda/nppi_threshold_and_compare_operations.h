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
#ifndef NV_NPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H
#define NV_NPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H
 
/**
 * \file nppi_threshold_and_compare_operations.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup image_threshold_and_compare_operations Threshold and Compare Operations
 *  @ingroup nppi
 *
 * Methods for pixel-wise threshold and compare operations.
 *
 * @{
 *
 * These functions can be found in the nppitc library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
 *
 */

/** 
 * @defgroup image_thresholding_operations Thresholding Operations
 * Various types of image thresholding operations.
 *
 * @{
 *
 */

/** 
 * @defgroup image_threshold_operations Threshold Operations
 * Threshold image pixels.
 *
 * <h3><a name="CommonThresholdParameters">Common parameters for nppiThreshold non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                     Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, 
                               const Npp8u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                      Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16u nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                      Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16s nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16s nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp32f nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f nThreshold, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp32f nThreshold, NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                     Npp8u * pDst, int nDstStep, 
                               NppiSize oSizeROI, 
                               const Npp8u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                      Npp16u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16u rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                      Npp16s * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp16s rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16s rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp32f rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp32f rThresholds[3], NppCmpOp eComparisonOperation); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                          Npp8u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                      Npp8u * pDst, int nDstStep, 
                                NppiSize oSizeROI, 
                                const Npp8u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp8u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                           Npp16u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                       Npp16u * pDst, int nDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp16u rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                           Npp16s * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                       Npp16s * pDst, int nDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp16s rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp16s rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           Npp32f * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                       Npp32f * pDst, int nDstStep, 
                                 NppiSize oSizeROI, 
                                 const Npp32f rThresholds[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdParameters">Common parameters for nppiThreshold functions</a>.
 *
 */
NppStatus nppiThreshold_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp32f rThresholds[3], NppCmpOp eComparisonOperation);

/** @} image_threshold_operations */

/** 
 * @defgroup image_threshold_greater_than_operations Threshold Greater Than Operations
 * Threshold greater than image pixels.
 *
 * <h3><a name="CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u nThreshold); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s nThreshold); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f nThreshold); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u rThresholds[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GT_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                             Npp8u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp8u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                              Npp16u * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                              Npp16s * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16s rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                              Npp32f * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanParameters">Common parameters for nppiThreshold_GT functions</a>.
 *
 */
NppStatus nppiThreshold_GT_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp32f rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GT_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3]);

/** @} image_threshold_greater_than_operations */

/** 
 * @defgroup image_threshold_less_than_operations Threshold Less Than Operations
 * Threshold less than image pixels.
 *
 * <h3><a name="CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u nThreshold); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s nThreshold); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f nThreshold); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f nThreshold, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                        Npp8u * pDst, int nDstStep, 
                                  NppiSize oSizeROI, 
                                  const Npp8u rThresholds[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                         Npp16u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                         Npp16s * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp16s rThresholds[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp32f rThresholds[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LT_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                             Npp8u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp8u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                              Npp16u * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16u rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                              Npp16s * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16s rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                              Npp32f * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * value is set to nThreshold, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanParameters">Common parameters for nppiThreshold_LT functions</a>.
 *
 */
NppStatus nppiThreshold_LT_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp32f rThresholds[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LT_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3]);

/** @} image_threshold_less_than_operations */

/** 
 * @defgroup image_threshold_value_operations Threshold Value Operations
 * Replace thresholded image pixels with a value.
 *
 * <h3><a name="CommonThresholdValueParameters">Common parameters for nppiThreshold_Val non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nValue The threshold replacement value.
 * \param eComparisonOperation The type of comparison operation to be used. The only valid
 *      values are: NPP_CMP_LESS and NPP_CMP_GREATER.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid
 * comparison operation type is specified.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                             Npp8u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u nThreshold, const Npp8u nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u nThreshold, const Npp8u nValue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp8u nThreshold, const Npp8u nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u nThreshold, const Npp8u nValue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                              Npp16u * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u nThreshold, const Npp16u nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u nThreshold, const Npp16u nValue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16u nThreshold, const Npp16u nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u nThreshold, const Npp16u nValue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                              Npp16s * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s nThreshold, const Npp16s nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s nThreshold, const Npp16s nValue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16s nThreshold, const Npp16s nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 
                                     
NppStatus nppiThreshold_Val_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s nThreshold, const Npp16s nValue, NppCmpOp eComparisonOperation); 
                                     
/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                              Npp32f * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f nThreshold, const Npp32f nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f nThreshold, const Npp32f nValue, NppCmpOp eComparisonOperation); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp32f nThreshold, const Npp32f nValue, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f nThreshold, const Npp32f nValue, NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                             Npp8u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                         Npp8u * pDst, int nDstStep, 
                                   NppiSize oSizeROI, 
                                   const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                              Npp16u * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                          Npp16u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                              Npp16s * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                              Npp32f * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_Val_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation); 


/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                              Npp8u * pDst, int nDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                          Npp8u * pDst, int nDstStep, 
                                    NppiSize oSizeROI, 
                                    const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], const Npp8u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                               Npp16u * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                           Npp16u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], const Npp16u rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                               Npp16s * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                           Npp16s * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], const Npp16s rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                               Npp32f * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                           Npp32f * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel
 * value is set to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdValueParameters">Common parameters for nppiThreshold_Val functions</a>.
 *
 */
NppStatus nppiThreshold_Val_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_Val_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], const Npp32f rValues[3], NppCmpOp eComparisonOperation);

/** @} image_threshold_value_operations */

/** 
 * @defgroup image_threshold_greater_than_value_operations Threshold Greater Than Value Operations
 * Replace image pixels greater than threshold with a value.
 *
 * <h3><a name="CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nValue The threshold replacement value.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */



/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                               Npp8u * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp8u nThreshold, const Npp8u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u nThreshold, const Npp8u nValue); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u nThreshold, const Npp8u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u nThreshold, const Npp8u nValue); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                Npp16u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16u nThreshold, const Npp16u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u nThreshold, const Npp16u nValue); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u nThreshold, const Npp16u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u nThreshold, const Npp16u nValue); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                Npp16s * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16s nThreshold, const Npp16s nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s nThreshold, const Npp16s nValue); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s nThreshold, const Npp16s nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s nThreshold, const Npp16s nValue); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                Npp32f * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp32f nThreshold, const Npp32f nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f nThreshold, const Npp32f nValue); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f nThreshold, const Npp32f nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f nThreshold, const Npp32f nValue); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                               Npp8u * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                Npp16u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], const Npp16u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiThreshold_GTVal_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]); 
                                       
/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                Npp16s * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                Npp32f * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_GTVal_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                                Npp8u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                 Npp16u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                 Npp16s * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                 Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdGreaterThanValueParameters">Common parameters for nppiThreshold_GTVal functions</a>.
 *
 */
NppStatus nppiThreshold_GTVal_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_GTVal_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** @} image_threshold_greater_than_value_operations */

/** 
 * @defgroup image_threshold_less_than_value_operations Threshold Less Than Value Operations
 * Replace image pixels less than threshold with a value.
 *
 * <h3><a name="CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThreshold The threshold value.
 * \param nValue The threshold replacement value.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                               Npp8u * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp8u nThreshold, const Npp8u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u nThreshold, const Npp8u nValue); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u nThreshold, const Npp8u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u nThreshold, const Npp8u nValue); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                Npp16u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16u nThreshold, const Npp16u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u nThreshold, const Npp16u nValue); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u nThreshold, const Npp16u nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u nThreshold, const Npp16u nValue); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                Npp16s * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16s nThreshold, const Npp16s nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s nThreshold, const Npp16s nValue); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s nThreshold, const Npp16s nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s nThreshold, const Npp16s nValue); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                Npp32f * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp32f nThreshold, const Npp32f nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f nThreshold, const Npp32f nValue); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set
 * to nValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f nThreshold, const Npp32f nValue, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f nThreshold, const Npp32f nValue); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                               Npp8u * pDst, int nDstStep, 
                                         NppiSize oSizeROI, 
                                         const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                           Npp8u * pDst, int nDstStep, 
                                     NppiSize oSizeROI, 
                                     const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                Npp16u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                            Npp16u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16u rThresholds[3], const Npp16u rValues[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                Npp16s * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                Npp32f * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTVal_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                                Npp8u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                            Npp8u * pDst, int nDstStep, 
                                      NppiSize oSizeROI, 
                                      const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u rThresholds[3], const Npp8u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp8u rThresholds[3], const Npp8u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                 Npp16u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                             Npp16u * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u rThresholds[3], const Npp16u rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16u rThresholds[3], const Npp16u rValues[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                 Npp16s * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                             Npp16s * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s rThresholds[3], const Npp16s rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp16s rThresholds[3], const Npp16s rValues[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                 Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, 
                                       NppiSize oSizeROI, 
                                       const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set
 * value is set to rValue, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueParameters">Common parameters for nppiThreshold_LTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTVal_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f rThresholds[3], const Npp32f rValues[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTVal_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, 
                                        const Npp32f rThresholds[3], const Npp32f rValues[3]);

/** @} image_threshold_less_than_value_operations */

/** 
 * @defgroup image_threshold_less_than_value_greater_than_value_operations Threshold Less Than Value Or Greater Than Value Operations
 * Replace image pixels less than thresholdLT or greater than thresholdGT with with valueLT or valueGT respectively.
 *
 * <h3><a name="CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oSizeROI \ref roi_specification.
 * \param nThresholdLT The thresholdLT value.
 * \param nValueLT The thresholdLT replacement value.
 * \param nThresholdGT The thresholdGT value.
 * \param nValueGT The thresholdGT replacement value.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes.
 *
 * @{
 *
 */


/** 
 * 1 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                                    Npp8u * pDst, int nDstStep, 
                                              NppiSize oSizeROI, 
                                              const Npp8u nThresholdLT, const Npp8u nValueLT, const Npp8u nThresholdGT, const Npp8u nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                                                Npp8u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u nThresholdLT, const Npp8u nValueLT, const Npp8u nThresholdGT, const Npp8u nValueGT); 

/** 
 * 1 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp8u nThresholdLT, const Npp8u nValueLT, const Npp8u nThresholdGT, const Npp8u nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u nThresholdLT, const Npp8u nValueLT, const Npp8u nThresholdGT, const Npp8u nValueGT); 

/** 
 * 1 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                     Npp16u * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp16u nThresholdLT, const Npp16u nValueLT, const Npp16u nThresholdGT, const Npp16u nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                                                 Npp16u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u nThresholdLT, const Npp16u nValueLT, const Npp16u nThresholdGT, const Npp16u nValueGT); 

/** 
 * 1 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp16u nThresholdLT, const Npp16u nValueLT, const Npp16u nThresholdGT, const Npp16u nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u nThresholdLT, const Npp16u nValueLT, const Npp16u nThresholdGT, const Npp16u nValueGT); 

/** 
 * 1 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                     Npp16s * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp16s nThresholdLT, const Npp16s nValueLT, const Npp16s nThresholdGT, const Npp16s nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                                                 Npp16s * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s nThresholdLT, const Npp16s nValueLT, const Npp16s nThresholdGT, const Npp16s nValueGT); 

/** 
 * 1 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp16s nThresholdLT, const Npp16s nValueLT, const Npp16s nThresholdGT, const Npp16s nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s nThresholdLT, const Npp16s nValueLT, const Npp16s nThresholdGT, const Npp16s nValueGT); 

/** 
 * 1 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                     Npp32f * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp32f nThresholdLT, const Npp32f nValueLT, const Npp32f nThresholdGT, const Npp32f nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                                 Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f nThresholdLT, const Npp32f nValueLT, const Npp32f nThresholdGT, const Npp32f nValueGT); 

/** 
 * 1 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set
 * to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp32f nThresholdLT, const Npp32f nValueLT, const Npp32f nThresholdGT, const Npp32f nValueGT, NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f nThresholdLT, const Npp32f nValueLT, const Npp32f nThresholdGT, const Npp32f nValueGT); 

/** 
 * 3 channel 8-bit unsigned char threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                                    Npp8u * pDst, int nDstStep, 
                                              NppiSize oSizeROI, 
                                              const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                                                Npp8u * pDst, int nDstStep, 
                                          NppiSize oSizeROI, 
                                          const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]); 

/** 
 * 3 channel 8-bit unsigned char in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]); 

/** 
 * 3 channel 16-bit unsigned short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                     Npp16u * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                                                 Npp16u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]); 

/** 
 * 3 channel 16-bit unsigned short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]); 

/** 
 * 3 channel 16-bit signed short threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                     Npp16s * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                                                 Npp16s * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]); 

/** 
 * 3 channel 16-bit signed short in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]); 

/** 
 * 3 channel 32-bit floating point threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                     Npp32f * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                                 Npp32f * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]); 

/** 
 * 3 channel 32-bit floating point in place threshold.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3], NppStreamContext nppStreamCtx); 

NppStatus nppiThreshold_LTValGTVal_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]); 

/** 
 * 4 channel 8-bit unsigned char image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                                     Npp8u * pDst, int nDstStep, 
                                               NppiSize oSizeROI, 
                                               const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                                                 Npp8u * pDst, int nDstStep, 
                                           NppiSize oSizeROI, 
                                           const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]);

/** 
 * 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3], const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]);

/** 
 * 4 channel 16-bit unsigned short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                                      Npp16u * pDst, int nDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                                  Npp16u * pDst, int nDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]);

/** 
 * 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, 
                                                 NppiSize oSizeROI, 
                                                 const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, 
                                             NppiSize oSizeROI, 
                                             const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3], const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]);

/** 
 * 4 channel 16-bit signed short image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                                      Npp16s * pDst, int nDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                                  Npp16s * pDst, int nDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]);

/** 
 * 4 channel 16-bit signed short in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, 
                                                 NppiSize oSizeROI, 
                                                 const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, 
                                             NppiSize oSizeROI, 
                                             const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3], const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]);

/** 
 * 4 channel 32-bit floating point image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                                      Npp32f * pDst, int nDstStep, 
                                                NppiSize oSizeROI, 
                                                const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                                  Npp32f * pDst, int nDstStep, 
                                            NppiSize oSizeROI, 
                                            const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]);

/** 
 * 4 channel 32-bit floating point in place image threshold, not affecting Alpha.
 * If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set
 * value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
 * 
 * For common parameter descriptions, see <a href="#CommonThresholdLessThanValueGreaterThanValueParameters">Common parameters for nppiThreshold_LTValGTVal functions</a>.
 *
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, 
                                                 NppiSize oSizeROI, 
                                                 const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3], NppStreamContext nppStreamCtx);

NppStatus nppiThreshold_LTValGTVal_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, 
                                             NppiSize oSizeROI, 
                                             const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3], const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]);

/** @} image_threshold_less_than_value_greater_than_value_operations */

/** @} image_thresholding_operations */

/** @defgroup image_comparison_operations Comparison Operations
 * Compare the pixels of two images or one image and a constant value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 *
 * @{
 *
 */


/** @defgroup compare_images_operations Compare Images Operations
 * Compare the pixels of two images and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 *
 * <h3><a name="CommonCompareImagesParameters">Common parameters for nppiCompare functions include:</a></h3>
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step,
                                 const Npp8u * pSrc2, int nSrc2Step,
                                       Npp8u * pDst,  int nDstStep,
                                 NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_8u_C1R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step,
                                 const Npp8u * pSrc2, int nSrc2Step,
                                       Npp8u * pDst,  int nDstStep,
                                 NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_8u_C3R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step,
                                 const Npp8u * pSrc2, int nSrc2Step,
                                       Npp8u * pDst,  int nDstStep,
                                 NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_8u_C4R(const Npp8u * pSrc1, int nSrc1Step,
                             const Npp8u * pSrc2, int nSrc2Step,
                                   Npp8u * pDst,  int nDstStep,
                             NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step,
                                  const Npp8u * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step,
                              const Npp8u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step,
                                  const Npp16u * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16u_C1R(const Npp16u * pSrc1, int nSrc1Step,
                              const Npp16u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step,
                                  const Npp16u * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16u_C3R(const Npp16u * pSrc1, int nSrc1Step,
                              const Npp16u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step,
                                  const Npp16u * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16u_C4R(const Npp16u * pSrc1, int nSrc1Step,
                              const Npp16u * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step,
                                   const Npp16u * pSrc2, int nSrc2Step,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step,
                               const Npp16u * pSrc2, int nSrc2Step,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16s_C1R_Ctx(const Npp16s * pSrc1, int nSrc1Step,
                                  const Npp16s * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16s_C1R(const Npp16s * pSrc1, int nSrc1Step,
                              const Npp16s * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16s_C3R_Ctx(const Npp16s * pSrc1, int nSrc1Step,
                                  const Npp16s * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16s_C3R(const Npp16s * pSrc1, int nSrc1Step,
                              const Npp16s * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16s_C4R_Ctx(const Npp16s * pSrc1, int nSrc1Step,
                                  const Npp16s * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16s_C4R(const Npp16s * pSrc1, int nSrc1Step,
                              const Npp16s * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_16s_AC4R_Ctx(const Npp16s * pSrc1, int nSrc1Step,
                                   const Npp16s * pSrc2, int nSrc2Step,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_16s_AC4R(const Npp16s * pSrc1, int nSrc1Step,
                               const Npp16s * pSrc2, int nSrc2Step,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                  const Npp32f * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_32f_C1R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                  const Npp32f * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_32f_C3R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image compare.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                  const Npp32f * pSrc2, int nSrc2Step,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_32f_C4R(const Npp32f * pSrc1, int nSrc1Step,
                              const Npp32f * pSrc2, int nSrc2Step,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImagesParameters">Common parameters for nppiCompare functions</a>.
 *
 */
NppStatus nppiCompare_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                   const Npp32f * pSrc2, int nSrc2Step,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompare_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step,
                               const Npp32f * pSrc2, int nSrc2Step,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** @} compare_images_operations */

/** @defgroup compare_image_with_constant_operations Compare Image With Constant Operations
 * Compare the pixels of an image with a constant value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 *
 * <h3><a name="CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value for single channel functions.
 * \param pConstants pointer to a list of constant values, one per color channel for multi-channel functions.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eComparisonOperation Specifies the comparison operation to be used in the pixel comparison.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                  const Npp8u nConstant,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_8u_C1R(const Npp8u * pSrc, int nSrcStep,
                              const Npp8u nConstant,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                  const Npp8u * pConstants,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_8u_C3R(const Npp8u * pSrc, int nSrcStep,
                              const Npp8u * pConstants,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                  const Npp8u * pConstants,
                                        Npp8u * pDst,  int nDstStep,
                                  NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_8u_C4R(const Npp8u * pSrc, int nSrcStep,
                              const Npp8u * pConstants,
                                    Npp8u * pDst,  int nDstStep,
                              NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 8-bit unsigned char image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep,
                                   const Npp8u * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_8u_AC4R(const Npp8u * pSrc, int nSrcStep,
                               const Npp8u * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                   const Npp16u nConstant,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16u_C1R(const Npp16u * pSrc, int nSrcStep,
                               const Npp16u nConstant,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                   const Npp16u * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16u_C3R(const Npp16u * pSrc, int nSrcStep,
                               const Npp16u * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                   const Npp16u * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16u_C4R(const Npp16u * pSrc, int nSrcStep,
                               const Npp16u * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit unsigned short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep,
                                    const Npp16u * pConstants,
                                          Npp8u * pDst,  int nDstStep,
                                    NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16u_AC4R(const Npp16u * pSrc, int nSrcStep,
                                const Npp16u * pConstants,
                                      Npp8u * pDst,  int nDstStep,
                                NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                   const Npp16s nConstant,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16s_C1R(const Npp16s * pSrc, int nSrcStep,
                               const Npp16s nConstant,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                   const Npp16s * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16s_C3R(const Npp16s * pSrc, int nSrcStep,
                               const Npp16s * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                   const Npp16s * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16s_C4R(const Npp16s * pSrc, int nSrcStep,
                               const Npp16s * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 16-bit signed short image compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                    const Npp16s * pConstants,
                                          Npp8u * pDst,  int nDstStep,
                                    NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                const Npp16s * pConstants,
                                      Npp8u * pDst,  int nDstStep,
                                NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 1 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                   const Npp32f nConstant,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                               const Npp32f nConstant,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 3 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                   const Npp32f * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                               const Npp32f * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit floating point image compare with constant value.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                   const Npp32f * pConstants,
                                         Npp8u * pDst,  int nDstStep,
                                   NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_32f_C4R(const Npp32f * pSrc, int nSrcStep,
                               const Npp32f * pConstants,
                                     Npp8u * pDst,  int nDstStep,
                               NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** 
 * 4 channel 32-bit signed floating point compare, not affecting Alpha.
 * Compare pSrc's pixels with constant value. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageWithConstantParameters">Common parameters for nppiCompareC functions</a>.
 *
 */
NppStatus nppiCompareC_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                    const Npp32f * pConstants,
                                          Npp8u * pDst,  int nDstStep,
                                    NppiSize oSizeROI, NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);

NppStatus nppiCompareC_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                const Npp32f * pConstants,
                                      Npp8u * pDst,  int nDstStep,
                                NppiSize oSizeROI, NppCmpOp eComparisonOperation);

/** @} compare_image_with_constant_operations */

/** @defgroup compare_image_differences_with_epsilon_operations Compare Image Differences With Epsilon Operations
 * Compare the pixels value differences of two images with an epsilon value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 *
 * <h3><a name="CommonCompareImageDifferencesWithEpsilonParameters">Common parameters for nppiCompareEqualEps functions include:</a></h3>
 *
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to pixel absolute differences
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferencesWithEpsilonParameters">Common parameters for nppiCompareEqualEps functions</a>.
 *
 */
NppStatus nppiCompareEqualEps_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                          const Npp32f * pSrc2, int nSrc2Step,
                                                Npp8u * pDst,   int nDstStep,
                                          NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEps_32f_C1R(const Npp32f * pSrc1, int nSrc1Step,
                                      const Npp32f * pSrc2, int nSrc2Step,
                                            Npp8u * pDst,   int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 3 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferencesWithEpsilonParameters">Common parameters for nppiCompareEqualEps functions</a>.
 *
 */
NppStatus nppiCompareEqualEps_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                          const Npp32f * pSrc2, int nSrc2Step,
                                                Npp8u * pDst,   int nDstStep,
                                          NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEps_32f_C3R(const Npp32f * pSrc1, int nSrc1Step,
                                      const Npp32f * pSrc2, int nSrc2Step,
                                            Npp8u * pDst,   int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit floating point image compare whether two images are equal within epsilon.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferencesWithEpsilonParameters">Common parameters for nppiCompareEqualEps functions</a>.
 *
 */
NppStatus nppiCompareEqualEps_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                          const Npp32f * pSrc2, int nSrc2Step,
                                                Npp8u * pDst,   int nDstStep,
                                          NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEps_32f_C4R(const Npp32f * pSrc1, int nSrc1Step,
                                      const Npp32f * pSrc2, int nSrc2Step,
                                            Npp8u * pDst,   int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit signed floating point compare whether two images are equal within epsilon, not affecting Alpha.
 * Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferencesWithEpsilonParameters">Common parameters for nppiCompareEqualEps functions</a>.
 *
 */
NppStatus nppiCompareEqualEps_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step,
                                           const Npp32f * pSrc2, int nSrc2Step,
                                                 Npp8u * pDst,   int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEps_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step,
                                       const Npp32f * pSrc2, int nSrc2Step,
                                             Npp8u * pDst,   int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** @} compare_image_differences_with_epsilon_operations */


/** @defgroup compare_image_difference_to_constant_with_epsilon_operations Compare Image Difference With Constant Within Epsilon Operations
 * Compare differences between image pixels and constant within an epsilon value and create a binary result image. In case of multi-channel
 * image types, the condition must be fulfilled for all channels, otherwise the comparison
 * is considered false.
 * The "binary" result image is of type 8u_C1. False is represented by 0, true by NPP_MAX_8U.
 *
 * <h3><a name="CommonCompareImageDifferenceWithConstantWithinEpsilonParameters">Common parameters for nppiCompareEqualEpsC functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param nConstant constant value for single channel functions.
 * \param pConstants pointer to a list of constants, one per color channel for multi-channel image functions.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nEpsilon epsilon tolerance value to compare to per color channel pixel absolute differences
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferenceWithConstantWithinEpsilonParameters">Common parameters for nppiCompareEqualEpsC functions</a>.
 *
 */
NppStatus nppiCompareEqualEpsC_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           const Npp32f nConstant,
                                                 Npp8u * pDst,  int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEpsC_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                       const Npp32f nConstant,
                                             Npp8u * pDst,  int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 3 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferenceWithConstantWithinEpsilonParameters">Common parameters for nppiCompareEqualEpsC functions</a>.
 *
 */
NppStatus nppiCompareEqualEpsC_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           const Npp32f * pConstants,
                                                 Npp8u * pDst,  int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEpsC_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                       const Npp32f * pConstants,
                                             Npp8u * pDst,  int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit floating point image compare whether image and constant are equal within epsilon.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferenceWithConstantWithinEpsilonParameters">Common parameters for nppiCompareEqualEpsC functions</a>.
 *
 */
NppStatus nppiCompareEqualEpsC_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           const Npp32f * pConstants,
                                                 Npp8u * pDst,  int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEpsC_32f_C4R(const Npp32f * pSrc, int nSrcStep,
                                       const Npp32f * pConstants,
                                             Npp8u * pDst,  int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);

/** 
 * 4 channel 32-bit signed floating point compare whether image and constant are equal within epsilon, not affecting Alpha.
 * Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon. 
 * 
 * For common parameter descriptions, see <a href="#CommonCompareImageDifferenceWithConstantWithinEpsilonParameters">Common parameters for nppiCompareEqualEpsC functions</a>.
 *
 */
NppStatus nppiCompareEqualEpsC_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                            const Npp32f * pConstants,
                                                  Npp8u * pDst,  int nDstStep,
                                            NppiSize oSizeROI, Npp32f nEpsilon, NppStreamContext nppStreamCtx);

NppStatus nppiCompareEqualEpsC_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                        const Npp32f * pConstants,
                                              Npp8u * pDst,  int nDstStep,
                                        NppiSize oSizeROI, Npp32f nEpsilon);

/** @} compare_image_difference_to_constant_within_epsilon_operations */

/** @} image_comparison_operations */

/** @} image_threshold_and_compare_operations */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPI_THRESHOLD_AND_COMPARE_OPERATIONS_H */
