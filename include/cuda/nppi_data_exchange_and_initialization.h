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
#ifndef NV_NPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
#define NV_NPPI_DATA_EXCHANGE_AND_INITIALIZATION_H
 
/**
 * \file nppi_data_exchange_and_initialization.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_data_exchange_and_initialization Data Exchange and Initialization
 *  @ingroup nppi
 *
 * Functions for initializing, copying and converting image data.
 *
 * @{
 *
 * These functions can be found in the nppidei library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
 *
 */

/** 
 * @defgroup image_set_operations Image Set Operations
 * Functions for setting all pixels within the ROI to a specific value.
 *
 * @{
 *
 */

/** 
 * @defgroup image_set Set
 * Functions for setting all pixels within the ROI to a specific value.
 *
 * <h3><a name="CommonImageSetParameters">Common parameters for nppiSet functions include:</a></h3>
 *
 * \param nValue The pixel value to be set.
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
 * 8-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8s_C1R_Ctx(const Npp8s nValue, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8s_C1R(const Npp8s nValue, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit two-channel image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8s_C2R_Ctx(const Npp8s aValue[2], Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8s_C2R(const Npp8s aValue[2], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit three-channel image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8s_C3R_Ctx(const Npp8s aValue[3], Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8s_C3R(const Npp8s aValue[3], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit four-channel image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8s_C4R_Ctx(const Npp8s aValue[4], Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8s_C4R(const Npp8s aValue[4], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit four-channel image set ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8s_AC4R_Ctx(const Npp8s aValue[3], Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8s_AC4R(const Npp8s aValue[3], Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C1R_Ctx(const Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C1R(const Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C2R_Ctx(const Npp8u aValue[2], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C2R(const Npp8u aValue[2], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C3R_Ctx(const Npp8u aValue[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C3R(const Npp8u aValue[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C4R_Ctx(const Npp8u aValue[4], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C4R(const Npp8u aValue[4], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_8u_AC4R_Ctx(const Npp8u aValue[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_AC4R(const Npp8u aValue[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C1R_Ctx(const Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C1R(const Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C2R_Ctx(const Npp16u aValue[2], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C2R(const Npp16u aValue[2], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C3R_Ctx(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C3R(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C4R_Ctx(const Npp16u aValue[4], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C4R(const Npp16u aValue[4], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16u_AC4R_Ctx(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_AC4R(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C1R_Ctx(const Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C1R(const Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C2R_Ctx(const Npp16s aValue[2], Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C2R(const Npp16s aValue[2], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C3R_Ctx(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C3R(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C4R_Ctx(const Npp16s aValue[4], Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C4R(const Npp16s aValue[4], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16s_AC4R_Ctx(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_AC4R(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16sc_C1R_Ctx(const Npp16sc oValue, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16sc_C1R(const Npp16sc oValue, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer two-channel image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16sc_C2R_Ctx(const Npp16sc aValue[2], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16sc_C2R(const Npp16sc aValue[2], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer three-channel image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16sc_C3R_Ctx(const Npp16sc aValue[3], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16sc_C3R(const Npp16sc aValue[3], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer four-channel image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16sc_C4R_Ctx(const Npp16sc aValue[4], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16sc_C4R(const Npp16sc aValue[4], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex integer four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16sc_AC4R_Ctx(const Npp16sc aValue[3], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16sc_AC4R(const Npp16sc aValue[3], Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C1R_Ctx(const Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C1R(const Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C2R_Ctx(const Npp32s aValue[2], Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C2R(const Npp32s aValue[2], Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C3R_Ctx(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C3R(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C4R_Ctx(const Npp32s aValue[4], Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C4R(const Npp32s aValue[4], Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32s_AC4R_Ctx(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_AC4R(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32u_C1R_Ctx(const Npp32u nValue, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32u_C1R(const Npp32u nValue, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32u_C2R_Ctx(const Npp32u aValue[2], Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32u_C2R(const Npp32u aValue[2], Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32u_C3R_Ctx(const Npp32u aValue[3], Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32u_C3R(const Npp32u aValue[3], Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32u_C4R_Ctx(const Npp32u aValue[4], Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32u_C4R(const Npp32u aValue[4], Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32u_AC4R_Ctx(const Npp32u aValue[3], Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32u_AC4R(const Npp32u aValue[3], Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32sc_C1R_Ctx(const Npp32sc oValue, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32sc_C1R(const Npp32sc oValue, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32sc_C2R_Ctx(const Npp32sc aValue[2], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32sc_C2R(const Npp32sc aValue[2], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32sc_C3R_Ctx(const Npp32sc aValue[3], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32sc_C3R(const Npp32sc aValue[3], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit complex integer image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32sc_C4R_Ctx(const Npp32sc aValue[4], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32sc_C4R(const Npp32sc aValue[4], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit complex integer four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32sc_AC4R_Ctx(const Npp32sc aValue[3], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32sc_AC4R(const Npp32sc aValue[3], Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16f_C1R_Ctx(const Npp32f nValue, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16f_C1R(const Npp32f nValue, Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16f_C2R_Ctx(const Npp32f aValues[2], Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16f_C2R(const Npp32f aValues[2], Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16f_C3R_Ctx(const Npp32f aValues[3], Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16f_C3R(const Npp32f aValues[3], Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_16f_C4R_Ctx(const Npp32f aValues[4], Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16f_C4R(const Npp32f aValues[4], Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C1R_Ctx(const Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C1R(const Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 2 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C2R_Ctx(const Npp32f aValue[2], Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C2R(const Npp32f aValue[2], Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C3R_Ctx(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C3R(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C4R_Ctx(const Npp32f aValue[4], Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C4R(const Npp32f aValue[4], Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32f_AC4R_Ctx(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_AC4R(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32fc_C1R_Ctx(const Npp32fc oValue, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32fc_C1R(const Npp32fc oValue, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32fc_C2R_Ctx(const Npp32fc aValue[2], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32fc_C2R(const Npp32fc aValue[2], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32fc_C3R_Ctx(const Npp32fc aValue[3], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32fc_C3R(const Npp32fc aValue[3], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit complex image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32fc_C4R_Ctx(const Npp32fc aValue[4], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32fc_C4R(const Npp32fc aValue[4], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit complex four-channel image set ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageSetParameters">Common parameters for nppiSet functions</a>.
 *
 */
NppStatus 
nppiSet_32fc_AC4R_Ctx(const Npp32fc aValue[3], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32fc_AC4R(const Npp32fc aValue[3], Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_set */

/** 
 * @defgroup image_masked_set Masked Set
 * The masked set primitives have an additional "mask image" input. The mask  
 * controls which pixels within the ROI are set. For details see \ref masked_operation.
 *
 * <h3><a name="CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions include:</a></h3>
 *
 * \param nValue The pixel value to be set for single channel functions.
 * \param aValue The pixel-value to be set for multi-channel functions.
 * \param pDst Pointer \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Masked 8-bit unsigned image set. 
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C1MR_Ctx(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C1MR(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C3MR_Ctx(const Npp8u aValue[3], Npp8u* pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C3MR(const Npp8u aValue[3], Npp8u* pDst, int nDstStep, NppiSize oSizeROI,
                const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 8-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C4MR_Ctx(const Npp8u aValue[4], Npp8u* pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C4MR(const Npp8u aValue[4], Npp8u* pDst, int nDstStep, NppiSize oSizeROI,
                const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_8u_AC4MR_Ctx(const Npp8u aValue[3], Npp8u * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_AC4MR(const Npp8u aValue[3], Npp8u * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C1MR_Ctx(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C1MR(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C3MR_Ctx(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C3MR(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 16-bit unsigned image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C4MR_Ctx(const Npp16u aValue[4], Npp16u * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C4MR(const Npp16u aValue[4], Npp16u * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16u_AC4MR_Ctx(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, 
                      NppiSize oSizeROI,
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_AC4MR(const Npp16u aValue[3], Npp16u * pDst, int nDstStep, 
                  NppiSize oSizeROI,
                  const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C1MR_Ctx(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C1MR(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C3MR_Ctx(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);
                          
NppStatus 
nppiSet_16s_C3MR(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 16-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C4MR_Ctx(const Npp16s aValue[4], Npp16s * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);
                          
NppStatus 
nppiSet_16s_C4MR(const Npp16s aValue[4], Npp16s * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_16s_AC4MR_Ctx(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, 
                      NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_AC4MR(const Npp16s aValue[3], Npp16s * pDst, int nDstStep, 
                  NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C1MR_Ctx(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C1MR(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C3MR_Ctx(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);
                          
NppStatus 
nppiSet_32s_C3MR(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 32-bit image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C4MR_Ctx(const Npp32s aValue[4], Npp32s * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);
                          
NppStatus 
nppiSet_32s_C4MR(const Npp32s aValue[4], Npp32s * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 16-bit image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32s_AC4MR_Ctx(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, 
                      NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_AC4MR(const Npp32s aValue[3], Npp32s * pDst, int nDstStep, 
                  NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C1MR_Ctx(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C1MR(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 3 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C3MR_Ctx(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C3MR(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);

/** 
 * Masked 4 channel 32-bit floating point image set.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C4MR_Ctx(const Npp32f aValue[4], Npp32f * pDst, int nDstStep, 
                     NppiSize oSizeROI,
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);
                          
NppStatus 
nppiSet_32f_C4MR(const Npp32f aValue[4], Npp32f * pDst, int nDstStep, 
                 NppiSize oSizeROI,
                 const Npp8u * pMask, int nMaskStep);
                          
/** 
 * Masked 4 channel 32-bit floating point image set method, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedSetParameters">Common parameters for nppiSet_CXM functions</a>.
 *
 */
NppStatus 
nppiSet_32f_AC4MR_Ctx(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, 
                      NppiSize oSizeROI,
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_AC4MR(const Npp32f aValue[3], Npp32f * pDst, int nDstStep, 
                  NppiSize oSizeROI,
                  const Npp8u * pMask, int nMaskStep);

/** @} image_masked_set */

/** 
 * @defgroup image_channel_set Channel Set
 * The selected-channel set primitives set a single color channel in multi-channel images
 * to a given value. The channel is selected by adjusting the pDst pointer to point to 
 * the desired color channel (see \ref channel_of_interest).
 *
 * <h3><a name="CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions include:</a></h3>
 *
 * \param nValue The pixel-value to be set.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 3 channel 8-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C3CR_Ctx(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C3CR(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_8u_C4CR_Ctx(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_8u_C4CR(Npp8u nValue, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 16-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C3CR_Ctx(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C3CR(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_16u_C4CR_Ctx(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16u_C4CR(Npp16u nValue, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 16-bit signed image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C3CR_Ctx(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C3CR(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit signed image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_16s_C4CR_Ctx(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_16s_C4CR(Npp16s nValue, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 32-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C3CR_Ctx(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C3CR(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit unsigned image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_32s_C4CR_Ctx(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32s_C4CR(Npp32s nValue, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 3 channel 32-bit floating point image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C3CR_Ctx(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C3CR(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image set affecting only single channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelSetParameters">Common parameters for nppiSet_CXC functions</a>.
 *
 */
NppStatus 
nppiSet_32f_C4CR_Ctx(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSet_32f_C4CR(Npp32f nValue, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_channel_set */

/** @} image_set_operations */

/** 
 * @defgroup image_copy_operations Image Copy Operations
 * Functions for copying image pixels.
 *
 * @{
 *
 */

/** 
 * @defgroup image_copy Copy
 * Copy pixels from one image to another.
 *
 * <h3><a name="CommonImageCopyParameters">Common parameters for nppiCopy functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
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
 * 8-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8s_C1R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8s_C2R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8s_C2R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8s_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8s_C3R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8s_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8s_C4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit image copy, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8s_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 8-bit unsigned image copy, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit unsigned image copy, not affecting Alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16sc_C1R_Ctx(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16sc_C1R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16sc_C2R_Ctx(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16sc_C2R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16sc_C3R_Ctx(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16sc_C3R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16sc_C4R_Ctx(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16sc_C4R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16sc_AC4R_Ctx(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16sc_AC4R(const Npp16sc * pSrc, int nSrcStep, Npp16sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 32-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32sc_C1R_Ctx(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32sc_C1R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32sc_C2R_Ctx(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32sc_C2R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32sc_C3R_Ctx(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32sc_C3R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32sc_C4R_Ctx(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32sc_C4R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32sc_AC4R_Ctx(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32sc_AC4R(const Npp32sc * pSrc, int nSrcStep, Npp32sc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 16-bit floating point image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 4 channel 32-bit floating point image copy, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32fc_C1R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32fc_C1R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32fc_C2R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32fc_C2R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32fc_C3R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32fc_C3R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit floating-point complex image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32fc_C4R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32fc_C4R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit floating-point complex image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyParameters">Common parameters for nppiCopy functions</a>.
 *
 */
NppStatus 
nppiCopy_32fc_AC4R_Ctx(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32fc_AC4R(const Npp32fc * pSrc, int nSrcStep, Npp32fc * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_copy */

/** 
 * @defgroup image_masked_copy Masked Copy
 * The masked copy primitives have an additional "mask image" input. The mask  
 * controls which pixels within the ROI are copied. For details see \ref masked_operation.
 *
 * <h3><a name="CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * \ref masked_operation 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_C1MR_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_C1MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_C3MR_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_C3MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_C4MR_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                     const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_C4MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                 const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 8-bit unsigned image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_8u_AC4MR_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_8u_AC4MR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_C1MR_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_C1MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_C3MR_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_C3MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_C4MR_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_C4MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit unsigned image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16u_AC4MR_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16u_AC4MR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_C1MR_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_C1MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_C3MR_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_C3MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_C4MR_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_C4MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 16-bit signed image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_16s_AC4MR_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_16s_AC4MR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_C1MR_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_C1MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_C3MR_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_C3MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_C4MR_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_C4MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit signed image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32s_AC4MR_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32s_AC4MR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_C1MR_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_C1MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation three channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_C3MR_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_C3MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_C4MR_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                      const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_C4MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                  const Npp8u * pMask, int nMaskStep);

/** 
 * \ref masked_operation four channel 32-bit float image copy, ignoring alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageMaskedCopyParameters">Common parameters for nppiCopy_CXM functions</a>.
 *
 */
NppStatus 
nppiCopy_32f_AC4MR_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                       const Npp8u * pMask, int nMaskStep, NppStreamContext nppStreamCtx);

NppStatus 
nppiCopy_32f_AC4MR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, 
                   const Npp8u * pMask, int nMaskStep);

/** @} image_masked_copy */


/** 
 * @defgroup image_channel_copy Channel Copy
 * The channel copy primitives copy a single color channel from a multi-channel source image
 * to any other color channel in a multi-channel destination image. The channel is selected 
 * by adjusting the respective image pointers to point to the desired color channel 
 * (see \ref channel_of_interest).
 *
 * <h3><a name="CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions include:</a></h3>
 *
 * \param pSrc \ref select_source_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Selected channel 8-bit unsigned image copy for three-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C3CR_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C3CR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 8-bit unsigned image copy for four-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C4CR_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C4CR(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 16-bit signed image copy for three-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C3CR_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C3CR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 16-bit signed image copy for four-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C4CR_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C4CR(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 16-bit unsigned image copy for three-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C3CR_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C3CR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 16-bit unsigned image copy for four-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C4CR_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C4CR(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 32-bit signed image copy for three-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C3CR_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C3CR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 32-bit signed image copy for four-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C4CR_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C4CR(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 32-bit float image copy for three-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C3CR_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C3CR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Selected channel 32-bit float image copy for four-channel images.
 * 
 * For common parameter descriptions, see <a href="#CommonImageChannelCopyParameters">Common parameters for nppiCopy_CXC functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C4CR_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C4CR(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_channel_copy */

/** 
 * @defgroup image_extract_channel_copy Extract Channel Copy
 * The channel extract primitives copy a single color channel from a multi-channel source image
 * to singl-channel destination image. The channel is selected by adjusting the source image pointer
 * to point to the desired color channel (see \ref channel_of_interest).
 *
 * <h3><a name="CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions include:</a></h3>
 *
 * \param pSrc \ref select_source_pointer.
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
 * Three-channel to single-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C3C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C3C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C4C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C4C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C3C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C3C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C4C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C4C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C3C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C3C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C4C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C4C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C3C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C3C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C4C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C4C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Two-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C2C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C2C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C3C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C3C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel to single-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageExtractChannelCopyParameters">Common parameters for nppiCopy_CXC1 functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C4C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C4C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_extract_channel_copy */

/** 
 * @defgroup image_insert_channel_copy Insert Channel Copy
 * The channel insert primitives copy a single-channel source image into one of the color channels
 * in a multi-channel destination image. The channel is selected by adjusting the destination image pointer
 * to point to the desired color channel (see \ref channel_of_interest).
 *
 * <h3><a name="CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref select_destination_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Single-channel to three-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C1C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C1C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 8-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C1C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C1C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C1C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C1C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 16-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C1C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C1C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C1C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C1C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 16-bit unsigned image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C1C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C1C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C1C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C1C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 32-bit signed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C1C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C1C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to two-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C1C2R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C1C2R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to three-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C1C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C1C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single-channel to four-channel 32-bit float image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageInsertChannelCopyParameters">Common parameters for nppiCopy_C1CX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C1C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C1C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_insert_channel_copy */


/** 
 * @defgroup image_packed_to_planar_channel_copy Packed To Planar Channel Copy
 * Split a packed multi-channel image into multiple single channel planes.
 *
 * E.g. copy the three channels of an RGB image into three separate single-channel
 * images.
 *
 * <h3><a name="CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param aDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Three-channel 8-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C3P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_8u_C4P4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[4], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_C4P4R(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C3P3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C3P3R(const Npp16s * pSrc, int nSrcStep, Npp16s * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_16s_C4P4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * const aDst[4], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_C4P4R(const Npp16s * pSrc, int nSrcStep, Npp16s * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C3P3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C3P3R(const Npp16u * pSrc, int nSrcStep, Npp16u * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit unsigned packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_16u_C4P4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * const aDst[4], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_C4P4R(const Npp16u * pSrc, int nSrcStep, Npp16u * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C3P3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C3P3R(const Npp32s * pSrc, int nSrcStep, Npp32s * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit signed packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_32s_C4P4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * const aDst[4], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_C4P4R(const Npp32s * pSrc, int nSrcStep, Npp32s * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit float packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C3P3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C3P3R(const Npp32f * pSrc, int nSrcStep, Npp32f * const aDst[3], int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit float packed to planar image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePackedToPlanarChannelCopyParameters">Common parameters for nppiCopy_CXPX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_C4P4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * const aDst[4], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_C4P4R(const Npp32f * pSrc, int nSrcStep, Npp32f * const aDst[4], int nDstStep, NppiSize oSizeROI);

/** @} image_packed_to_planar_channel_copy */


/** 
 * @defgroup image_planar_to_packed_channel_copy Planar To Packed Channel Copy
 * Combine multiple image planes into a packed multi-channel image.
 *
 * E.g. copy three single-channel images into a single 3-channel image.
 *
 * <h3><a name="CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions include:</a></h3>
 *
 * \param aSrc Planar \ref source_image_pointer.
 * \param nSrcStep \ref source_planar_image_pointer_array.
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
 * Three-channel 8-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_8u_P3C3R_Ctx(const Npp8u * const aSrc[3], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_P3C3R(const Npp8u * const aSrc[3], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 8-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_8u_P4C4R_Ctx(const Npp8u * const aSrc[4], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_8u_P4C4R(const Npp8u * const aSrc[4], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_16u_P3C3R_Ctx(const Npp16u * const aSrc[3], int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_P3C3R(const Npp16u * const aSrc[3], int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit unsigned planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_16u_P4C4R_Ctx(const Npp16u * const aSrc[4], int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16u_P4C4R(const Npp16u * const aSrc[4], int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 16-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_16s_P3C3R_Ctx(const Npp16s * const aSrc[3], int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_P3C3R(const Npp16s * const aSrc[3], int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 16-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_16s_P4C4R_Ctx(const Npp16s * const aSrc[4], int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_16s_P4C4R(const Npp16s * const aSrc[4], int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_32s_P3C3R_Ctx(const Npp32s * const aSrc[3], int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_P3C3R(const Npp32s * const aSrc[3], int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit signed planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_32s_P4C4R_Ctx(const Npp32s * const aSrc[4], int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32s_P4C4R(const Npp32s * const aSrc[4], int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three-channel 32-bit float planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_P3C3R_Ctx(const Npp32f * const aSrc[3], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_P3C3R(const Npp32f * const aSrc[3], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four-channel 32-bit float planar to packed image copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImagePlanarToPackedChannelCopyParameters">Common parameters for nppiCopy_PXCX functions</a>.
 *
 */
NppStatus
nppiCopy_32f_P4C4R_Ctx(const Npp32f * const aSrc[4], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus
nppiCopy_32f_P4C4R(const Npp32f * const aSrc[4], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_planar_to_packed_channel_copy */

/** 
 * @defgroup image_copy_constant_border Copy Constant Border
 * Methods for copying images and padding borders with a constant, user-specifiable color.
 * 
 * <h3><a name="CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and constant border color (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
 *      destination ROI that will be filled with the constant border color.
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param nValue The pixel value to be set for border pixels for single channel functions.
 * \param aValue Vector of the RGBA values of the border pixels to be set for multi-channel functions.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_8u_C1R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         Npp8u nValue, NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_8u_C1R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     Npp8u nValue);

/**
 * 3 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_8u_C3R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         const Npp8u aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_8u_C3R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Npp8u aValue[3]);

/**
 * 4 channel 8-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_8u_C4R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         const Npp8u aValue[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyConstBorder_8u_C4R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Npp8u aValue[4]);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_8u_AC4R_Ctx(const Npp8u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp8u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp8u aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_8u_AC4R(const Npp8u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp8u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp8u aValue[3]);

/** 
 * 1 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16u_C1R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp16u nValue, NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_16u_C1R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp16u nValue);

/**
 * 3 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16u_C3R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp16u aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_16u_C3R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp16u aValue[3]);

/**
 * 4 channel 16-bit unsigned integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp16u aValue[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyConstBorder_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp16u aValue[4]);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16u_AC4R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Npp16u aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_16u_AC4R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                             Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Npp16u aValue[3]);

/** 
 * 1 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16s_C1R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp16s nValue, NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_16s_C1R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp16s nValue);

/**
 * 3 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16s_C3R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp16s aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_16s_C3R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp16s aValue[3]);

/**
 * 4 channel 16-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16s_C4R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp16s aValue[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyConstBorder_16s_C4R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp16s aValue[4]);
                                       
/**
 * 4 channel 16-bit signed integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_16s_AC4R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Npp16s aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_16s_AC4R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                             Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Npp16s aValue[3]);

/** 
 * 1 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32s_C1R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp32s nValue, NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_32s_C1R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp32s nValue);

/**
 * 3 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32s_C3R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp32s aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_32s_C3R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp32s aValue[3]);

/**
 * 4 channel 32-bit signed integer image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp32s aValue[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyConstBorder_32s_C4R(const Npp32s * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp32s aValue[4]);
                                       
/**
 * 4 channel 32-bit signed integer image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32s_AC4R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Npp32s aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_32s_AC4R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                             Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Npp32s aValue[3]);

/** 
 * 1 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32f_C1R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp32f nValue, NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_32f_C1R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp32f nValue);

/**
 * 3 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp32f aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp32f aValue[3]);

/**
 * 4 channel 32-bit floating point image copy with constant border color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          const Npp32f aValue[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyConstBorder_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      const Npp32f aValue[4]);
                                       
/**
 * 4 channel 32-bit floating point image copy with constant border color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyConstantBorderParameters">Common parameters for nppiCopyConstBorder functions</a>.
 *
 */
NppStatus nppiCopyConstBorder_32f_AC4R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth,
                                           const Npp32f aValue[3], NppStreamContext nppStreamCtx);

NppStatus nppiCopyConstBorder_32f_AC4R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                             Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight, int nLeftBorderWidth,
                                       const Npp32f aValue[3]);

/** @} image_copy_constant_border */

/** 
 * @defgroup image_copy_replicate_border Copy Replicate Border
 * Methods for copying images and padding borders with a replicates of the nearest source image pixel color.
 *
 * <h3><a name="CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and nearest source image pixel color (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
 *      destination ROI that will be filled with the nearest source image pixel color.
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_8u_C1R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                                   Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                             int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_8u_C1R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_8u_C3R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                                   Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                             int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_8u_C3R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_8u_C4R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                                   Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                             int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyReplicateBorder_8u_C4R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_8u_AC4R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_8u_AC4R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16u_C1R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_16u_C1R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16u_C3R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_16u_C3R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit unsigned integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16u_C4R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyReplicateBorder_16u_C4R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit unsigned image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16u_AC4R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                     Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_16u_AC4R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16s_C1R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_16s_C1R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16s_C3R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_16s_C3R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16s_C4R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyReplicateBorder_16s_C4R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_16s_AC4R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                     Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_16s_AC4R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32s_C1R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_32s_C1R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit signed image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32s_C3R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_32s_C3R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit signed integer image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32s_C4R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyReplicateBorder_32s_C4R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 32-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32s_AC4R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                     Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_32s_AC4R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32f_C1R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_32f_C1R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32f_C3R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_32f_C3R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit floating point image copy with nearest source image pixel color.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32f_C4R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                    Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyReplicateBorder_32f_C4R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 32-bit floating point image copy with nearest source image pixel color with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyReplicateBorderParameters">Common parameters for nppiCopyReplicateBorder functions</a>.
 *
 */
NppStatus nppiCopyReplicateBorder_32f_AC4R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                     Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyReplicateBorder_32f_AC4R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                 Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight, int nLeftBorderWidth);

/** @} image_copy_replicate_border */

/** 
 * @defgroup image_copy_wrap_border Copy Wrap Border
 * Methods for copying images and padding borders with wrapped replications of the source image pixel colors.
 *
 * <h3><a name="CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSizeROI Size of the source region of pixels.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).
 * \param nTopBorderHeight Height (in pixels) of the top border. The number of pixel rows at the top of the
 *      destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors.
 *      nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.
 * \param nLeftBorderWidth Width (in pixels) of the left border. The width of the border at the right side of the
 *      destination ROI is implicitly defined by the size of the source ROI:
 *      nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_8u_C1R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                              Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                        int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_8u_C1R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                          Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_8u_C3R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                              Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                        int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_8u_C3R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                          Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_8u_C4R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                              Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                        int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyWrapBorder_8u_C4R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                          Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_8u_AC4R_Ctx(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_8u_AC4R(const Npp8u * pSrc,   int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp8u * pDst,   int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16u_C1R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_16u_C1R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16u_C3R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_16u_C3R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16u_C4R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyWrapBorder_16u_C4R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16u_AC4R_Ctx(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_16u_AC4R(const Npp16u * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16u * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16s_C1R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_16s_C1R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16s_C3R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_16s_C3R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16s_C4R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyWrapBorder_16s_C4R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_16s_AC4R_Ctx(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_16s_AC4R(const Npp16s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp16s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32s_C1R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_32s_C1R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32s_C3R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_32s_C3R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32s_C4R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyWrapBorder_32s_C4R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32s_AC4R_Ctx(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_32s_AC4R(const Npp32s * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32s * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** 
 * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32f_C1R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_32f_C1R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 3 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32f_C3R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_32f_C3R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);

/**
 * 4 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32f_C4R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopyWrapBorder_32f_C4R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                           Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth);
                                       
/**
 * 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopyWrapBorderParameters">Common parameters for nppiCopyWrapBorder functions</a>.
 *
 */
NppStatus nppiCopyWrapBorder_32f_AC4R_Ctx(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                                Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth, NppStreamContext nppStreamCtx);

NppStatus nppiCopyWrapBorder_32f_AC4R(const Npp32f * pSrc,  int nSrcStep, NppiSize oSrcSizeROI,
                                            Npp32f * pDst,  int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth);

/** @} image_copy_wrap_border */

/** 
 * @defgroup image_copy_sub_pixel Copy Sub-Pixel
 * Functions for copying linearly interpolated images using source image subpixel coordinates.
 *
 * <h3><a name="CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image, source image ROI is assumed to be same as destination image ROI.
 * \param nDx Fractional part of source image X coordinate.
 * \param nDy Fractional part of source image Y coordinate.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                          Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                    Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_8u_C1R(const Npp8u * pSrc, int nSrcStep, 
                                      Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                Npp32f nDx, Npp32f nDy);

/**
 * 3 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                          Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                    Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_8u_C3R(const Npp8u * pSrc, int nSrcStep, 
                                      Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                Npp32f nDx, Npp32f nDy);

/**
 * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                          Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                    Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopySubpix_8u_C4R(const Npp8u * pSrc, int nSrcStep, 
                                      Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                Npp32f nDx, Npp32f nDy);
                                       
/**
 * 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                           Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_8u_AC4R(const Npp8u * pSrc, int nSrcStep, 
                                       Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);

/** 
 * 1 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                           Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_16u_C1R(const Npp16u * pSrc, int nSrcStep, 
                                       Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);

/**
 * 3 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                           Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);
 
NppStatus nppiCopySubpix_16u_C3R(const Npp16u * pSrc, int nSrcStep, 
                                       Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);

/**
 * 4 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                           Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopySubpix_16u_C4R(const Npp16u * pSrc, int nSrcStep, 
                                       Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);
                                       
/**
 * 4 channel 16-bit unsigned linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                            Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                      Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_16u_AC4R(const Npp16u * pSrc, int nSrcStep, 
                                        Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                  Npp32f nDx, Npp32f nDy);

/** 
 * 1 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                           Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_16s_C1R(const Npp16s * pSrc, int nSrcStep, 
                                       Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);

/**
 * 3 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                           Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_16s_C3R(const Npp16s * pSrc, int nSrcStep, 
                                       Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);

/**
 * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                           Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopySubpix_16s_C4R(const Npp16s * pSrc, int nSrcStep, 
                                       Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, 
                                 Npp32f nDx, Npp32f nDy);
                                       
/**
 * 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                            Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                        Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                  Npp32f nDx, Npp32f nDy);

/** 
 * 1 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32s_C1R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                           Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_32s_C1R(const Npp32s * pSrc, int nSrcStep,
                                       Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                 Npp32f nDx, Npp32f nDy);

/**
 * 3 channel 32-bit signed linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                           Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_32s_C3R(const Npp32s * pSrc, int nSrcStep,
                                       Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                 Npp32f nDx, Npp32f nDy);

/**
 * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                           Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopySubpix_32s_C4R(const Npp32s * pSrc, int nSrcStep,
                                       Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                 Npp32f nDx, Npp32f nDy);
                                       
/**
 * 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32s_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                            Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_32s_AC4R(const Npp32s * pSrc, int nSrcStep,
                                        Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI,
                                  Npp32f nDx, Npp32f nDy);

/** 
 * 1 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_32f_C1R(const Npp32f * pSrc, int nSrcStep,
                                       Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                 Npp32f nDx, Npp32f nDy);

/**
 * 3 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                       Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                 Npp32f nDx, Npp32f nDy);

/**
 * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                           Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                     Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiCopySubpix_32f_C4R(const Npp32f * pSrc, int nSrcStep,
                                       Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                 Npp32f nDx, Npp32f nDy);
                                       
/**
 * 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected.
 * 
 * For common parameter descriptions, see <a href="#CommonImageCopySubPixelParameters">Common parameters for nppiCopySubpix functions</a>.
 *
 */
NppStatus nppiCopySubpix_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                            Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                      Npp32f nDx, Npp32f nDy, NppStreamContext nppStreamCtx);

NppStatus nppiCopySubpix_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                        Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI,
                                  Npp32f nDx, Npp32f nDy);

/** @} image_copy_sub_pixel */

/** @} image_copy_operations */

/** 
 * @defgroup image_convert Convert Bit Depth
 * Functions for converting bit depth without scaling.
 *
 * @{
 *
 */

/** 
 * @defgroup image_convert_increase Convert To Increased Bit Depth
 * The integer conversion methods do not involve any scaling. Also, even when increasing the bit-depth
 * loss of information may occur:
 * - When converting integers (e.g. Npp32u) to float (e.g. Npp32f) integer value not accurately representable 
 *   by the float are rounded to the closest floating-point value.
 * - When converting signed integers to unsigned integers all negative values are lost (saturated to 0).
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 *  
 * <h3><a name="CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
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
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16u_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16u_C1R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16u_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16u_C3R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16u_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16u_C4R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16u_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16u_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16s_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16s_C1R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16s_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16s_C3R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16s_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16s_C4R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u16s_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u16s_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32s_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32s_C1R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32s_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32s_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32s_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32s_C4R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32s_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32s_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32f_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32f_C1R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32f_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32f_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32f_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32f_C4R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u32f_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u32f_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32s_C1R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32s_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32s_C3R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32s_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32s_C4R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32s_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32f_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32f_C1R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32f_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32f_C3R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32f_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32f_C4R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32f_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32f_AC4R(const Npp8s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32s_C1R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32s_C1R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32s_C3R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32s_C3R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32s_C4R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32s_C4R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32s_AC4R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32s_AC4R(const Npp16u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32f_C1R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32f_C1R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32f_C3R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32f_C3R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32f_C4R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32f_C4R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32f_AC4R_Ctx(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32f_AC4R(const Npp16u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32s_C1R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit signed to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32s_C3R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion.
 * 
 */
NppStatus 
nppiConvert_16s32s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32s_C4R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit signed conversion, not affecting Alpha.
 * 
  * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
*/
NppStatus 
nppiConvert_16s32s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32f_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32f_C1R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32f_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32f_C3R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32f_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32f_C4R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32f_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32f_AC4R(const Npp16s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit floating-point to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16f32f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16f32f_C1R(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit floating-point to 32-bit floating-point conversion. 
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned 
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16f32f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16f32f_C3R(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit floating-point to 32-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16f32f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16f32f_C4R(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit floating-point to 32-bit floating-point conversion, not affecting Alpha.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16f32f_AC4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16f32f_AC4R(const Npp16f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 8-bit unsigned conversion with saturation.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s8u_C1Rs_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s8u_C1Rs(const Npp8s * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s16u_C1Rs_Ctx(const Npp8s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s16u_C1Rs(const Npp8s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s16s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s16s_C1R(const Npp8s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8s32u_C1Rs_Ctx(const Npp8s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8s32u_C1Rs(const Npp8s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 16-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s16u_C1Rs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s16u_C1Rs(const Npp16s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s32u_C1Rs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s32u_C1Rs(const Npp16s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 16-bit unsigned to 32-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u32u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u32u_C1R(const Npp16u * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);


/** 
 * Single channel 32-bit signed to 32-bit unsigned conversion with saturation.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s32u_C1Rs_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s32u_C1Rs(const Npp32s * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit signed to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s32f_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s32f_C1R(const Npp32s * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 32-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToIncreasedBitDepthParameters">Common parameters for nppiConvert to increased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32u32f_C1R_Ctx(const Npp32u * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32u32f_C1R(const Npp32u * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI);

/** @} image_convert_increase */

/** 
 * @defgroup image_convert_decrease Convert To Decreased Bit Depth
 * The integer conversion methods do not involve any scaling. When converting floating-point values
 * to integers the user may choose the most appropriate rounding-mode. Typically information is lost when
 * converting to lower bit depth:
 * - All converted values are saturated to the destination type's range. E.g. any values larger than
 *   the largest value of the destination type are clamped to the destination's maximum.
 * - Converting floating-point values to integer also involves rounding, effectively loosing all
 *   fractional value information in the process.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * <h3><a name="CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eRoundMode \ref rounding_mode_parameter.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u8u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u8u_C1R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u8u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u8u_C3R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u8u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u8u_C4R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u8u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiConvert_16u8u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s8u_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiConvert_16s8u_C1R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s8u_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s8u_C3R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s8u_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s8u_C4R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus
nppiConvert_16s8u_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus
nppiConvert_16s8u_AC4R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s8u_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiConvert_32s8u_C1R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s8u_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s8u_C3R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s8u_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s8u_C4R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus
nppiConvert_32s8u_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus
nppiConvert_32s8u_AC4R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Single channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s8s_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiConvert_32s8s_C1R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Three channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s8s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s8s_C3R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s8s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s8s_C4R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 32-bit signed to 8-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus
nppiConvert_32s8s_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
          
NppStatus
nppiConvert_32s8s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI);
          
/** 
 * Single channel 8-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_8u8s_C1RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_8u8s_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 16-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u8s_C1RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u8s_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 16-bit signed to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16s8s_C1RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16s8s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 16-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_16u16s_C1RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_16u16s_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32u8u_C1RSfs_Ctx(const Npp32u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32u8u_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32u8s_C1RSfs_Ctx(const Npp32u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32u8s_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32u16u_C1RSfs_Ctx(const Npp32u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32u16u_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32u16s_C1RSfs_Ctx(const Npp32u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32u16s_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32u32s_C1RSfs_Ctx(const Npp32u * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32u32s_C1RSfs(const Npp32u * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s16u_C1RSfs_Ctx(const Npp32s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s16u_C1RSfs(const Npp32s * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32s16s_C1RSfs_Ctx(const Npp32s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32s16s_C1RSfs(const Npp32s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8u_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8u_C1R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8u_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8u_C3R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8u_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8u_C4R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8u_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8u_AC4R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8s_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8s_C1R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8s_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8s_C3R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8s_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8s_C4R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 8-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8s_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8s_AC4R(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16u_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16u_C1R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16u_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16u_C3R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16u_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16u_C4R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16u_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16u_AC4R(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16s_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16s_C1R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16s_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16s_C3R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit signed conversion.
 *
  * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
*/
NppStatus 
nppiConvert_32f16s_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16s_C4R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16s_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16s_AC4R(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16f_C1R(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Three channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16f_C3R(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
*/
NppStatus 
nppiConvert_32f16f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16f_C4R(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Four channel 32-bit floating point to 16-bit floating-point conversion.
 *  
 * Note that all pointers and step sizes for images with 16f (Npp16f) data types perform best when they are at least 16 byte aligned. 
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode);

/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8u_C1RSfs_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8u_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 8-bit signed conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f8s_C1RSfs_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f8s_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16u_C1RSfs_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16u_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 16-bit signed conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f16s_C1RSfs_Ctx(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f16s_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 32-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f32u_C1RSfs_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f32u_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp32u * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** 
 * Single channel 32-bit floating point to 32-bit signed conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageConvertToDecreasedBitDepthParameters">Common parameters for nppiConvert to decreased bit depth functions</a>.
 *
 */
NppStatus 
nppiConvert_32f32s_C1RSfs_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiConvert_32f32s_C1RSfs(const Npp32f * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode, int nScaleFactor);

/** @} image_convert_decrease */

/** @} image_convert */

/** 
 * @defgroup image_scale Scale Bit Depth
 * Functions for scaling bit depth up or down.
 *
 * @{
 *
 */

/** 
 * @defgroup image_scale_to_higher_bit_depth Scale To Higher Bit Depth
 * Functions for scaling images to higher bit depth.
 *
 * To map source pixel srcPixelValue to destination pixel dstPixelValue the following equation is used:
 * 
 *      dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)
 *
 * where scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
 *
 * For conversions between integer data types, the entire integer numeric range of the input data type is mapped onto 
 * the entire integer numeric range of the output data type.
 *
 * For conversions to floating point data types the floating point data range is defined by the user supplied floating point values
 * of nMax and nMin which are used as the dstMaxRangeValue and dstMinRangeValue respectively in the scaleFactor and dstPixelValue 
 * calculations and also as the saturation values to which output data is clamped.
 *
 * When converting from floating-point values to integer values, nMax and nMin are used as the srcMaxRangeValue and srcMinRangeValue
 * respectively in the scaleFactor and dstPixelValue calculations. Output values are saturated and clamped to the full output integer
 * pixel value range.
 *
 * <h3><a name="CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nMin specifies the minimum saturation value to which every output value will be clamped.
 * \param nMax specifies the maximum saturation value to which every output value will be clamped.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
 *
 * @{
 *
 */

/** 
 * Single channel 8-bit unsigned to 16-bit unsigned conversion.
 *
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16u_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16u_C1R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16u_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16u_C3R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned  conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16u_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16u_C4R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16u_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16u_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16s_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16s_C1R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16s_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16s_C3R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16s_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16s_C4R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u16s_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u16s_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32s_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32s_C1R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32s_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32s_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32s_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32s_C4R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32s_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32s_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Single channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32f_C1R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32f_C1R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** 
 * Three channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32f_C3R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32f_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32f_C4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32f_C4R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** 
 * Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToHigherBitDepthParameters">Common parameters for nppiScale to higher bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_8u32f_AC4R_Ctx(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_8u32f_AC4R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** @} image_scale_to_higher_bit_depth */

/** 
 * @defgroup image_scale_to_lower_bit_depth Scale To Lower Bit Depth
 * Functions for scaling images to lower bit depth.
 *
 * To map source pixel srcPixelValue to destination pixel dstPixelValue the following equation is used:
 * 
 *      dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue - srcMinRangeValue)
 *
 * where scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue - srcMinRangeValue).
 *
 * For conversions between integer data types, the entire integer numeric range of the input data type is mapped onto 
 * the entire integer numeric range of the output data type.
 *
 * For conversions to floating point data types the floating point data range is defined by the user supplied floating point values
 * of nMax and nMin which are used as the dstMaxRangeValue and dstMinRangeValue respectively in the scaleFactor and dstPixelValue 
 * calculations and also as the saturation values to which output data is clamped.
 *
 * When converting from floating-point values to integer values, nMax and nMin are used as the srcMaxRangeValue and srcMinRangeValue
 * respectively in the scaleFactor and dstPixelValue calculations. Output values are saturated and clamped to the full output integer
 * pixel value range.
 *
 * <h3><a name="CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param hint algorithm performance or accuracy selector, currently ignored
 * \param nMin specifies the minimum saturation value to which every output value will be clamped.
 * \param nMax specifies the maximum saturation value to which every output value will be clamped.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.
 *
 * @{
 *
 */

/** 
 * Single channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16u8u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_16u8u_C1R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Three channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16u8u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_16u8u_C3R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16u8u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_16u8u_C4R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16u8u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiScale_16u8u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);
          
/** 
 * Single channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16s8u_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiScale_16s8u_C1R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);
          
/** 
 * Three channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16s8u_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_16s8u_C3R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_16s8u_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_16s8u_C4R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus
nppiScale_16s8u_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);
          
NppStatus
nppiScale_16s8u_AC4R(const Npp16s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);
          
          
/** 
 * Single channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32s8u_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);
          
NppStatus 
nppiScale_32s8u_C1R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);
          
/** 
 * Three channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32s8u_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_32s8u_C3R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32s8u_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_32s8u_C4R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);

/** 
 * Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus
nppiScale_32s8u_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint, NppStreamContext nppStreamCtx);
          
NppStatus
nppiScale_32s8u_AC4R(const Npp32s * pSrc, int nSrcStep, Npp8u  * pDst, int nDstStep, NppiSize oSizeROI, NppHintAlgorithm hint);
          
/** 
 * Single channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32f8u_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_32f8u_C1R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** 
 * Three channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32f8u_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_32f8u_C3R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32f8u_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_32f8u_C4R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** 
 * Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
 * 
 * For common parameter descriptions, see <a href="#CommonImageScaleToLowerBitDepthParameters">Common parameters for nppiScale to lower bit depth functions</a>.
 *
 */
NppStatus 
nppiScale_32f8u_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax, NppStreamContext nppStreamCtx);

NppStatus 
nppiScale_32f8u_AC4R(const Npp32f * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32f nMin, Npp32f nMax);

/** @} image_scale_to_lower_bit_depth */

/** @} image_scale */

/** 
 * @defgroup image_duplicate_channel Duplicate Channel
 * Functions for duplicating a single channel image in a multiple channel image.
 *
 * <h3><a name="CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Size (width, height) of the destination region, i.e. the region that gets filled with
 *      data from the source image, source image ROI is assumed to be same as destination image ROI.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/** 
 * 1 channel 8-bit unsigned integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_8u_C1C3R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                     Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDup_8u_C1C3R(const Npp8u * pSrc, int nSrcStep, 
                                 Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI);

/**
 * 1 channel 8-bit unsigned integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_8u_C1C4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                     Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiDup_8u_C1C4R(const Npp8u * pSrc, int nSrcStep, 
                                 Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI); 
                                       
/**
 * 1 channel 8-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_8u_C1AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                      Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 

NppStatus nppiDup_8u_C1AC4R(const Npp8u * pSrc, int nSrcStep, 
                                  Npp8u * pDst, int nDstStep, NppiSize oDstSizeROI); 

/** 
 * 1 channel 16-bit unsigned integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_16u_C1C3R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                      Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 

NppStatus nppiDup_16u_C1C3R(const Npp16u * pSrc, int nSrcStep, 
                                  Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI); 

/**
 * 1 channel 16-bit unsigned integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_16u_C1C4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                      Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiDup_16u_C1C4R(const Npp16u * pSrc, int nSrcStep, 
                                  Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI); 
                                       
/**
 * 1 channel 16-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_16u_C1AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                       Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 

NppStatus nppiDup_16u_C1AC4R(const Npp16u * pSrc, int nSrcStep, 
                                   Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI); 

/** 
 * 1 channel 16-bit signed integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_16s_C1C3R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                      Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 

NppStatus nppiDup_16s_C1C3R(const Npp16s * pSrc, int nSrcStep, 
                                  Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI); 

/**
 * 1 channel 16-bit signed integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_16s_C1C4R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                      Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiDup_16s_C1C4R(const Npp16s * pSrc, int nSrcStep, 
                                  Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI); 
                                       
/**
 * 1 channel 16-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_16s_C1AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                       Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDup_16s_C1AC4R(const Npp16s * pSrc, int nSrcStep,
                                   Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI);

/** 
 * 1 channel 32-bit signed integer source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_32s_C1C3R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                      Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDup_32s_C1C3R(const Npp32s * pSrc, int nSrcStep,
                                  Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI);

/**
 * 1 channel 32-bit signed integer source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_32s_C1C4R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                      Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiDup_32s_C1C4R(const Npp32s * pSrc, int nSrcStep,
                                  Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI);
                                       
/**
 * 1 channel 32-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_32s_C1AC4R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                       Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDup_32s_C1AC4R(const Npp32s * pSrc, int nSrcStep,
                                   Npp32s * pDst, int nDstStep, NppiSize oDstSizeROI);

/** 
 * 1 channel 32-bit floating point source image duplicated in all 3 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_32f_C1C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDup_32f_C1C3R(const Npp32f * pSrc, int nSrcStep,
                                  Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI);

/**
 * 1 channel 32-bit floating point source image duplicated in all 4 channels of destination image.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_32f_C1C4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                      Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);
                                       
NppStatus nppiDup_32f_C1C4R(const Npp32f * pSrc, int nSrcStep,
                                  Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI);
                                       
/**
 * 1 channel 32-bit floating point source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
 *
 * For common parameter descriptions, see <a href="#CommonImageDuplicateChannelParameters">Common parameters for nppiDup functions</a>.
 *
 */
NppStatus nppiDup_32f_C1AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                       Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDup_32f_C1AC4R(const Npp32f * pSrc, int nSrcStep,
                                   Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI);

/** @} image_duplicate_channel */


/** 
 * @defgroup image_transpose Transpose 
 * Functions for transposing images of various types. Like matrix transpose, image transpose is a mirror along the image's
 * diagonal (upper-left to lower-right corner).
 * 
 * <h3><a name="CommonImageTransposeParameters">Common parameters for nppiTranspose functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst Pointer to the destination ROI.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSrcROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 3 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 4 channel 8-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 1 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 3 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 4 channel 16-bit unsigned int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 1 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 3 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 4 channel 16-bit signed int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 1 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_32s_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 3 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_32s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_32s_C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 4 channel 32-bit signed int image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 1 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 3 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSrcROI);

/**
 * 4 channel 32-bit floating point image transpose.
 *
 * For common parameter descriptions, see <a href="#CommonImageTransposeParameters">Common parameters for nppiTranspose functions</a>.
 *
 */
NppStatus 
nppiTranspose_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSrcROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiTranspose_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSrcROI);

/** @} image_transpose */

/** 
 * @defgroup image_swap_channels Swap Channels
 * Functions for swapping and duplicating channels in multiple channel images. 
 * The methods support arbitrary permutations of the original channels, including replication and
 * setting one or more channels to a constant value.
 *
 * @{
 *
 */

/** 
 * 3 channel 8-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                            Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_8u_C3R(const Npp8u * pSrc, int nSrcStep, 
                                        Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 8-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 8-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C4C3R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                              Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_8u_C4C3R(const Npp8u * pSrc, int nSrcStep, 
                                          Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 8-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                            Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiSwapChannels_8u_C4R(const Npp8u * pSrc, int nSrcStep, 
                                        Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4]); 
                                       
/** 
 * 4 channel 8-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_8u_C4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 8-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_C3C4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                              Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp8u nValue, NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_8u_C3C4R(const Npp8u * pSrc, int nSrcStep, 
                                          Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp8u nValue);

/**
 * 4 channel 8-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, 
                                             Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx); 

NppStatus nppiSwapChannels_8u_AC4R(const Npp8u * pSrc, int nSrcStep, 
                                         Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                             Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx); 

NppStatus nppiSwapChannels_16u_C3R(const Npp16u * pSrc, int nSrcStep, 
                                         Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 16-bit unsigned integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_C4C3R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                               Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16u_C4C3R(const Npp16u * pSrc, int nSrcStep, 
                                           Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 16-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                             Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiSwapChannels_16u_C4R(const Npp16u * pSrc, int nSrcStep, 
                                         Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4]); 
                                       
/** 
 * 4 channel 16-bit unsigned integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_C4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16u_C4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 16-bit unsigned integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_C3C4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                               Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp16u nValue, NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16u_C3C4R(const Npp16u * pSrc, int nSrcStep, 
                                           Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp16u nValue);

/**
 * 4 channel 16-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, 
                                              Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx); 

NppStatus nppiSwapChannels_16u_AC4R(const Npp16u * pSrc, int nSrcStep, 
                                          Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                             Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx); 

NppStatus nppiSwapChannels_16s_C3R(const Npp16s * pSrc, int nSrcStep, 
                                         Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]); 

/** 
 * 3 channel 16-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 16-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_C4C3R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                               Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16s_C4C3R(const Npp16s * pSrc, int nSrcStep, 
                                           Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 16-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                             Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx); 
                                       
NppStatus nppiSwapChannels_16s_C4R(const Npp16s * pSrc, int nSrcStep, 
                                         Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4]); 
                                       
/** 
 * 4 channel 16-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_C4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16s_C4IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 16-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_C3C4R_Ctx(const Npp16s * pSrc, int nSrcStep, 
                                               Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp16s nValue, NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16s_C3C4R(const Npp16s * pSrc, int nSrcStep, 
                                           Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp16s nValue);

/**
 * 4 channel 16-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep,
                                              Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_16s_AC4R(const Npp16s * pSrc, int nSrcStep,
                                          Npp16s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                             Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32s_C3R(const Npp32s * pSrc, int nSrcStep,
                                         Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_C3IR_Ctx(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32s_C3IR(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 32-bit signed integer source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_C4C3R_Ctx(const Npp32s * pSrc, int nSrcStep, 
                                               Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32s_C4C3R(const Npp32s * pSrc, int nSrcStep, 
                                           Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 32-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                             Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiSwapChannels_32s_C4R(const Npp32s * pSrc, int nSrcStep,
                                         Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4]);
                                       
/** 
 * 4 channel 32-bit signed integer in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_C4IR_Ctx(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32s_C4IR(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 32-bit signed integer source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_C3C4R_Ctx(const Npp32s * pSrc, int nSrcStep, 
                                               Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp32s nValue, NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32s_C3C4R(const Npp32s * pSrc, int nSrcStep, 
                                           Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp32s nValue);

/**
 * 4 channel 32-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32s_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep,
                                              Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32s_AC4R(const Npp32s * pSrc, int nSrcStep,
                                          Npp32s * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit floating point source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32f_C3R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 3 channel 32-bit floating point in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** 
 * 4 channel 32-bit floating point source image to 3 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_C4C3R_Ctx(const Npp32f * pSrc, int nSrcStep, 
                                               Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32f_C4C3R(const Npp32f * pSrc, int nSrcStep, 
                                           Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/**
 * 4 channel 32-bit floating point source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                             Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);
                                       
NppStatus nppiSwapChannels_32f_C4R(const Npp32f * pSrc, int nSrcStep,
                                         Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4]);
                                       
/** 
 * 4 channel 32-bit floating point in place image.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA
 *      channel order.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_C4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32f_C4IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[4]);

/** 
 * 3 channel 32-bit floating point source image to 4 channel destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR
 *      channel order.
 * \param nValue (V) Single channel constant value that can be replicated in one or more of the 4 destination channels.
 *      nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination
 *      channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that
 *      particular destination channel value unmodified.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_C3C4R_Ctx(const Npp32f * pSrc, int nSrcStep, 
                                               Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp32f nValue, NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32f_C3C4R(const Npp32f * pSrc, int nSrcStep, 
                                           Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[4], const Npp32f nValue);

/**
 * 4 channel 32-bit floating point source image to 4 channel destination image with destination alpha channel unaffected.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param aDstOrder Host memory integer array describing how channel values are permutated. The n-th entry
 *      of the array contains the number of the channel that is stored in the n-th channel of
 *      the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA
 *      channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSwapChannels_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep,
                                              Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);

NppStatus nppiSwapChannels_32f_AC4R(const Npp32f * pSrc, int nSrcStep,
                                          Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3]);

/** @} image_swap_channels */

/** @} image_data_exchange_and_initialization */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPI_DATA_EXCHANGE_AND_INITIALIZATION_H */
