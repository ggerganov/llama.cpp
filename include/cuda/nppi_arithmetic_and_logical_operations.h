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
#ifndef NV_NPPI_ARITHMETIC_AND_LOGICAL_OPERATIONS_H
#define NV_NPPI_ARITHMETIC_AND_LOGICAL_OPERATIONS_H
 
/**
 * \file nppi_arithmetic_and_logical_operations.h
 * Image Arithmetic and Logical Operations.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 * @defgroup image_arithmetic_and_logical_operations Arithmetic and Logical Operations
 * @ingroup nppi
 * @{
 *
 * These functions can be found in the nppial library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
 */

/** 
 * @defgroup image_arithmetic_operations Arithmetic Operations
 * The set of image processing arithmetic operations available in the library.
 * @{
 */

/** 
 * @defgroup image_addc AddC
 *
 * Adds a constant value to each pixel of an image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling. 
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus 
nppiAddC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel..
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[4], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus 
nppiAddC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[4], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus 
nppiAddC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[4], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[4], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[4], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C1IRSfs_Ctx(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                                Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                            Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C1IRSfs_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s aConstants[3], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C1IRSfs_Ctx(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                                Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant 32-bit floating point constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image add constant.
 * \param nConstant 32-bit floating point constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16f_C1IR_Ctx(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16f_C1IR(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image add constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16f_C3IR(const Npp32f aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[4], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[4], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image add constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_16f_C4IR(const Npp32f aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image add constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAddC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image add constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_C3IR(const Npp32f aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                            Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image add constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_AC4IR(const Npp32f aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[4], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[4], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image add constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32f_C4IR(const Npp32f aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C1IR_Ctx(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc aConstants[3], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAddC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc aConstants[3], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C3IR_Ctx(const Npp32fc aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_C3IR(const Npp32fc aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc aConstants[3], 
                             Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc aConstants[3], 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image add constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_AC4IR_Ctx(const Npp32fc aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_AC4IR(const Npp32fc aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc aConstants[4], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc aConstants[4], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddC_32fc_C4IR_Ctx(const Npp32fc aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddC_32fc_C4IR(const Npp32fc aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_addc */ 


/** 
 * @defgroup image_mulc MulC
 *
 * Multiplies each pixel of an image by a constant value.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[4], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u aConstants[4], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[4], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[3], 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[4], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s aConstants[4], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C1IRSfs_Ctx(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                                Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc aConstants[3], 
                            Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C1IRSfs_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s aConstants[3], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C1IRSfs_Ctx(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc aConstants[3], 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant 32-bit floating point constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image multiply by constant.
 * \param nConstant 32-bit floating point constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16f_C1IR_Ctx(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16f_C1IR(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image multiply by constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16f_C3IR(const Npp32f aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[4], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f aConstants[4], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image multiply by constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_16f_C4IR(const Npp32f aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image multiply by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f aConstants[3], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image multiply by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_C3IR(const Npp32f aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                            Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image multiply by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_AC4IR_Ctx(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_AC4IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image multiply by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32f_C4IR_Ctx(const Npp32f  aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32f_C4IR(const Npp32f  aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C1IR_Ctx(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C3IR_Ctx(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_C3IR(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                             Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image multiply by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_AC4IR_Ctx(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_AC4IR(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[4], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[4], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulC_32fc_C4IR_Ctx(const Npp32fc  aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulC_32fc_C4IR(const Npp32fc  aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_mulc */ 

/** 
 * @defgroup image_mulcscale MulCScale
 *
 * Multiplies each pixel of an image by a constant value then scales the result
 * by the maximum value for the data bit width.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                               Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C1IR_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                               Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant and scale by max bit width value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C3IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_C3IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                                Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale and scale by max bit width value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_AC4IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_AC4IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                               Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_8u_C4IR_Ctx(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_8u_C4IR(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                                Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C1IR_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                                Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C3IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_C3IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                                 Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant and scale by max bit width value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_AC4IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_AC4IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                                Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulCScale_16u_C4IR_Ctx(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulCScale_16u_C4IR(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_mulcscale */ 

/** @defgroup image_subc SubC
 * Subtracts a constant value from each pixel of an image.
 * @{
 */

/** 
 * One 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C3IRSfs_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_C3IRSfs(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_AC4IRSfs_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_AC4IRSfs(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_8u_C4IRSfs_Ctx(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_8u_C4IRSfs(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C3IRSfs_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_C3IRSfs(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_AC4IRSfs_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_AC4IRSfs(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16u_C4IRSfs_Ctx(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16u_C4IRSfs(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C3IRSfs_Ctx(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_C3IRSfs(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_AC4IRSfs_Ctx(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_AC4IRSfs(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[4], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[4], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16s_C4IRSfs_Ctx(const Npp16s  aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16s_C4IRSfs(const Npp16s  aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C1IRSfs_Ctx(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_C3IRSfs_Ctx(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16sc_C3IRSfs(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                                Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                            Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16sc_AC4IRSfs_Ctx(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16sc_AC4IRSfs(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C1IRSfs_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32s_C3IRSfs_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32s_C3IRSfs(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C1IRSfs_Ctx(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_C3IRSfs_Ctx(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32sc_C3IRSfs(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                                Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32sc_AC4IRSfs_Ctx(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32sc_AC4IRSfs(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant 32-bit floating point constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image subtract constant.
 * \param nConstant 32-bit floating point constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16f_C1IR_Ctx(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16f_C1IR(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image subtract constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16f_C3IR_Ctx(const Npp32f  aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
 
NppStatus 
nppiSubC_16f_C3IR(const Npp32f  aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
 
/** 
 * Four 16-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image subtract constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_16f_C4IR_Ctx(const Npp32f  aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_16f_C4IR(const Npp32f  aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image subtract constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image subtract constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C3IR_Ctx(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_C3IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                            Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image subtract constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_AC4IR_Ctx(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_AC4IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image subtract constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32f_C4IR_Ctx(const Npp32f  aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32f_C4IR(const Npp32f  aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C1IR_Ctx(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C3IR_Ctx(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_C3IR(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                             Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image subtract constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_AC4IR_Ctx(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_AC4IR(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[4], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[4], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSubC_32fc_C4IR_Ctx(const Npp32fc  aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSubC_32fc_C4IR(const Npp32fc  aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_subc */ 

/** 
 * @defgroup image_divc DivC
 *
 * Divides each pixel of an image by a constant value.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C1IRSfs_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_C1IRSfs(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel 8-bit unsigned char in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C3IRSfs_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_C3IRSfs(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_AC4IRSfs_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_AC4IRSfs(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_8u_C4IRSfs_Ctx(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_8u_C4IRSfs(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C1IRSfs_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_C1IRSfs(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C3IRSfs_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_C3IRSfs(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_AC4IRSfs_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_AC4IRSfs(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16u_C4IRSfs_Ctx(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16u_C4IRSfs(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C1IRSfs_Ctx(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_C1IRSfs(const Npp16s nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C3IRSfs_Ctx(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_C3IRSfs(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[3], 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_AC4IRSfs_Ctx(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_AC4IRSfs(const Npp16s  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[4], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s  aConstants[4], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16s_C4IRSfs_Ctx(const Npp16s  aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16s_C4IRSfs(const Npp16s  aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc nConstant, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C1IRSfs_Ctx(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16sc_C1IRSfs(const Npp16sc nConstant, Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_C3IRSfs_Ctx(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16sc_C3IRSfs(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                                Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc  aConstants[3], 
                            Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16sc_AC4IRSfs_Ctx(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16sc_AC4IRSfs(const Npp16sc  aConstants[3], Npp16sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C1IRSfs_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32s_C1IRSfs(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32s_C3IRSfs_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32s_C3IRSfs(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc nConstant, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C1IRSfs_Ctx(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32sc_C1IRSfs(const Npp32sc nConstant, Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_C3IRSfs_Ctx(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32sc_C3IRSfs(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                                Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc  aConstants[3], 
                            Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32sc_AC4IRSfs_Ctx(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32sc_AC4IRSfs(const Npp32sc  aConstants[3], Npp32sc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant 32-bit floating point constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image divided by constant.
 * \param nConstant 32-bit floating point constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16f_C1IR_Ctx(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16f_C1IR(const Npp32f nConstant, Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image divided by constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16f_C3IR_Ctx(const Npp32f  aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16f_C3IR(const Npp32f  aConstants[3], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                           Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                       Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image divided by constant.
 * \param aConstants fixed size array of 32-bit floating point constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_16f_C4IR_Ctx(const Npp32f  aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_16f_C4IR(const Npp32f  aConstants[4], Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image divided by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C1IR_Ctx(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_C1IR(const Npp32f nConstant, Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image divided by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C3IR_Ctx(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_C3IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                            Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[3], 
                        Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image divided by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_AC4IR_Ctx(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_AC4IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f  aConstants[4], 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image divided by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32f_C4IR_Ctx(const Npp32f  aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32f_C4IR(const Npp32f  aConstants[4], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc nConstant, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C1IR_Ctx(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_C1IR(const Npp32fc nConstant, Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C3IR_Ctx(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_C3IR(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                             Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[3], 
                         Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image divided by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_AC4IR_Ctx(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_AC4IR(const Npp32fc  aConstants[3], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[4], 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc  aConstants[4], 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDivC_32fc_C4IR_Ctx(const Npp32fc  aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDivC_32fc_C4IR(const Npp32fc  aConstants[4], Npp32fc * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_divc */ 

/** 
 * @defgroup image_absdiffc AbsDiffC
 *
 * Determines absolute difference between each pixel of an image and a constant value.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image absolute difference with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiffC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, Npp8u nConstant, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiffC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, Npp8u nConstant);

/** 
 * One 16-bit unsigned short channel image absolute difference with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiffC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, Npp16u nConstant, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiffC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, Npp16u nConstant);

/** 
 * One 32-bit floating point channel image absolute difference with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiffC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, Npp32f nConstant, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiffC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, Npp32f nConstant);

/** @} image_absdiffc */ 

/** 
 * @defgroup image_add Add
 *
 * Pixel by pixel addition of two images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C1IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C3IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_AC4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                              Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_8u_C4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C1IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C3IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_AC4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                               Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16u_C4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C1IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C3IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_AC4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                               Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16s_C4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C1IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_C3IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16sc_AC4IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                                Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/**
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
 * 32-bit image add.
 * Add the pixel values of corresponding pixels in the ROI and write them to the output image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiAdd_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, 
                              const Npp32s * pSrc2, int nSrc2Step, 
                                    Npp32s * pDst,  int nDstStep, 
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx);                        

NppStatus nppiAdd_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI);                        

/** 
 * One 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C1IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32s_C3IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C1IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_C3IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32sc_AC4IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                                Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16f_C1IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16f_C1IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16f_C3IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16f_C3IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_16f_C4IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_16f_C4IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C1IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C3IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_AC4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                            Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32f_C4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C1IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C3IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_AC4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                             Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAdd_32fc_C4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAdd_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_add */ 

/** 
 * @defgroup image_addsquare AddSquare
 *
 * Pixel by pixel addition of squared pixels from source image to floating point
 * pixel values of destination image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_8u32f_C1IMR_Ctx(const Npp8u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                    Npp32f * pSrcDst, int nSrcDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddSquare_8u32f_C1IMR(const Npp8u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                Npp32f * pSrcDst, int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel image squared then added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_8u32f_C1IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                   Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddSquare_8u32f_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                               Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_16u32f_C1IMR_Ctx(const Npp16u * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                     Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddSquare_16u32f_C1IMR(const Npp16u * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                 Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image squared then added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_16u32f_C1IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                    Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddSquare_16u32f_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                                Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_32f_C1IMR_Ctx(const Npp32f * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                  Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddSquare_32f_C1IMR(const Npp32f * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                              Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image squared then added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddSquare_32f_C1IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                                 Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddSquare_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                             Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_addsquare */ 

/** 
 * @defgroup image_addproduct AddProduct
 * Pixel by pixel addition of product of pixels from two source images to
 * floating point pixel values of destination image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_8u32f_C1IMR_Ctx(const Npp8u * pSrc1,  int nSrc1Step,    const Npp8u * pSrc2, int nSrc2Step,
                               const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,    int nSrcDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_8u32f_C1IMR(const Npp8u * pSrc1,  int nSrc1Step,    const Npp8u * pSrc2, int nSrc2Step,
                           const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,    int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_8u32f_C1IR_Ctx(const Npp8u * pSrc1,    int nSrc1Step,   const Npp8u * pSrc2, int nSrc2Step,
                                    Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_8u32f_C1IR(const Npp8u * pSrc1,    int nSrc1Step,   const Npp8u * pSrc2, int nSrc2Step,
                                Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_16u32f_C1IMR_Ctx(const Npp16u * pSrc1, int nSrc1Step,    const Npp16u * pSrc2, int nSrc2Step,
                                const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,     int nSrcDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_16u32f_C1IMR(const Npp16u * pSrc1, int nSrc1Step,    const Npp16u * pSrc2, int nSrc2Step,
                            const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,     int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_16u32f_C1IR_Ctx(const Npp16u * pSrc1,    int nSrc1Step,   const Npp16u * pSrc2, int nSrc2Step,
                                     Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_16u32f_C1IR(const Npp16u * pSrc1,    int nSrc1Step,   const Npp16u * pSrc2, int nSrc2Step,
                                 Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_32f_C1IMR_Ctx(const Npp32f * pSrc1, int nSrc1Step,    const Npp32f * pSrc2, int nSrc2Step,
                             const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,     int nSrcDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_32f_C1IMR(const Npp32f * pSrc1, int nSrc1Step,    const Npp32f * pSrc2, int nSrc2Step,
                         const Npp8u  * pMask, int nMaskStep,    Npp32f * pSrcDst,     int nSrcDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_32f_C1IR_Ctx(const Npp32f * pSrc1,    int nSrc1Step,   const Npp32f * pSrc2, int nSrc2Step,
                                  Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_32f_C1IR(const Npp32f * pSrc1,    int nSrc1Step,   const Npp32f * pSrc2, int nSrc2Step,
                              Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel image product added to in place floating point destination image.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddProduct_16f_C1IR_Ctx(const Npp16f * pSrc1,    int nSrc1Step,   const Npp16f * pSrc2, int nSrc2Step,
                                  Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddProduct_16f_C1IR(const Npp16f * pSrc1,    int nSrc1Step,   const Npp16f * pSrc2, int nSrc2Step,
                              Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_addproduct */ 

/** 
 * @defgroup image_addweighted AddWeighted
 * Pixel by pixel addition of alpha weighted pixel values from a source image to
 * floating point pixel values of destination image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_8u32f_C1IMR_Ctx(const Npp8u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                      Npp32f * pSrcDst, int nSrcDstStep,  NppiSize oSizeROI,   Npp32f nAlpha, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddWeighted_8u32f_C1IMR(const Npp8u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                  Npp32f * pSrcDst, int nSrcDstStep,  NppiSize oSizeROI,   Npp32f nAlpha);

/** 
 * One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_8u32f_C1IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                     Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddWeighted_8u32f_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                                 Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** 
 * One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_16u32f_C1IMR_Ctx(const Npp16u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                       Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI,   Npp32f nAlpha, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddWeighted_16u32f_C1IMR(const Npp16u * pSrc,     int nSrcStep,     const Npp8u * pMask, int nMaskStep, 
                                   Npp32f * pSrcDst,  int nSrcDstStep,  NppiSize oSizeROI,   Npp32f nAlpha);

/** 
 * One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_16u32f_C1IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                      Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddWeighted_16u32f_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                                  Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** 
 * One 32-bit floating point channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pMask \ref mask_image_pointer.
 * \param nMaskStep \ref mask_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_32f_C1IMR_Ctx(const Npp32f * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                    Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddWeighted_32f_C1IMR(const Npp32f * pSrc,     int nSrcStep, const Npp8u * pMask, int nMaskStep, 
                                Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** 
 * One 32-bit floating point channel alpha weighted image added to in place floating point destination image.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nAlpha Alpha weight to be applied to source image pixels (0.0F to 1.0F)
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAddWeighted_32f_C1IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                                   Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha, NppStreamContext nppStreamCtx);

NppStatus 
nppiAddWeighted_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                               Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, Npp32f nAlpha);

/** @} image_addweighted */ 

/** 
 * @defgroup image_mul Mul
 *
 * Pixel by pixel multiply of two images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C1IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C3IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_AC4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                              Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_8u_C4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C1IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C3IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_AC4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                               Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16u_C4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C1IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C3IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_AC4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                               Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16s_C4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C1IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_C3IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16sc_AC4IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                                Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead.
 * 1 channel 32-bit image multiplication.
 * Multiply corresponding pixels in ROI. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiMul_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, 
                              const Npp32s * pSrc2, int nSrc2Step, 
                                    Npp32s * pDst,  int nDstStep, 
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx); 

NppStatus nppiMul_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI); 

/** 
 * One 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C1IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32s_C3IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C1IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_C3IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32sc_AC4IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                                Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16f_C1IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16f_C1IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16f_C3IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16f_C3IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_16f_C4IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_16f_C4IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C1IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C3IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_AC4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                            Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32f_C4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C1IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C3IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_AC4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                             Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMul_32fc_C4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMul_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_mul */ 

/** 
 * @defgroup image_mulscale MulScale
 *
 * Pixel by pixel multiplies each pixel of two images then scales the result by
 * the maximum value for the data bit width.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C1IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C3IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                               Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_AC4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                            Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_8u_C4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C1IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C3IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_AC4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                 Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                             Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiMulScale_16u_C4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiMulScale_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_mulscale */ 

/** 
 * @defgroup image_sub Sub
 *
 * Pixel by pixel subtraction of two images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C1IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C3IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_AC4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                              Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_8u_C4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);
NppStatus 
nppiSub_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C1IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C3IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_AC4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                               Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16u_C4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C1IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C3IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_AC4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                               Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16s_C4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C1IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_C3IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16sc_AC4IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                                Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/**
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
 * 32-bit image subtraction.
 * Subtract pSrc1's pixels from corresponding pixels in pSrc2. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiSub_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, 
                              const Npp32s * pSrc2, int nSrc2Step, 
                                    Npp32s * pDst,  int nDstStep, 
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiSub_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C1IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C3IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C4RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32s_C4RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32s_C4IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32s_C4IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C1IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_C3IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32sc_AC4IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                                Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16f_C1IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16f_C1IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16f_C3IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16f_C3IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_16f_C4IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_16f_C4IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C1IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C3IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_AC4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                            Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiSub_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32f_C4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C1IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C3IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_AC4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                             Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSub_32fc_C4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSub_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_sub */ 

/** 
 * @defgroup image_div Div
 *
 * Pixel by pixel division of two images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C1IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C3IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_AC4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                              Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_8u_C4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                             Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C1IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C3IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_AC4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                               Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16u_C4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                              Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C1IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C3IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_AC4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                               Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                           Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                         Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16s_C4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                              Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                          Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C1RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16sc_C1RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C1IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16sc_C1IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C3RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                              Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16sc_C3RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                          Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_C3IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                               Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16sc_C3IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                           Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_AC4RSfs_Ctx(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                               Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16sc_AC4RSfs(const Npp16sc * pSrc1, int nSrc1Step, const Npp16sc * pSrc2, int nSrc2Step, 
                           Npp16sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16sc_AC4IRSfs_Ctx(const Npp16sc * pSrc,     int nSrcStep, 
                                Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16sc_AC4IRSfs(const Npp16sc * pSrc,     int nSrcStep, 
                            Npp16sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C1RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32s_C1RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/**
 * Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 
 * 32-bit image division.
 * Divide pixels in pSrc2 by pSrc1's pixels. 
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiDiv_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, 
                              const Npp32s * pSrc2, int nSrc2Step, 
                                    Npp32s * pDst,  int nDstStep, 
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiDiv_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, 
                          const Npp32s * pSrc2, int nSrc2Step, 
                                Npp32s * pDst,  int nDstStep, 
                                NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C1IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32s_C1IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C3RSfs_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32s_C3RSfs(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32s_C3IRSfs_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                              Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32s_C3IRSfs(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C1RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32sc_C1RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C1IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32sc_C1IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C3RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                              Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32sc_C3RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                          Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_C3IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                               Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32sc_C3IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                           Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_AC4RSfs_Ctx(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                               Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32sc_AC4RSfs(const Npp32sc * pSrc1, int nSrc1Step, const Npp32sc * pSrc2, int nSrc2Step, 
                           Npp32sc * pDst,  int nDstStep,  NppiSize oSizeROI,   int nScaleFactor);

/** 
 * Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32sc_AC4IRSfs_Ctx(const Npp32sc * pSrc,     int nSrcStep, 
                                Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32sc_AC4IRSfs(const Npp32sc * pSrc,     int nSrcStep, 
                            Npp32sc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16f_C1IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16f_C1IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16f_C3R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16f_C3R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16f_C3IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16f_C3IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16f_C4R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                          Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16f_C4R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, 
                      Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_16f_C4IR_Ctx(const Npp16f * pSrc,     int nSrcStep, 
                           Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_16f_C4IR(const Npp16f * pSrc,     int nSrcStep, 
                       Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C1IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_C1IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C3R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_C3R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C3IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_C3IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                           Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                       Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel with unmodified alpha in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_AC4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                            Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_AC4IR(const Npp32f * pSrc,     int nSrcStep, 
                        Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                          Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_C4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                      Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32f_C4IR_Ctx(const Npp32f * pSrc,     int nSrcStep, 
                           Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32f_C4IR(const Npp32f * pSrc,     int nSrcStep, 
                       Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C1R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_C1R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C1IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_C1IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C3R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_C3R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C3IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_C3IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_AC4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                            Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_AC4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                        Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_AC4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                             Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_AC4IR(const Npp32fc * pSrc,     int nSrcStep, 
                         Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C4R_Ctx(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                           Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_C4R(const Npp32fc * pSrc1, int nSrc1Step, const Npp32fc * pSrc2, int nSrc2Step, 
                       Npp32fc * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_32fc_C4IR_Ctx(const Npp32fc * pSrc,     int nSrcStep, 
                            Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_32fc_C4IR(const Npp32fc * pSrc,     int nSrcStep, 
                        Npp32fc * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_div */ 

/** 
 * @defgroup image_divround Div_Round
 *
 * Pixel by pixel division of two images using result rounding modes.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C1RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                                  Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_C1RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C1IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                   Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_C1IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C3RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                                  Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_C3RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C3IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                   Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);
                               
NppStatus 
nppiDiv_Round_8u_C3IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);
                               
/** 
 * Four 8-bit unsigned char channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_AC4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                                   Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_AC4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                               Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_AC4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                    Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_AC4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                                Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C4RSfs_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                                  Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_C4RSfs(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_8u_C4IRSfs_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                                   Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_8u_C4IRSfs(const Npp8u * pSrc,     int nSrcStep, 
                               Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C1RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                   Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_C1RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C1IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                    Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_C1IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C3RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                   Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_C3RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C3IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                    Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);
                               
NppStatus 
nppiDiv_Round_16u_C3IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);
                               
/** 
 * Four 16-bit unsigned short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_AC4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                    Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_AC4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_AC4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                     Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_AC4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                 Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C4RSfs_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                   Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_C4RSfs(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16u_C4IRSfs_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                                    Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16u_C4IRSfs(const Npp16u * pSrc,     int nSrcStep, 
                                Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C1RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                   Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_C1RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppRoundMode rndMode, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C1IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                                    Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_C1IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C3RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                   Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_C3RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C3IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                                    Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);
                               
NppStatus 
nppiDiv_Round_16s_C3IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);
                               
/** 
 * Four 16-bit signed short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_AC4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                    Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_AC4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_AC4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                                     Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_AC4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                 Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C4RSfs_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                   Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_C4RSfs(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,    NppRoundMode rndMode, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param rndMode Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiDiv_Round_16s_C4IRSfs_Ctx(const Npp16s * pSrc,     int nSrcStep, 
                                    Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiDiv_Round_16s_C4IRSfs(const Npp16s * pSrc,     int nSrcStep, 
                                Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppRoundMode rndMode, int nScaleFactor);

/** @} image_divround */ 

/** 
 * @defgroup image_abs Abs
 *
 * Absolute value of each pixel value in an image.
 *
 * @{
 */

/** 
 * One 16-bit signed short channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit signed short channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C1IR_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_C1IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C3IR_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_C3IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image absolute value with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image absolute value with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_AC4IR_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_AC4IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16s_C4IR_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16s_C4IR(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16f_C1IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16f_C1IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16f_C3IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16f_C3IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_16f_C4IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_16f_C4IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C1IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C3IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image absolute value with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image absolute value with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_AC4IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_AC4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image absolute value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image absolute value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbs_32f_C4IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbs_32f_C4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_abs */ 

/** 
 * @defgroup image_absdiff AbsDiff
 *
 * Pixel by pixel absolute difference between two images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiff_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channels absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiff_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channels absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiff_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiff_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_16f_C1R_Ctx(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiff_16f_C1R(const Npp16f * pSrc1, int nSrc1Step, const Npp16f * pSrc2, int nSrc2Step, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel absolute difference of image1 minus image2.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAbsDiff_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAbsDiff_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** @} image_absdiff */ 

/** 
 * @defgroup image_sqr Sqr
 *
 * Square each pixel in an image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C1RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C1IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C3RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C3IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_AC4RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_AC4RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_AC4IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_AC4IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C4RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_C4RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_8u_C4IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_8u_C4IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C1RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C1IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C3RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C3IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_AC4RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_AC4RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_AC4IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_AC4IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C4RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_C4RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16u_C4IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16u_C4IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C1RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C1IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C3RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C3IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_AC4RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_AC4RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_AC4IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_AC4IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C4RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_C4RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16s_C4IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16s_C4IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16f_C1IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16f_C1IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16f_C3IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16f_C3IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_16f_C4IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_16f_C4IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C1IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C3IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image squared with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image squared with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_AC4IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_AC4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image squared.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image squared.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqr_32f_C4IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqr_32f_C4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_sqr */ 

/** @defgroup image_sqrt Sqrt
 *
 * Pixel by pixel square root of each pixel in an image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C1RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C1IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C3RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_C3IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_AC4RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_8u_AC4RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 8-bit unsigned char channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_8u_AC4IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_8u_AC4IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C1RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C1IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C3RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_C3IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_AC4RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16u_AC4RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit unsigned short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16u_AC4IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16u_AC4IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C1RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C1IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C3RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_C3IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_AC4RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16s_AC4RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Four 16-bit signed short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16s_AC4IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16s_AC4IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16f_C1IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16f_C1IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16f_C3IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16f_C3IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_16f_C4IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_16f_C4IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C1IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C3IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image square root with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image square root with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_AC4IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_AC4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel image square root.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit floating point channel in place image square root.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiSqrt_32f_C4IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiSqrt_32f_C4IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_sqrt */ 

/** @defgroup image_ln Ln
 *
 * Pixel by pixel natural logarithm of each pixel in an image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C1RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C1IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C3RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_8u_C3IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C1RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C1IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C3RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16u_C3IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C1RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C1IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C3RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16s_C3IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit floating point channel image natural logarithm.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit floating point channel in place image natural logarithm.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16f_C1IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16f_C1IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel image natural logarithm.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit floating point channel in place image natural logarithm.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_16f_C3IR_Ctx(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_16f_C3IR(Npp16f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel image natural logarithm.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image natural logarithm.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C1IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image natural logarithm.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image natural logarithm.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLn_32f_C3IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLn_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_ln */ 

/** 
 * @defgroup image_exp Exp
 *
 * Exponential value of each pixel in an image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C1RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_8u_C1RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C1IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_8u_C1IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C3RSfs_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_8u_C3RSfs(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_8u_C3IRSfs_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_8u_C3IRSfs(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C1RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16u_C1RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C1IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16u_C1IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C3RSfs_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16u_C3RSfs(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16u_C3IRSfs_Ctx(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16u_C3IRSfs(Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C1RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16s_C1RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C1IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16s_C1IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C3RSfs_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16s_C3RSfs(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, int nScaleFactor);

/** 
 * Three 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_16s_C3IRSfs_Ctx(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_16s_C3IRSfs(Npp16s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, int nScaleFactor);

/** 
 * One 32-bit floating point channel image exponential.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit floating point channel in place image exponential.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C1IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_32f_C1IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel image exponential.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit floating point channel in place image exponential.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiExp_32f_C3IR_Ctx(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiExp_32f_C3IR(Npp32f * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_exp */ 

/** @} image_arithmetic_operations */ 

/** 
 * @defgroup image_logical_operations Logical Operations
 * The set of image processing logical operations available in the library.
 * @{
 */

/** @defgroup image_andc AndC
 *
 * Pixel by pixel logical and of an image with a constant.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical and with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C1IR_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical and with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C3IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_C3IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical and with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                       Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical and with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_AC4IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_AC4IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical and with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_8u_C4IR_Ctx(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_8u_C4IR(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image logical and with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C1IR_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image logical and with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C3IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_C3IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical and with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                        Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                        Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical and with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_AC4IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_AC4IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical and with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_16u_C4IR_Ctx(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_16u_C4IR(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image logical and with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C1IR_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_C1IR(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image logical and with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C3IR_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_C3IR(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical and with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                            Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                        Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical and with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_AC4IR_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_AC4IR(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical and with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[4], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[4], 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical and with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAndC_32s_C4IR_Ctx(const Npp32s  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAndC_32s_C4IR(const Npp32s  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_andc */ 


/** 
 * @defgroup image_orc OrC
 *
 * Pixel by pixel logical or of an image with a constant.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C1IR_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C3IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_C3IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical or with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_AC4IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_AC4IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_8u_C4IR_Ctx(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_8u_C4IR(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image logical or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C1IR_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image logical or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C3IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_C3IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical or with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_AC4IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_AC4IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_16u_C4IR_Ctx(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_16u_C4IR(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image logical or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C1IR_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_C1IR(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image logical or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C3IR_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_C3IR(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical or with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_AC4IR_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_AC4IR(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[4], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[4], 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOrC_32s_C4IR_Ctx(const Npp32s  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiOrC_32s_C4IR(const Npp32s  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_orc */ 

/** 
 * @defgroup image_xorc XorC
 *
 * Pixel by pixel logical exclusive or of an image with a constant.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u nConstant, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical exclusive or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C1IR_Ctx(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_C1IR(const Npp8u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical exclusive or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C3IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_C3IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical exclusive or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                           Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[3], 
                       Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_AC4IR_Ctx(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_AC4IR(const Npp8u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u  aConstants[4], 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_8u_C4IR_Ctx(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_8u_C4IR(const Npp8u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u nConstant, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image logical exclusive or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C1IR_Ctx(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_C1IR(const Npp16u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image logical exclusive or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C3IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_C3IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical exclusive or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                            Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXorC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3], 
                        Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_AC4IR_Ctx(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_AC4IR(const Npp16u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[4], 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_16u_C4IR_Ctx(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_16u_C4IR(const Npp16u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s nConstant, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image logical exclusive or with constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C1IR_Ctx(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_C1IR(const Npp32s nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image logical exclusive or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C3IR_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_C3IR(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or with constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                            Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3], 
                        Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical exclusive or with constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_AC4IR_Ctx(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_AC4IR(const Npp32s  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or with constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[4], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[4], 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image logical exclusive or with constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXorC_32s_C4IR_Ctx(const Npp32s  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiXorC_32s_C4IR(const Npp32s  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_xorc */ 

/** 
 * @defgroup image_rshiftc RShiftC
 *
 * Pixel by pixel right shift of an image by a constant value.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C1IR_Ctx(const Npp32u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_C1IR(const Npp32u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C3IR_Ctx(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_C3IR(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image right shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_AC4IR_Ctx(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_AC4IR(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8u_C4IR_Ctx(const Npp32u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8u_C4IR(const Npp32u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 8-bit signed char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C1R_Ctx(const Npp8s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                             Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_C1R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                         Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit signed char channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C1IR_Ctx(const Npp32u nConstant, Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_C1IR(const Npp32u nConstant, Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit signed char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C3R_Ctx(const Npp8s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                             Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_C3R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                         Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit signed char channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C3IR_Ctx(const Npp32u  aConstants[3], Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_C3IR(const Npp32u  aConstants[3], Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_AC4R_Ctx(const Npp8s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_AC4R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel in place image right shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_AC4IR_Ctx(const Npp32u  aConstants[3], Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_AC4IR(const Npp32u  aConstants[3], Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C4R_Ctx(const Npp8s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                             Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_C4R(const Npp8s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                         Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit signed char channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_8s_C4IR_Ctx(const Npp32u  aConstants[4], Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_8s_C4IR(const Npp32u  aConstants[4], Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C1IR_Ctx(const Npp32u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_C1IR(const Npp32u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C3IR_Ctx(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_C3IR(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image right shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_AC4IR_Ctx(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_AC4IR(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16u_C4IR_Ctx(const Npp32u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16u_C4IR(const Npp32u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit signed short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C1R_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_C1R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit signed short channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C1IR_Ctx(const Npp32u nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_C1IR(const Npp32u nConstant, Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C3R_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_C3R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit signed short channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C3IR_Ctx(const Npp32u  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_C3IR(const Npp32u  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_AC4R_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                               Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_AC4R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                           Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image right shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_AC4IR_Ctx(const Npp32u  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_AC4IR(const Npp32u  aConstants[3], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C4R_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                              Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_C4R(const Npp16s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                          Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit signed short channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_16s_C4IR_Ctx(const Npp32u  aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_16s_C4IR(const Npp32u  aConstants[4], Npp16s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image right shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C1IR_Ctx(const Npp32u nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_C1IR(const Npp32u nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C3IR_Ctx(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_C3IR(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image right shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                               Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image right shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_AC4IR_Ctx(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_AC4IR(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image right shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image right shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiRShiftC_32s_C4IR_Ctx(const Npp32u  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiRShiftC_32s_C4IR(const Npp32u  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_rshiftc */ 

/** 
 * @defgroup image_lshiftc LShiftC
 *
 * Pixel by pixel left shift of an image by a constant value.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image left shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C1IR_Ctx(const Npp32u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_C1IR(const Npp32u nConstant, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image left shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C3IR_Ctx(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_C3IR(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image left shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image left shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_AC4IR_Ctx(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_AC4IR(const Npp32u  aConstants[3], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image left shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_8u_C4IR_Ctx(const Npp32u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_8u_C4IR(const Npp32u  aConstants[4], Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image left shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C1IR_Ctx(const Npp32u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_C1IR(const Npp32u nConstant, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image left shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C3IR_Ctx(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_C3IR(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image left shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                               Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image left shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_AC4IR_Ctx(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_AC4IR(const Npp32u  aConstants[3], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image left shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_16u_C4IR_Ctx(const Npp32u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_16u_C4IR(const Npp32u  aConstants[4], Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nConstant Constant.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u nConstant, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * One 32-bit signed integer channel in place image left shift by constant.
 * \param nConstant Constant.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C1IR_Ctx(const Npp32u nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_C1IR(const Npp32u nConstant, Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel in place image left shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C3IR_Ctx(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_C3IR(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image left shift by constant with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                               Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[3], 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image left shift by constant with unmodified alpha.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_AC4IR_Ctx(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_AC4IR(const Npp32u  aConstants[3], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image left shift by constant.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                              Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32u  aConstants[4], 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel in place image left shift by constant.
 * \param aConstants fixed size array of constant values, one per channel.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiLShiftC_32s_C4IR_Ctx(const Npp32u  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiLShiftC_32s_C4IR(const Npp32u  aConstants[4], Npp32s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_lshiftc */ 

/** 
 * @defgroup image_and And
 *
 * Pixel by pixel logical and of images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 8-bit unsigned char channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C1IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C3IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel image logical and with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical and with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_AC4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                       Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_8u_C4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C1IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C3IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel image logical and with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical and with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_AC4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                        Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_16u_C4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C1IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_C1IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C3IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_C3IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel image logical and with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical and with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_AC4IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                            Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_AC4IR(const Npp32s * pSrc,     int nSrcStep, 
                        Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel image logical and.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical and.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAnd_32s_C4IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiAnd_32s_C4IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_and */ 

/** 
 * @defgroup image_or Or
 *
 * Pixel by pixel logical or of images.
 *
 * @{
 */
 
/** 
 * One 8-bit unsigned char channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                    Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 8-bit unsigned char channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C1IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                     Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                    Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C3IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                     Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel image logical or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_AC4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                        Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                    Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_8u_C4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                         Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                     Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                     Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C1IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                      Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                     Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C3IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                      Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel image logical or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_AC4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                         Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                     Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_16u_C4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                          Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                      Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                     Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C1IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_C1IR(const Npp32s * pSrc,     int nSrcStep, 
                      Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                     Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C3IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_C3IR(const Npp32s * pSrc,     int nSrcStep, 
                      Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel image logical or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_AC4IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_AC4IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel image logical or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                         Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                     Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiOr_32s_C4IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                          Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiOr_32s_C4IR(const Npp32s * pSrc,     int nSrcStep, 
                      Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_or */ 

/** 
 * @defgroup image_xor Xor
 *
 * Pixel by pixel logical exclusive or of images.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 8-bit unsigned char channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C1IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_C1IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 8-bit unsigned char channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C3IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_C3IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel image logical exclusive or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                          Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                      Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_AC4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                           Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_AC4IR(const Npp8u * pSrc,     int nSrcStep, 
                       Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                         Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                     Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 8-bit unsigned char channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_8u_C4IR_Ctx(const Npp8u * pSrc,     int nSrcStep, 
                          Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_8u_C4IR(const Npp8u * pSrc,     int nSrcStep, 
                      Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 16-bit unsigned short channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C1IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_C1IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 16-bit unsigned short channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C3IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_C3IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel image logical exclusive or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                           Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                       Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_AC4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                            Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_AC4IR(const Npp16u * pSrc,     int nSrcStep, 
                        Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                          Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                      Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 16-bit unsigned short channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_16u_C4IR_Ctx(const Npp16u * pSrc,     int nSrcStep, 
                           Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_16u_C4IR(const Npp16u * pSrc,     int nSrcStep, 
                       Npp16u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * One 32-bit signed integer channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C1IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_C1IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 32-bit signed integer channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C3R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Three 32-bit signed integer channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C3IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_C3IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 32-bit signed integer channel image logical exclusive or with unmodified alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                           Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                       Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical exclusive or with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_AC4IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                            Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_AC4IR(const Npp32s * pSrc,     int nSrcStep, 
                        Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel image logical exclusive or.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                          Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_C4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                      Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI);
/** 
 * Four 32-bit signed integer channel in place image logical exclusive or.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiXor_32s_C4IR_Ctx(const Npp32s * pSrc,     int nSrcStep, 
                           Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus 
nppiXor_32s_C4IR(const Npp32s * pSrc,     int nSrcStep, 
                       Npp32s * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_xor */ 

/** 
 * @defgroup image_not Not
 *
 * Pixel by pixel logical not of image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image logical not.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image logical not.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C1IR_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_C1IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image logical not.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image logical not.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C3IR_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_C3IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical not with unmodified alpha.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical not with unmodified alpha.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_AC4IR_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_AC4IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image logical not.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image logical not.
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiNot_8u_C4IR_Ctx(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiNot_8u_C4IR(Npp8u * pSrcDst,  int nSrcDstStep, NppiSize oSizeROI);

/** @} image_not */ 

/** @} image_logical_operations */ 

/** 
 * @defgroup image_alpha_composition_operations Alpha Composition
 * The set of alpha composition operations available in the library.
 * @{
 */

/** 
 * @defgroup image_alphacompc AlphaCompC
 *
 * Composite two images using constant alpha values.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2,
                                Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2,
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Three 8-bit unsigned char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                                Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 8-bit unsigned char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                                Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 8-bit unsigned char channel image composition with alpha using constant source alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                                 Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u * pSrc2, int nSrc2Step, Npp8u nAlpha2, 
                             Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 8-bit signed char channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_8s_C1R_Ctx(const Npp8s * pSrc1, int nSrc1Step, Npp8s nAlpha1, const Npp8s * pSrc2, int nSrc2Step, Npp8s nAlpha2,
                                Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_8s_C1R(const Npp8s * pSrc1, int nSrc1Step, Npp8s nAlpha1, const Npp8s * pSrc2, int nSrc2Step, Npp8s nAlpha2,
                            Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit unsigned short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2,
                                 Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2,
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Three 16-bit unsigned short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                                 Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 16-bit unsigned short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                                 Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * Four 16-bit unsigned short channel image composition with alpha using constant source alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                                  Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, const Npp16u * pSrc2, int nSrc2Step, Npp16u nAlpha2, 
                              Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit signed short channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_16s_C1R_Ctx(const Npp16s * pSrc1, int nSrc1Step, Npp16s nAlpha1, const Npp16s * pSrc2, int nSrc2Step, Npp16s nAlpha2,
                                 Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_16s_C1R(const Npp16s * pSrc1, int nSrc1Step, Npp16s nAlpha1, const Npp16s * pSrc2, int nSrc2Step, Npp16s nAlpha2,
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit unsigned integer channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_32u_C1R_Ctx(const Npp32u * pSrc1, int nSrc1Step, Npp32u nAlpha1, const Npp32u * pSrc2, int nSrc2Step, Npp32u nAlpha2,
                                 Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_32u_C1R(const Npp32u * pSrc1, int nSrc1Step, Npp32u nAlpha1, const Npp32u * pSrc2, int nSrc2Step, Npp32u nAlpha2,
                             Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit signed integer channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_32s_C1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, Npp32s nAlpha1, const Npp32s * pSrc2, int nSrc2Step, Npp32s nAlpha2,
                                 Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_32s_C1R(const Npp32s * pSrc1, int nSrc1Step, Npp32s nAlpha1, const Npp32s * pSrc2, int nSrc2Step, Npp32s nAlpha2,
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit floating point channel image composition using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0.0 - 1.0).
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param nAlpha2 Image alpha opacity (0.0 - 1.0).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaCompC_32f_C1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, Npp32f nAlpha1, const Npp32f * pSrc2, int nSrc2Step, Npp32f nAlpha2,
                                 Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaCompC_32f_C1R(const Npp32f * pSrc1, int nSrc1Step, Npp32f nAlpha1, const Npp32f * pSrc2, int nSrc2Step, Npp32f nAlpha2,
                             Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI,   NppiAlphaOp eAlphaOp);

/** @} image_alphacompc */ 

/** 
 * @defgroup image_alphapremulc AlphaPremulC
 * 
 * Premultiplies pixels of an image using a constant alpha value.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_C1R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * One 8-bit unsigned char channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C1IR_Ctx(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_C1IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C3R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_C3R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three 8-bit unsigned char channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C3IR_Ctx(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_C3IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_C4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_C4IR_Ctx(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_C4IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel image premultiplication with alpha using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u nAlpha1, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image premultiplication with alpha using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_8u_AC4IR_Ctx(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_8u_AC4IR(Npp8u nAlpha1, Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_C1R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * One 16-bit unsigned short channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C1IR_Ctx(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_C1IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C3R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Three 16-bit unsigned short channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C3IR_Ctx(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_C3IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image premultiplication using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_C4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image premultiplication using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_C4IR_Ctx(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_C4IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image premultiplication with alpha using constant alpha.
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u nAlpha1, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image premultiplication with alpha using constant alpha.
 * \param nAlpha1 Image alpha opacity (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremulC_16u_AC4IR_Ctx(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremulC_16u_AC4IR(Npp16u nAlpha1, Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_alphapremulc */ 

/** 
 * @defgroup image_alphacomp AlphaComp
 *
 * Composite two images using alpha opacity values contained in each image.
 *
 * @{
 */

/** 
 * One 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_8u_AC1R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                                Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_8u_AC1R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                                Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, 
                            Npp8u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 8-bit signed char channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_8s_AC1R_Ctx(const Npp8s * pSrc1, int nSrc1Step, const Npp8s * pSrc2, int nSrc2Step, 
                                Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_8s_AC1R(const Npp8s * pSrc1, int nSrc1Step, const Npp8s * pSrc2, int nSrc2Step, 
                            Npp8s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_16u_AC1R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                 Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_16u_AC1R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                                 Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u * pSrc2, int nSrc2Step, 
                             Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 16-bit signed short channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_16s_AC1R_Ctx(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                                 Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_16s_AC1R(const Npp16s * pSrc1, int nSrc1Step, const Npp16s * pSrc2, int nSrc2Step, 
                             Npp16s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32u_AC1R_Ctx(const Npp32u * pSrc1, int nSrc1Step, const Npp32u * pSrc2, int nSrc2Step, 
                                 Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_32u_AC1R(const Npp32u * pSrc1, int nSrc1Step, const Npp32u * pSrc2, int nSrc2Step, 
                             Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32u_AC4R_Ctx(const Npp32u * pSrc1, int nSrc1Step, const Npp32u * pSrc2, int nSrc2Step, 
                                 Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_32u_AC4R(const Npp32u * pSrc1, int nSrc1Step, const Npp32u * pSrc2, int nSrc2Step, 
                             Npp32u * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32s_AC1R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                                 Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_32s_AC1R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32s_AC4R_Ctx(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                                 Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_32s_AC4R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s * pSrc2, int nSrc2Step, 
                             Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * One 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32f_AC1R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                                 Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_32f_AC1R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                             Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** 
 * Four 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pSrc2 \ref source_image_pointer.
 * \param nSrc2Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param eAlphaOp alpha-blending operation..
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaComp_32f_AC4R_Ctx(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                                 Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaComp_32f_AC4R(const Npp32f * pSrc1, int nSrc1Step, const Npp32f * pSrc2, int nSrc2Step, 
                             Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI, NppiAlphaOp eAlphaOp);

/** @} image_alphacomp */ 

/** 
 * @defgroup image_alphapremul AlphaPremul
 * 
 * Premultiplies image pixels by image alpha opacity values.
 *
 * @{
 */

/** 
 * Four 8-bit unsigned char channel image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_8u_AC4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremul_8u_AC4R(const Npp8u * pSrc1, int nSrc1Step, Npp8u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 8-bit unsigned char channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremul_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrc1 \ref source_image_pointer.
 * \param nSrc1Step \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_16u_AC4R_Ctx(const Npp16u * pSrc1, int nSrc1Step, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremul_16u_AC4R(const Npp16u * pSrc1, int nSrc1Step, Npp16u * pDst, int nDstStep, NppiSize oSizeROI);

/** 
 * Four 16-bit unsigned short channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
 * \param pSrcDst \ref in_place_image_pointer.
 * \param nSrcDstStep \ref in_place_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus 
nppiAlphaPremul_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus 
nppiAlphaPremul_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI);

/** @} image_alphapremul */ 

/** @} image_alpha_composition*/ 

/** @} image_arithmetic_and_logical_operations */ 

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPI_ARITHMETIC_AND_LOGICAL_OPERATIONS_H */
