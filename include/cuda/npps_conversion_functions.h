 /* Copyright 2010-2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved. 
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
#ifndef NV_NPPS_CONVERSION_FUNCTIONS_H
#define NV_NPPS_CONVERSION_FUNCTIONS_H
 
/**
 * \file npps_conversion_functions.h
 * NPP Signal Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup signal_conversion_functions Conversion Functions
 *  @ingroup npps
 * Functions that provide conversion and threshold operations
 * @{
 *
 */

/** @defgroup signal_convert Convert
 * The set of conversion operations available in the library
 * @{
 *
 */

/** @name Convert
 * Routines for converting the sample-data type of signals.
 *
 * @{
 *
 */

NppStatus 
nppsConvert_8s16s_Ctx(const Npp8s * pSrc, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_8s16s(const Npp8s * pSrc, Npp16s * pDst, int nLength);

NppStatus 
nppsConvert_8s32f_Ctx(const Npp8s * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_8s32f(const Npp8s * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsConvert_8u32f_Ctx(const Npp8u * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_8u32f(const Npp8u * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsConvert_16s8s_Sfs_Ctx(const Npp16s * pSrc, Npp8s * pDst, Npp32u nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_16s8s_Sfs(const Npp16s * pSrc, Npp8s * pDst, Npp32u nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_16s32s_Ctx(const Npp16s * pSrc, Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_16s32s(const Npp16s * pSrc, Npp32s * pDst, int nLength);

NppStatus 
nppsConvert_16s32f_Ctx(const Npp16s * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_16s32f(const Npp16s * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsConvert_16u32f_Ctx(const Npp16u * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_16u32f(const Npp16u * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsConvert_32s16s_Ctx(const Npp32s * pSrc, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32s16s(const Npp32s * pSrc, Npp16s * pDst, int nLength);

NppStatus 
nppsConvert_32s32f_Ctx(const Npp32s * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32s32f(const Npp32s * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsConvert_32s64f_Ctx(const Npp32s * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32s64f(const Npp32s * pSrc, Npp64f * pDst, int nLength);

NppStatus 
nppsConvert_32f64f_Ctx(const Npp32f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32f64f(const Npp32f * pSrc, Npp64f * pDst, int nLength);

NppStatus 
nppsConvert_64s64f_Ctx(const Npp64s * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_64s64f(const Npp64s * pSrc, Npp64f * pDst, int nLength);

NppStatus 
nppsConvert_64f32f_Ctx(const Npp64f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_64f32f(const Npp64f * pSrc, Npp32f * pDst, int nLength);

NppStatus 
nppsConvert_16s32f_Sfs_Ctx(const Npp16s * pSrc, Npp32f * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_16s32f_Sfs(const Npp16s * pSrc, Npp32f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsConvert_16s64f_Sfs_Ctx(const Npp16s * pSrc, Npp64f * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_16s64f_Sfs(const Npp16s * pSrc, Npp64f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsConvert_32s16s_Sfs_Ctx(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32s16s_Sfs(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsConvert_32s32f_Sfs_Ctx(const Npp32s * pSrc, Npp32f * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32s32f_Sfs(const Npp32s * pSrc, Npp32f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsConvert_32s64f_Sfs_Ctx(const Npp32s * pSrc, Npp64f * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32s64f_Sfs(const Npp32s * pSrc, Npp64f * pDst, int nLength, int nScaleFactor);

NppStatus 
nppsConvert_32f8s_Sfs_Ctx(const Npp32f * pSrc, Npp8s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32f8s_Sfs(const Npp32f * pSrc, Npp8s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_32f8u_Sfs_Ctx(const Npp32f * pSrc, Npp8u * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32f8u_Sfs(const Npp32f * pSrc, Npp8u * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_32f16s_Sfs_Ctx(const Npp32f * pSrc, Npp16s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32f16s_Sfs(const Npp32f * pSrc, Npp16s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_32f16u_Sfs_Ctx(const Npp32f * pSrc, Npp16u * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32f16u_Sfs(const Npp32f * pSrc, Npp16u * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_32f32s_Sfs_Ctx(const Npp32f * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_32f32s_Sfs(const Npp32f * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_64s32s_Sfs_Ctx(const Npp64s * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_64s32s_Sfs(const Npp64s * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_64f16s_Sfs_Ctx(const Npp64f * pSrc, Npp16s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_64f16s_Sfs(const Npp64f * pSrc, Npp16s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_64f32s_Sfs_Ctx(const Npp64f * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_64f32s_Sfs(const Npp64f * pSrc, Npp32s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

NppStatus 
nppsConvert_64f64s_Sfs_Ctx(const Npp64f * pSrc, Npp64s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsConvert_64f64s_Sfs(const Npp64f * pSrc, Npp64s * pDst, int nLength, NppRoundMode eRoundMode, int nScaleFactor);

/** @} end of Convert */

/** @} signal_convert */

/** @defgroup signal_threshold Threshold
 * The set of threshold operations available in the library.
 * @{
 *
 */

/** @name Threshold Functions
 * Performs the threshold operation on the samples of a signal by limiting the sample values by a specified constant value.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit in place signed short signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16s_I_Ctx(Npp16s * pSrcDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit signed short complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit in place signed short complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_16sc_I_Ctx(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 32-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit floating point signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit in place floating point signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64f_I_Ctx(Npp64f * pSrcDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit floating point complex number signal threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 64-bit in place floating point complex number signal threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nRelOp NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, NppCmpOp nRelOp);

/** 
 * 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16s_I_Ctx(Npp16s * pSrcDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_16sc_I_Ctx(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64f_I_Ctx(Npp64f * pSrcDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LT_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LT_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16s_I_Ctx(Npp16s * pSrcDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_16sc_I_Ctx(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel);

/** 
 * 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel);

/** 
 * 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64f_I_Ctx(Npp64f * pSrcDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GT_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GT_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel);

/** 
 * 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, Npp16s nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, Npp16s nValue);

/** 
 * 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16s_I_Ctx(Npp16s * pSrcDst, int nLength, Npp16s nLevel, Npp16s nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel, Npp16s nValue);

/** 
 * 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, Npp16sc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, Npp16sc nValue);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_16sc_I_Ctx(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, Npp16sc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, Npp16sc nValue);

/** 
 * 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, Npp32f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, Npp32f nValue);

/** 
 * 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nLevel, Npp32f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel, Npp32f nValue);

/** 
 * 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, Npp32fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, Npp32fc nValue);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, Npp32fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, Npp32fc nValue);

/** 
 * 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, Npp64f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, Npp64f nValue);

/** 
 * 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64f_I_Ctx(Npp64f * pSrcDst, int nLength, Npp64f nLevel, Npp64f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel, Npp64f nValue);

/** 
 * 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, Npp64fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, Npp64fc nValue);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_LTVal_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, Npp64fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_LTVal_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, Npp64fc nValue);

/** 
 * 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, Npp16s nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s nLevel, Npp16s nValue);

/** 
 * 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16s_I_Ctx(Npp16s * pSrcDst, int nLength, Npp16s nLevel, Npp16s nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_16s_I(Npp16s * pSrcDst, int nLength, Npp16s nLevel, Npp16s nValue);

/** 
 * 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, Npp16sc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16s nLevel, Npp16sc nValue);

/** 
 * 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_16sc_I_Ctx(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, Npp16sc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_16sc_I(Npp16sc * pSrcDst, int nLength, Npp16s nLevel, Npp16sc nValue);

/** 
 * 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, Npp32f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f nLevel, Npp32f nValue);

/** 
 * 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nLevel, Npp32f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nLevel, Npp32f nValue);

/** 
 * 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, Npp32fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32f nLevel, Npp32fc nValue);

/** 
 * 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, Npp32fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_32fc_I(Npp32fc * pSrcDst, int nLength, Npp32f nLevel, Npp32fc nValue);

/** 
 * 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, Npp64f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f nLevel, Npp64f nValue);

/** 
 * 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64f_I_Ctx(Npp64f * pSrcDst, int nLength, Npp64f nLevel, Npp64f nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_64f_I(Npp64f * pSrcDst, int nLength, Npp64f nLevel, Npp64f nValue);

/** 
 * 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, Npp64fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64f nLevel, Npp64fc nValue);

/** 
 * 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nLevel Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample
 * \param nValue Constant value to replace source value when threshold test is true.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsThreshold_GTVal_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, Npp64fc nValue, NppStreamContext nppStreamCtx);

NppStatus 
nppsThreshold_GTVal_64fc_I(Npp64fc * pSrcDst, int nLength, Npp64f nLevel, Npp64fc nValue);

/** @} end of Threshold */

/** @} signal_threshold */

/** @} signal_conversion_functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPS_CONVERSION_FUNCTIONS_H */
