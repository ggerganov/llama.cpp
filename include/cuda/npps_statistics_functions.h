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
#ifndef NV_NPPS_STATISTICS_FUNCTIONS_H
#define NV_NPPS_STATISTICS_FUNCTIONS_H
 
/**
 * \file npps_statistics_functions.h
 * NPP Signal Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup signal_statistical_functions Statistical Functions
 *  @ingroup npps
 * Functions that provide global signal statistics like: sum, mean, standard
 * deviation, min, max, etc.
 *
 * @{
 *
 */

/** @defgroup signal_min_every_or_max_every MinEvery And MaxEvery Functions
 * Performs the min or max operation on the samples of a signal.
 *
 * @{  
 *
 */

/** 
 * 8-bit in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinEvery_8u_I_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinEvery_8u_I(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinEvery_16u_I_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinEvery_16u_I(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength);

/** 
 * 16-bit signed short integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinEvery_16s_I_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinEvery_16s_I(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit signed integer in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinEvery_32s_I_Ctx(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinEvery_32s_I(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength);

/** 
 * 32-bit floating point in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinEvery_32f_I_Ctx(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinEvery_32f_I(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point in place min value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinEvery_64f_I_Ctx(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinEvery_64f_I(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength);

/** 
 * 8-bit in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxEvery_8u_I_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxEvery_8u_I(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxEvery_16u_I_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxEvery_16u_I(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength);

/** 
 * 16-bit signed short integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxEvery_16s_I_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxEvery_16s_I(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit signed integer in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxEvery_32s_I_Ctx(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxEvery_32s_I(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength);

/** 
 * 32-bit floating point in place max value for each pair of elements.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxEvery_32f_I_Ctx(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxEvery_32f_I(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength);

/** 
 *
 * @} signal_min_every_or_max_every
 *
 */

/** @defgroup signal_sum Sum
 * Performs the sum operation on the samples of a signal.
 * @{  
 *
 */
 
 /** 
 * Device scratch buffer size (in bytes) for nppsSum_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_32fc_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_32fc(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_64fc_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_64fc(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_16s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_16s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_16sc_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_16sc_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_16sc32sc_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_16sc32sc_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsSum_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_32s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);
 
NppStatus 
nppsSumGetBufferSize_32s_Sfs(int nLength, int * hpBufferSize /* host pointer */);
 
/** 
 * Device scratch buffer size (in bytes) for nppsSum_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumGetBufferSize_16s32s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumGetBufferSize_16s32s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_32f(const Npp32f * pSrc, int nLength, Npp32f * pSum, Npp8u * pDeviceBuffer);

/** 
 * 32-bit float complex vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_32fc_Ctx(const Npp32fc * pSrc, int nLength, Npp32fc * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_32fc(const Npp32fc * pSrc, int nLength, Npp32fc * pSum, Npp8u * pDeviceBuffer);

/** 
 * 64-bit double vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_64f(const Npp64f * pSrc, int nLength, Npp64f * pSum, Npp8u * pDeviceBuffer);

/** 
 * 64-bit double complex vector sum method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_64fc_Ctx(const Npp64fc * pSrc, int nLength, Npp64fc * pSum, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_64fc(const Npp64fc * pSrc, int nLength, Npp64fc * pSum, Npp8u * pDeviceBuffer);

/** 
 * 16-bit short vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_16s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pSum, int nScaleFactor, 
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_16s_Sfs(const Npp16s * pSrc, int nLength, Npp16s * pSum, int nScaleFactor, 
                Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_32s_Sfs_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pSum, int nScaleFactor, 
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_32s_Sfs(const Npp32s * pSrc, int nLength, Npp32s * pSum, int nScaleFactor, 
                Npp8u * pDeviceBuffer);

/** 
 * 16-bit short complex vector sum with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_16sc_Sfs_Ctx(const Npp16sc * pSrc, int nLength, Npp16sc * pSum, int nScaleFactor, 
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_16sc_Sfs(const Npp16sc * pSrc, int nLength, Npp16sc * pSum, int nScaleFactor, 
                 Npp8u * pDeviceBuffer);

/** 
 * 16-bit short complex vector sum (32bit int complex) with integer scaling
 * method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_16sc32sc_Sfs_Ctx(const Npp16sc * pSrc, int nLength, Npp32sc * pSum, int nScaleFactor, 
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_16sc32sc_Sfs(const Npp16sc * pSrc, int nLength, Npp32sc * pSum, int nScaleFactor, 
                     Npp8u * pDeviceBuffer);

/** 
 * 16-bit integer vector sum (32bit) with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pSum Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsSumGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSum_16s32s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp32s * pSum, int nScaleFactor,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSum_16s32s_Sfs(const Npp16s * pSrc, int nLength, Npp32s * pSum, int nScaleFactor,
                   Npp8u * pDeviceBuffer);

/** @} signal_sum */


/** @defgroup signal_max Maximum
 * Performs the maximum operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for nppsMax_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMax_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMax_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMax_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMax_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMax_16s(const Npp16s * pSrc, int nLength, Npp16s * pMax, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMax_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMax_32s(const Npp32s * pSrc, int nLength, Npp32s * pMax, Npp8u * pDeviceBuffer);

/** 
 * 32-bit float vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMax_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMax_32f(const Npp32f * pSrc, int nLength, Npp32f * pMax, Npp8u * pDeviceBuffer);

/** 
 * 64-bit float vector max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMax_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMax, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMax_64f(const Npp64f * pSrc, int nLength, Npp64f * pMax, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxIndxGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndxGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxIndxGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndxGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxIndxGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndxGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxIndxGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndxGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxIndx_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndx_16s(const Npp16s * pSrc, int nLength, Npp16s * pMax, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxIndx_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndx_32s(const Npp32s * pSrc, int nLength, Npp32s * pMax, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 32-bit float vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxIndx_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndx_32f(const Npp32f * pSrc, int nLength, Npp32f * pMax, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 64-bit float vector max index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMax Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxIndx_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMax, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxIndx_64f(const Npp64f * pSrc, int nLength, Npp64f * pMax, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxAbs_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxAbsGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbsGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxAbs_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxAbsGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbsGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxAbs_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMaxAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbs_16s(const Npp16s * pSrc, int nLength, Npp16s * pMaxAbs, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxAbsGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxAbs_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMaxAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbs_32s(const Npp32s * pSrc, int nLength, Npp32s * pMaxAbs, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxAbsIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxAbsIndxGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbsIndxGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMaxAbsIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMaxAbsIndxGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbsIndxGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector max absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxAbsIndx_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbsIndx_16s(const Npp16s * pSrc, int nLength, Npp16s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector max absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMaxAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first maximum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaxAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaxAbsIndx_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaxAbsIndx_32s(const Npp32s * pSrc, int nLength, Npp32s * pMaxAbs, int * pIndx, Npp8u * pDeviceBuffer);

/** @} signal_max */

/** @defgroup signal_min Minimum
 * Performs the minimum operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for nppsMin_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMin_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMin_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMin_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMin_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMin_16s(const Npp16s * pSrc, int nLength, Npp16s * pMin, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMin_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMin_32s(const Npp32s * pSrc, int nLength, Npp32s * pMin, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMin_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMin_32f(const Npp32f * pSrc, int nLength, Npp32f * pMin, Npp8u * pDeviceBuffer);

/** 
 * 64-bit integer vector min method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMin_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMin, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMin_64f(const Npp64f * pSrc, int nLength, Npp64f * pMin, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for nppsMinIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinIndxGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndxGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMinIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinIndxGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndxGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMinIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinIndxGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndxGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMinIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinIndxGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndxGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinIndx_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndx_16s(const Npp16s * pSrc, int nLength, Npp16s * pMin, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinIndx_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndx_32s(const Npp32s * pSrc, int nLength, Npp32s * pMin, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 32-bit float vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinIndx_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndx_32f(const Npp32f * pSrc, int nLength, Npp32f * pMin, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 64-bit float vector min index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinIndx_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMin, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinIndx_64f(const Npp64f * pSrc, int nLength, Npp64f * pMin, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for nppsMinAbs_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinAbsGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbsGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMinAbs_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinAbsGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbsGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinAbs_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMinAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbs_16s(const Npp16s * pSrc, int nLength, Npp16s * pMinAbs, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min absolute method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinAbsGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinAbs_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMinAbs, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbs_32s(const Npp32s * pSrc, int nLength, Npp32s * pMinAbs, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for nppsMinAbsIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinAbsIndxGetBufferSize_16s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbsIndxGetBufferSize_16s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMinAbsIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMinAbsIndxGetBufferSize_32s_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbsIndxGetBufferSize_32s(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit integer vector min absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinAbsIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinAbsIndx_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbsIndx_16s(const Npp16s * pSrc, int nLength, Npp16s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector min absolute index method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMinAbs Pointer to the output result.
 * \param pIndx Pointer to the index value of the first minimum element.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinAbsIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinAbsIndx_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinAbsIndx_32s(const Npp32s * pSrc, int nLength, Npp32s * pMinAbs, int * pIndx, Npp8u * pDeviceBuffer);

/** @} signal_min */

/** @defgroup signal_mean Mean
 * Performs the mean operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for nppsMean_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMean_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_32fc_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_32fc(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMean_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMean_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_64fc_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_64fc(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMean_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_16s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_16s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMean_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_32s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_32s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMean_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanGetBufferSize_16sc_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanGetBufferSize_16sc_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_32f(const Npp32f * pSrc, int nLength, Npp32f * pMean, Npp8u * pDeviceBuffer);

/** 
 * 32-bit float complex vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_32fc_Ctx(const Npp32fc * pSrc, int nLength, Npp32fc * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_32fc(const Npp32fc * pSrc, int nLength, Npp32fc * pMean, Npp8u * pDeviceBuffer);

/** 
 * 64-bit double vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_64f(const Npp64f * pSrc, int nLength, Npp64f * pMean, Npp8u * pDeviceBuffer);

/** 
 * 64-bit double complex vector mean method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_64fc_Ctx(const Npp64fc * pSrc, int nLength, Npp64fc * pMean, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_64fc(const Npp64fc * pSrc, int nLength, Npp64fc * pMean, Npp8u * pDeviceBuffer);

/** 
 * 16-bit short vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_16s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMean, int nScaleFactor, 
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_16s_Sfs(const Npp16s * pSrc, int nLength, Npp16s * pMean, int nScaleFactor, 
                 Npp8u * pDeviceBuffer);

/** 
 * 32-bit integer vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_32s_Sfs_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMean, int nScaleFactor, 
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_32s_Sfs(const Npp32s * pSrc, int nLength, Npp32s * pMean, int nScaleFactor, 
                 Npp8u * pDeviceBuffer);

/** 
 * 16-bit short complex vector mean with integer scaling method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMean_16sc_Sfs_Ctx(const Npp16sc * pSrc, int nLength, Npp16sc * pMean, int nScaleFactor, 
                      Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMean_16sc_Sfs(const Npp16sc * pSrc, int nLength, Npp16sc * pMean, int nScaleFactor, 
                  Npp8u * pDeviceBuffer);

/** @} signal_mean */

/** @defgroup signal_standard_deviation Standard Deviation
 * Calculates the standard deviation for the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for nppsStdDev_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsStdDevGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDevGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsStdDev_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsStdDevGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDevGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsStdDev_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsStdDevGetBufferSize_16s32s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDevGetBufferSize_16s32s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsStdDev_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsStdDevGetBufferSize_16s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDevGetBufferSize_16s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsStdDevGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsStdDev_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDev_32f(const Npp32f * pSrc, int nLength, Npp32f * pStdDev, Npp8u * pDeviceBuffer);

/** 
 * 64-bit float vector standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsStdDevGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsStdDev_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDev_64f(const Npp64f * pSrc, int nLength, Npp64f * pStdDev, Npp8u * pDeviceBuffer);

/** 
 * 16-bit float vector standard deviation method (return value is 32-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsStdDev_16s32s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp32s * pStdDev, int nScaleFactor, 
                          Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDev_16s32s_Sfs(const Npp16s * pSrc, int nLength, Npp32s * pStdDev, int nScaleFactor, 
                      Npp8u * pDeviceBuffer);

/** 
 * 16-bit float vector standard deviation method (return value is also 16-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pStdDev Pointer to the output result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsStdDev_16s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pStdDev, int nScaleFactor, 
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsStdDev_16s_Sfs(const Npp16s * pSrc, int nLength, Npp16s * pStdDev, int nScaleFactor, 
                   Npp8u * pDeviceBuffer);

/** @} signal_standard_deviation */

/** @defgroup signal_mean_and_standard_deviation Mean And Standard Deviation
 * Performs the mean and calculates the standard deviation for the samples of a signal.
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for nppsMeanStdDev_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanStdDevGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDevGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMeanStdDev_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanStdDevGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDevGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMeanStdDev_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDevGetBufferSize_16s32s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * Device scratch buffer size (in bytes) for nppsMeanStdDev_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsMeanStdDevGetBufferSize_16s_Sfs_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDevGetBufferSize_16s_Sfs(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 32-bit float vector mean and standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanStdDevGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMeanStdDev_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMean, Npp32f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDev_32f(const Npp32f * pSrc, int nLength, Npp32f * pMean, Npp32f * pStdDev, Npp8u * pDeviceBuffer);

/** 
 * 64-bit float vector mean and standard deviation method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanStdDevGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMeanStdDev_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMean, Npp64f * pStdDev, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDev_64f(const Npp64f * pSrc, int nLength, Npp64f * pMean, Npp64f * pStdDev, Npp8u * pDeviceBuffer);

/** 
 * 16-bit float vector mean and standard deviation method (return values are 32-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanStdDevGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMeanStdDev_16s32s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp32s * pMean, Npp32s * pStdDev, int nScaleFactor, 
                              Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDev_16s32s_Sfs(const Npp16s * pSrc, int nLength, Npp32s * pMean, Npp32s * pStdDev, int nScaleFactor, 
                          Npp8u * pDeviceBuffer);

/** 
 * 16-bit float vector mean and standard deviation method (return values are also 16-bit)
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMean Pointer to the output mean value.
 * \param pStdDev Pointer to the output standard deviation value.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMeanStdDevGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMeanStdDev_16s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMean, Npp16s * pStdDev, int nScaleFactor, 
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMeanStdDev_16s_Sfs(const Npp16s * pSrc, int nLength, Npp16s * pMean, Npp16s * pStdDev, int nScaleFactor, 
                       Npp8u * pDeviceBuffer);

/** @} signal_mean_and_standard_deviation */

/** @defgroup signal_min_max Minimum Maximum
 * Performs the maximum and the minimum operation on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsMinMax_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_8u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_8u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMax_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_16s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_16s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMax_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_16u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_16u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMax_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMax_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_32u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_32u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMax_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMax_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 8-bit char vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_8u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_8u_Ctx(const Npp8u * pSrc, int nLength, Npp8u * pMin, Npp8u * pMax, 
                  Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_8u(const Npp8u * pSrc, int nLength, Npp8u * pMin, Npp8u * pMax, 
              Npp8u * pDeviceBuffer);

/** 
 * 16-bit signed short vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMin, Npp16s * pMax, 
                   Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_16s(const Npp16s * pSrc, int nLength, Npp16s * pMin, Npp16s * pMax, 
               Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_16u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_16u_Ctx(const Npp16u * pSrc, int nLength, Npp16u * pMin, Npp16u * pMax, 
                   Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_16u(const Npp16u * pSrc, int nLength, Npp16u * pMin, Npp16u * pMax, 
               Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned int vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_32u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_32u_Ctx(const Npp32u * pSrc, int nLength, Npp32u * pMin, Npp32u * pMax, 
                   Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_32u(const Npp32u * pSrc, int nLength, Npp32u * pMin, Npp32u * pMax, 
               Npp8u * pDeviceBuffer);

/** 
 * 32-bit signed int vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMin, Npp32s * pMax, 
                   Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_32s(const Npp32s * pSrc, int nLength, Npp32s * pMin, Npp32s * pMax, 
               Npp8u * pDeviceBuffer);

/** 
 * 32-bit float vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMin, Npp32f * pMax, 
                   Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_32f(const Npp32f * pSrc, int nLength, Npp32f * pMin, Npp32f * pMax, 
               Npp8u * pDeviceBuffer);

/** 
 * 64-bit double vector min and max method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMax Pointer to the max output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMax_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMin, Npp64f * pMax, 
                   Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMax_64f(const Npp64f * pSrc, int nLength, Npp64f * pMin, Npp64f * pMax, 
               Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_8u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_8u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_16s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_16s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_16u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_16u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_32u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_32u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMinMaxIndx_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMinMaxIndxGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMinMaxIndxGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 8-bit char vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_8u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_8u_Ctx(const Npp8u * pSrc, int nLength, Npp8u * pMin, int * pMinIndx, Npp8u * pMax, int * pMaxIndx,
                      Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMaxIndx_8u(const Npp8u * pSrc, int nLength, Npp8u * pMin, int * pMinIndx, Npp8u * pMax, int * pMaxIndx,
                  Npp8u * pDeviceBuffer);

/** 
 * 16-bit signed short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_16s_Ctx(const Npp16s * pSrc, int nLength, Npp16s * pMin, int * pMinIndx, Npp16s * pMax, int * pMaxIndx,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMaxIndx_16s(const Npp16s * pSrc, int nLength, Npp16s * pMin, int * pMinIndx, Npp16s * pMax, int * pMaxIndx,
                   Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_16u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_16u_Ctx(const Npp16u * pSrc, int nLength, Npp16u * pMin, int * pMinIndx, Npp16u * pMax, int * pMaxIndx,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);
        
NppStatus 
nppsMinMaxIndx_16u(const Npp16u * pSrc, int nLength, Npp16u * pMin, int * pMinIndx, Npp16u * pMax, int * pMaxIndx,
                   Npp8u * pDeviceBuffer);

/** 
 * 32-bit signed short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_32s_Ctx(const Npp32s * pSrc, int nLength, Npp32s * pMin, int * pMinIndx, Npp32s * pMax, int * pMaxIndx,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMaxIndx_32s(const Npp32s * pSrc, int nLength, Npp32s * pMin, int * pMinIndx, Npp32s * pMax, int * pMaxIndx,
                   Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_32u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_32u_Ctx(const Npp32u * pSrc, int nLength, Npp32u * pMin, int * pMinIndx, Npp32u * pMax, int * pMaxIndx,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMaxIndx_32u(const Npp32u * pSrc, int nLength, Npp32u * pMin, int * pMinIndx, Npp32u * pMax, int * pMaxIndx,
                   Npp8u * pDeviceBuffer);

/** 
 * 32-bit float vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pMin, int * pMinIndx, Npp32f * pMax, int * pMaxIndx,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMaxIndx_32f(const Npp32f * pSrc, int nLength, Npp32f * pMin, int * pMinIndx, Npp32f * pMax, int * pMaxIndx,
                   Npp8u * pDeviceBuffer);

/** 
 * 64-bit float vector min and max with indices method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pMin Pointer to the min output result.
 * \param pMinIndx Pointer to the index of the first min value.
 * \param pMax Pointer to the max output result.
 * \param pMaxIndx Pointer to the index of the first max value.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMinMaxIndxGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMinMaxIndx_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pMin, int * pMinIndx, Npp64f * pMax, int * pMaxIndx, 
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMinMaxIndx_64f(const Npp64f * pSrc, int nLength, Npp64f * pMin, int * pMinIndx, Npp64f * pMax, int * pMaxIndx, 
                   Npp8u * pDeviceBuffer);

/** @} signal_min_max */

/** @defgroup signal_infinity_norm Infinity Norm
 * Performs the infinity norm on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsNorm_Inf_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormInfGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormInfGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float vector C norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormInfGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_Inf_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pNorm,
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_Inf_32f(const Npp32f * pSrc, int nLength, Npp32f * pNorm,
                 Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_Inf_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormInfGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormInfGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float vector C norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormInfGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_Inf_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pNorm,
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_Inf_64f(const Npp64f * pSrc, int nLength, Npp64f * pNorm,
                 Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_Inf_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormInfGetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormInfGetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector C norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormInfGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_Inf_16s32f_Ctx(const Npp16s * pSrc, int nLength, Npp32f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_Inf_16s32f(const Npp16s * pSrc, int nLength, Npp32f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_Inf_32fc32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormInfGetBufferSize_32fc32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormInfGetBufferSize_32fc32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex vector C norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormInfGetBufferSize_32fc32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_Inf_32fc32f_Ctx(const Npp32fc * pSrc, int nLength, Npp32f * pNorm,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_Inf_32fc32f(const Npp32fc * pSrc, int nLength, Npp32f * pNorm,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_Inf_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormInfGetBufferSize_64fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormInfGetBufferSize_64fc64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex vector C norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormInfGetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_Inf_64fc64f_Ctx(const Npp64fc * pSrc, int nLength, Npp64f * pNorm,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_Inf_64fc64f(const Npp64fc * pSrc, int nLength, Npp64f * pNorm,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_Inf_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormInfGetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormInfGetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector C norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_Inf_16s32s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp32s * pNorm, int nScaleFactor,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_Inf_16s32s_Sfs(const Npp16s * pSrc, int nLength, Npp32s * pNorm, int nScaleFactor,
                        Npp8u * pDeviceBuffer);

/** @} signal_infinity_norm */

/** @defgroup signal_L1_norm L1 Norm
 * Performs the L1 norm on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float vector L1 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pNorm,
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_32f(const Npp32f * pSrc, int nLength, Npp32f * pNorm,
                Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float vector L1 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_64f(const Npp64f * pSrc, int nLength, Npp64f * pNorm,
                Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector L1 norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the L1 norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_16s32f_Ctx(const Npp16s * pSrc, int nLength, Npp32f * pNorm,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_16s32f(const Npp16s * pSrc, int nLength, Npp32f * pNorm,
                   Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_32fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_32fc64f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex vector L1 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_32fc64f_Ctx(const Npp32fc * pSrc, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_32fc64f(const Npp32fc * pSrc, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_64fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_64fc64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex vector L1 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_64fc64f_Ctx(const Npp64fc * pSrc, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_64fc64f(const Npp64fc * pSrc, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector L1 norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_16s32s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp32s * pNorm, int nScaleFactor,
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_16s32s_Sfs(const Npp16s * pSrc, int nLength, Npp32s * pNorm, int nScaleFactor,
                       Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L1_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL1GetBufferSize_16s64s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL1GetBufferSize_16s64s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector L1 norm method, return value is 64-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L1_16s64s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp64s * pNorm, int nScaleFactor,
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L1_16s64s_Sfs(const Npp16s * pSrc, int nLength, Npp64s * pNorm, int nScaleFactor,
                       Npp8u * pDeviceBuffer);

/** @} signal_L1_norm */

/** @defgroup signal_L2_norm L2 Norm
 * Performs the L2 norm on the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2GetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2GetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float vector L2 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2GetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2_32f(const Npp32f * pSrc, int nLength, Npp32f * pNorm, Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2GetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2GetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float vector L2 norm method
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2GetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2_64f(const Npp64f * pSrc, int nLength, Npp64f * pNorm,
                Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2GetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2GetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector L2 norm method, return value is 32-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2_16s32f_Ctx(const Npp16s * pSrc, int nLength, Npp32f * pNorm,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2_16s32f(const Npp16s * pSrc, int nLength, Npp32f * pNorm,
                   Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2GetBufferSize_32fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2GetBufferSize_32fc64f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex vector L2 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2_32fc64f_Ctx(const Npp32fc * pSrc, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2_32fc64f(const Npp32fc * pSrc, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2GetBufferSize_64fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2GetBufferSize_64fc64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex vector L2 norm method, return value is 64-bit float.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2_64fc64f_Ctx(const Npp64fc * pSrc, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2_64fc64f(const Npp64fc * pSrc, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2GetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2GetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector L2 norm method, return value is 32-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2_16s32s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp32s * pNorm, int nScaleFactor,
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2_16s32s_Sfs(const Npp16s * pSrc, int nLength, Npp32s * pNorm, int nScaleFactor,
                       Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNorm_L2Sqr_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormL2SqrGetBufferSize_16s64s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormL2SqrGetBufferSize_16s64s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer vector L2 Square norm method, return value is 64-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNorm_L2Sqr_16s64s_Sfs_Ctx(const Npp16s * pSrc, int nLength, Npp64s * pNorm, int nScaleFactor,
                              Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNorm_L2Sqr_16s64s_Sfs(const Npp16s * pSrc, int nLength, Npp64s * pNorm, int nScaleFactor,
                          Npp8u * pDeviceBuffer);

/** @} signal_L2_norm */

/** @defgroup signal_infinity_norm_diff Infinity Norm Diff
 * Performs the infinity norm on the samples of two input signals' difference.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_Inf_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffInfGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffInfGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float C norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffInfGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_Inf_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pNorm,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_Inf_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pNorm,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_Inf_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffInfGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffInfGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float C norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffInfGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_Inf_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pNorm,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_Inf_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pNorm,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffInfGetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffInfGetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffInfGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_Inf_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pNorm,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_Inf_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pNorm,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_Inf_32fc32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffInfGetBufferSize_32fc32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffInfGetBufferSize_32fc32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex C norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffInfGetBufferSize_32fc32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_Inf_32fc32f_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp32f * pNorm,
                             Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_Inf_32fc32f(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp32f * pNorm,
                         Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_Inf_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffInfGetBufferSize_64fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffInfGetBufferSize_64fc64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex C norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffInfGetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_Inf_64fc64f_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pNorm,
                             Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_Inf_64fc64f(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pNorm,
                         Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffInfGetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffInfGetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffInfGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_Inf_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pNorm, int nScaleFactor,
                                Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_Inf_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pNorm, int nScaleFactor,
                            Npp8u * pDeviceBuffer);

/** @} signal_infinity_norm_diff */

/** @defgroup signal_L1_norm_diff L1 Norm Diff
 * Performs the L1 norm on the samples of two input signals' difference.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float L1 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float L1 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the L1 norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pNorm,
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pNorm,
                       Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_32fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_32fc64f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_32fc64f_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pNorm,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_32fc64f(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_64fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_64fc64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_64fc64f_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pNorm,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_64fc64f(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer..
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pNorm, int nScaleFactor,
                               Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pNorm, int nScaleFactor,
                           Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L1_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s64s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL1GetBufferSize_16s64s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer L1 norm method on two vectors' difference, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL1GetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L1_16s64s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64s * pNorm, int nScaleFactor,
                               Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L1_16s64s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64s * pNorm, int nScaleFactor,
                           Npp8u * pDeviceBuffer);

/** @} signal_L1_norm_diff */

/** @defgroup signal_L2_norm_diff L2 Norm Diff
 * Performs the L2 norm on the samples of two input signals' difference.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2GetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2GetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float L2 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2GetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2GetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2GetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float L2 norm method on two vectors' difference
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2GetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pNorm,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2GetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2GetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2GetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pNorm,
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pNorm,
                       Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2_32fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2GetBufferSize_32fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2GetBufferSize_32fc64f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2GetBufferSize_32fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2_32fc64f_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pNorm,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2_32fc64f(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2_64fc64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2GetBufferSize_64fc64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2GetBufferSize_64fc64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2GetBufferSize_64fc64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2_64fc64f_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pNorm,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2_64fc64f(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pNorm,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2GetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2GetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2GetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pNorm, int nScaleFactor,
                               Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pNorm, int nScaleFactor,
                           Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsNormDiff_L2Sqr_16s64s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer L2 Square norm method on two vectors' difference, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pNorm Pointer to the norm result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormDiff_L2Sqr_16s64s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64s * pNorm, int nScaleFactor,
                                  Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormDiff_L2Sqr_16s64s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64s * pNorm, int nScaleFactor,
                              Npp8u * pDeviceBuffer);

/** @} signal_l2_norm_diff */

/** @defgroup signal_dot_product Dot Product
 * Performs the dot product operation on the samples of two input signals.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float dot product method, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pDp,
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp32f * pDp,
                Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32fc(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp32fc * pDp,
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp32fc * pDp,
                 Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32f32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32f32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32f32fc(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float and 32-bit float complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32f32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32f32fc_Ctx(const Npp32f * pSrc1, const Npp32fc * pSrc2, int nLength, Npp32fc * pDp,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32f32fc(const Npp32f * pSrc1, const Npp32fc * pSrc2, int nLength, Npp32fc * pDp,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32f64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32f64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32f64f(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float dot product method, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32f64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32f64f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDp,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32f64f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDp,
                   Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32fc64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32fc64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32fc64fc(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32fc64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32fc64fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64fc * pDp,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32fc64fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64fc * pDp,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32f32fc64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32f32fc64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32f32fc64fc(int nLength,  int * hpBufferSize);

/** 
 * 32-bit float and 32-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32f32fc64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32f32fc64fc_Ctx(const Npp32f * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64fc * pDp,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32f32fc64fc(const Npp32f * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64fc * pDp,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float dot product method, return value is 64-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDp,
                    Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDp,
                Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_64fc(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64fc * pDp,
                     Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64fc * pDp,
                 Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_64f64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_64f64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_64f64fc(int nLength,  int * hpBufferSize);

/** 
 * 64-bit float and 64-bit float complex dot product method, return value is 64-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_64f64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_64f64fc_Ctx(const Npp64f * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64fc * pDp,
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_64f64fc(const Npp64f * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64fc * pDp,
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s64s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s64s(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 64-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s64s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s64s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64s * pDp,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s64s(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64s * pDp,
                   Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16sc64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16sc64sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16sc64sc(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 64-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16sc64sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16sc64sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64sc * pDp,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16sc64sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64sc * pDp,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s16sc64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc64sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s16sc64sc(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer short dot product method, return value is 64-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s16sc64sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s16sc64sc_Ctx(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64sc * pDp,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s16sc64sc(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64sc * pDp,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s32f(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 32-bit float.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pDp,
                       Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32f * pDp,
                   Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16sc32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16sc32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16sc32fc(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16sc32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16sc32fc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32fc * pDp,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16sc32fc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32fc * pDp,
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s16sc32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s16sc32fc(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s16sc32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s16sc32fc_Ctx(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32fc * pDp,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s16sc32fc(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32fc * pDp,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 16-bit signed short integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp16s * pDp, int nScaleFactor, 
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp16s * pDp, int nScaleFactor, 
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16sc_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16sc_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16sc_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16sc_Sfs_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp16sc * pDp, int nScaleFactor, 
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16sc_Sfs(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp16sc * pDp, int nScaleFactor, 
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 32-bit signed integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp32s * pDp, int nScaleFactor, 
                        Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp32s * pDp, int nScaleFactor, 
                    Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32sc_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32sc_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 32-bit signed integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32sc_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32sc_Sfs_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32sc_Sfs(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                     Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pDp, int nScaleFactor, 
                           Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp32s * pDp, int nScaleFactor, 
                       Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc32sc_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s16sc32sc_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s16sc32sc_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s16sc32sc_Sfs_Ctx(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                                Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s16sc32sc_Sfs(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                            Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s32s32s_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s32s32s_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s32s32s_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer and 32-bit signed integer dot product method, return value is 32-bit signed integer.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s32s32s_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s32s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp32s * pSrc2, int nLength, Npp32s * pDp, int nScaleFactor, 
                              Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s32s32s_Sfs(const Npp16s * pSrc1, const Npp32s * pSrc2, int nLength, Npp32s * pDp, int nScaleFactor, 
                          Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16s16sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16s16sc_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16s16sc_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16s16sc_Sfs_Ctx(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp16sc * pDp, int nScaleFactor, 
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16s16sc_Sfs(const Npp16s * pSrc1, const Npp16sc * pSrc2, int nLength, Npp16sc * pDp, int nScaleFactor, 
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_16sc32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_16sc32sc_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_16sc32sc_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_16sc32sc_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_16sc32sc_Sfs_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                             Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_16sc32sc_Sfs(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                         Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsDotProd_32s32sc_Sfs.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsDotProdGetBufferSize_32s32sc_Sfs_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsDotProdGetBufferSize_32s32sc_Sfs(int nLength,  int * hpBufferSize);

/** 
 * 32-bit signed short integer and 32-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDp Pointer to the dot product result.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsDotProdGetBufferSize_32s32sc_Sfs to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDotProd_32s32sc_Sfs_Ctx(const Npp32s * pSrc1, const Npp32sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsDotProd_32s32sc_Sfs(const Npp32s * pSrc1, const Npp32sc * pSrc2, int nLength, Npp32sc * pDp, int nScaleFactor, 
                        Npp8u * pDeviceBuffer);

/** @} signal_dot_product */

/** @defgroup signal_count_in_range Count In Range
 * Calculates the number of elements from specified range in the samples of a signal.
 * @{
 *
 */

/** 
 * Device-buffer size (in bytes) for nppsCountInRange_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsCountInRangeGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsCountInRangeGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Computes the number of elements whose values fall into the specified range on a 32-bit signed integer array.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pCounts Pointer to the number of elements.
 * \param nLowerBound Lower bound of the specified range.
 * \param nUpperBound Upper bound of the specified range.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsCountInRangeGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus
nppsCountInRange_32s_Ctx(const Npp32s * pSrc, int nLength, int * pCounts, Npp32s nLowerBound, Npp32s nUpperBound,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus
nppsCountInRange_32s(const Npp32s * pSrc, int nLength, int * pCounts, Npp32s nLowerBound, Npp32s nUpperBound,
                     Npp8u * pDeviceBuffer);

/** @} signal_count_in_range */

/** @defgroup signal_count_zero_crossings Count Zero Crossings
 *
 * @{
 * Calculates the number of zero crossings in a signal.
 */

/** 
 * Device-buffer size (in bytes) for nppsZeroCrossing_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsZeroCrossingGetBufferSize_16s32f_Ctx(int nLength, int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsZeroCrossingGetBufferSize_16s32f(int nLength, int * hpBufferSize);

/** 
 * 16-bit signed short integer zero crossing method, return value is 32-bit floating point.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pValZC Pointer to the output result.
 * \param tZCType Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsZeroCrossingGetBufferSize_16s32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus
nppsZeroCrossing_16s32f_Ctx(const Npp16s * pSrc, int nLength, Npp32f * pValZC, NppsZCType tZCType,
                            Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus
nppsZeroCrossing_16s32f(const Npp16s * pSrc, int nLength, Npp32f * pValZC, NppsZCType tZCType,
                        Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsZeroCrossing_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsZeroCrossingGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsZeroCrossingGetBufferSize_32f(int nLength, int * hpBufferSize);

/** 
 * 32-bit floating-point zero crossing method, return value is 32-bit floating point.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pValZC Pointer to the output result.
 * \param tZCType Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsZeroCrossingGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus
nppsZeroCrossing_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pValZC, NppsZCType tZCType,
                         Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus
nppsZeroCrossing_32f(const Npp32f * pSrc, int nLength, Npp32f * pValZC, NppsZCType tZCType,
                     Npp8u * pDeviceBuffer);

/** @} signal_count_zero_crossings */

/** @defgroup signal_maximum_error MaximumError
 * Primitives for computing the maximum error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the maximum error is defined as the largest absolute difference between the corresponding
 * elements of two signals.
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 8-bit signed char maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex maximum method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_8u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_8u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_8s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_8s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_16u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_16u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_16s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_16s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_16sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_16sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_32u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_32u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_32sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_32sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_64s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_64s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_64sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_64sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_32fc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumErrorGetBufferSize_64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumErrorGetBufferSize_64fc(int nLength,  int * hpBufferSize);

/** @} */

/** @defgroup signal_average_error AverageError
 * Primitives for computing the Average error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the average error is defined as
 * \f[Average Error = \frac{1}{N}\sum_{n=0}^{N-1}\left|pSrc1(n) - pSrc2(n)\right|\f]
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 8-bit signed char Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex Average method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_8u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_8u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_8s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_8s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_16u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_16u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_16s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_16s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_16sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_16sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_32u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_32u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_32sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_32sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_64s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_64s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_64sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_64sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_32fc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageErrorGetBufferSize_64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageErrorGetBufferSize_64fc(int nLength,  int * hpBufferSize);

/** @} */

/** @defgroup signal_maximum_relative_error MaximumRelativeError
 * Primitives for computing the MaximumRelative error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the maximum relative error is defined as
 * \f[MaximumRelativeError = max{\frac{\left|pSrc1(n) - pSrc2(n)\right|}{max(\left|pSrc1(n)\right|, \left|pSrc2(n)\right|)}}\f]
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 8-bit signed char MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex MaximumRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsMaximumRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMaximumRelativeError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsMaximumRelativeError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_8u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_8u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_8s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_8s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_16u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_16s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_16sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_32u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_32sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_64s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_64sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_32fc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsMaximumRelativeError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsMaximumRelativeErrorGetBufferSize_64fc(int nLength,  int * hpBufferSize);

/** @} */

/** @defgroup signal_average_relative_error AverageRelativeError
 * Primitives for computing the AverageRelative error between two signals.
 * Given two signals \f$pSrc1\f$ and \f$pSrc2\f$ both with length \f$N\f$, 
 * the average relative error is defined as
 * \f[AverageRelativeError = \frac{1}{N}\sum_{n=0}^{N-1}\frac{\left|pSrc1(n) - pSrc2(n)\right|}{max(\left|pSrc1(n)\right|, \left|pSrc2(n)\right|)}\f]
 *
 * If the signal is in complex format, the absolute value of the complex number is used.
 * @{
 *
 */
/** 
 * 8-bit unsigned char AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_8u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 8-bit signed char AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_8s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_8s_Ctx(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_8s(const Npp8s * pSrc1, const Npp8s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_16u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_16s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 16-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_16sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_16sc_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_16sc(const Npp16sc * pSrc1, const Npp16sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_32u to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_32s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_32s_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_32s(const Npp32s * pSrc1, const Npp32s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_32sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_32sc_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_32sc(const Npp32sc * pSrc1, const Npp32sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit signed short integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_64s to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_64s_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_64s(const Npp64s * pSrc1, const Npp64s * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit unsigned short complex integer AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_64sc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_64sc_Ctx(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_64sc(const Npp64sc * pSrc1, const Npp64sc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_32f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 32-bit floating point complex AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_32fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_64f to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * 64-bit floating point complex AverageRelative method.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the error result.
 * \param pDeviceBuffer Pointer to the required device memory allocation, \ref general_scratch_buffer. 
 *        Use \ref nppsAverageRelativeErrorGetBufferSize_64fc to determine the minium number of bytes required.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAverageRelativeError_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsAverageRelativeError_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_8u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_8u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_8u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_8s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_8s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_8s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_16u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_16u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_16s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_16s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_16sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_16sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_32u.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32u_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_32u(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_32s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_32s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_32sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_32sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_64s.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64s_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_64s(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_64sc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64sc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_64sc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_32f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_32fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_32fc(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64f_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_64f(int nLength,  int * hpBufferSize);

/** 
 * Device-buffer size (in bytes) for nppsAverageRelativeError_64fc.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size.  Important: 
 *        hpBufferSize is a <em>host pointer.</em>
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64fc_Ctx(int nLength,  int * hpBufferSize, NppStreamContext nppStreamCtx);

NppStatus
nppsAverageRelativeErrorGetBufferSize_64fc(int nLength,  int * hpBufferSize);

/** @} */


/** @} signal_statistical_functions */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPS_STATISTICS_FUNCTIONS_H */
