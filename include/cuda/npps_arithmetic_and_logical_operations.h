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
#ifndef NV_NPPS_ARITHMETIC_AND_LOGICAL_OPERATIONS_H
#define NV_NPPS_ARITHMETIC_AND_LOGICAL_OPERATIONS_H
 
/**
 * \file npps_arithmetic_and_logical_operations.h
 * Signal Arithmetic and Logical Operations.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** 
 * @defgroup signal_arithmetic_and_logical_operations Arithmetic and Logical Operations
 * @ingroup npps
 * Functions that provide common arithmetic and logical operations.
 * @{
 *
 */

/** 
 * @defgroup signal_arithmetic Arithmetic Operations
 * The set of arithmetic operations for signal processing available in the library.
 * @{
 *
 */

/** 
 * @defgroup signal_addc AddC
 * Adds a constant value to each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal add constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_8u_ISfs_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_8u_ISfs(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned charvector add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_8u_Sfs(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal add constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short vector add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_16u_Sfs(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place  signal add constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal add constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_16s_Sfs(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal add constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal add constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_16sc_Sfs(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal add constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integersignal add constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32s_Sfs(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * add constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal add constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32sc_Sfs_Ctx(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32sc_Sfs(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32f_I_Ctx(Npp32f nValue, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32f_I(Npp32f nValue, Npp32f * pSrcDst, int nLength);

/** 
 * 32-bit floating point signal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32fc_I_Ctx(Npp32fc nValue, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32fc_I(Npp32fc nValue, Npp32fc * pSrcDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_32fc_Ctx(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_32fc(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength);

/** 
 * 64-bit floating point, in place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength Length of the vectors, number of items.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_64f_I(Npp64f nValue, Npp64f * pSrcDst, int nLength);

/** 
 * 64-bit floating pointsignal add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_64f_Ctx(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_64f(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal add constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * add constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be added to each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddC_64fc_Ctx(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddC_64fc(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength);

/** @} signal_addc */

/** 
 * @defgroup signal_addproductc AddProductC
 * Adds product of a constant and each sample of a source signal to the each sample of destination signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal add product of signal times constant to destination signal.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProductC_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProductC_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** @} signal_addproductc */

/** 
 * @defgroup signal_mulc MulC
 *
 * Multiplies each sample of a signal by a constant value.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal times constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_8u_ISfs_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_8u_ISfs(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_8u_Sfs(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal times constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_16u_Sfs(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal times constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal times constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_16s_Sfs(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_16sc_Sfs(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal times constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32s_Sfs(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * times constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal times constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32sc_Sfs_Ctx(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32sc_Sfs(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32f_I_Ctx(Npp32f nValue, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32f_I(Npp32f nValue, Npp32f * pSrcDst, int nLength);

/** 
 * 32-bit floating point signal times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point signal times constant with output converted to 16-bit signed integer.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_Low_32f16s_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_Low_32f16s(const Npp32f * pSrc, Npp32f nValue, Npp16s * pDst, int nLength);

/** 
 * 32-bit floating point signal times constant with output converted to 16-bit signed integer
 * with scaling and saturation of output result.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32f16s_Sfs_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32f16s_Sfs(const Npp32f * pSrc, Npp32f nValue, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32fc_I_Ctx(Npp32fc nValue, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32fc_I(Npp32fc nValue, Npp32fc * pSrcDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_32fc_Ctx(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_32fc(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength);

/** 
 * 64-bit floating point, in place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength Length of the vectors, number of items.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_64f_I(Npp64f nValue, Npp64f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_64f_Ctx(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_64f(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength);

/** 
 * 64-bit floating point signal times constant with in place conversion to 64-bit signed integer
 * and with scaling and saturation of output result.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_64f64s_ISfs_Ctx(Npp64f nValue, Npp64s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_64f64s_ISfs(Npp64f nValue, Npp64s * pDst, int nLength, int nScaleFactor);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal times constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * times constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be multiplied by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMulC_64fc_Ctx(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMulC_64fc(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength);

/** @} signal_mulc */

/** 
 * @defgroup signal_subc SubC
 *
 * Subtracts a constant from each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal subtract constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_8u_ISfs_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_8u_ISfs(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_8u_Sfs(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal subtract constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_16u_Sfs(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal subtract constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_16s_Sfs(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_16sc_Sfs(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal subtract constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal subtract constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32s_Sfs(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * subtract constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary)signal subtract constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32sc_Sfs_Ctx(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32sc_Sfs(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32f_I_Ctx(Npp32f nValue, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32f_I(Npp32f nValue, Npp32f * pSrcDst, int nLength);

/** 
 * 32-bit floating point signal subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32fc_I_Ctx(Npp32fc nValue, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32fc_I(Npp32fc nValue, Npp32fc * pSrcDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_32fc_Ctx(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_32fc(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength);

/** 
 * 64-bit floating point, in place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength Length of the vectors, number of items.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_64f_I(Npp64f nValue, Npp64f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_64f_Ctx(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_64f(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal subtract constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * subtract constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be subtracted from each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubC_64fc_Ctx(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubC_64fc(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength);

/** @} signal_subc */

/** 
 * @defgroup signal_subcrev SubCRev
 *
 * Subtracts each sample of a signal from a constant.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal subtract from constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_8u_ISfs_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_8u_ISfs(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_8u_Sfs(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_16u_Sfs(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal subtract from constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_16s_Sfs(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_16sc_Sfs(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal subtract from constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32s_ISfs_Ctx(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32s_ISfs(Npp32s nValue, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integersignal subtract from constant and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32s_Sfs(const Npp32s * pSrc, Npp32s nValue, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
 * subtract from constant and scale.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32sc_ISfs(Npp32sc nValue, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit integer complex number (32 bit real, 32 bit imaginary) signal subtract from constant
 * and scale.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32sc_Sfs_Ctx(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32sc_Sfs(const Npp32sc * pSrc, Npp32sc nValue, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32f_I_Ctx(Npp32f nValue, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32f_I(Npp32f nValue, Npp32f * pSrcDst, int nLength);

/** 
 * 32-bit floating point signal subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32fc_I_Ctx(Npp32fc nValue, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32fc_I(Npp32fc nValue, Npp32fc * pSrcDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_32fc_Ctx(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_32fc(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength);

/** 
 * 64-bit floating point, in place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength Length of the vectors, number of items.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_64f_I(Npp64f nValue, Npp64f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_64f_Ctx(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_64f(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal subtract from constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * subtract from constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value each vector element is to be subtracted from
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSubCRev_64fc_Ctx(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSubCRev_64fc(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength);

/** @} signal_subcrev */

/** 
 * @defgroup signal_divc DivC
 *
 * Divides each sample of a signal by a constant.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char in place signal divided by constant,
 * scale, then clamp to saturated value
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_8u_ISfs_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_8u_ISfs(Npp8u nValue, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_8u_Sfs(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal divided by constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_16u_ISfs_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_16u_ISfs(Npp16u nValue, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_16u_Sfs(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal divided by constant, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_16s_ISfs_Ctx(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_16s_ISfs(Npp16s nValue, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal divided by constant, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_16s_Sfs(const Npp16s * pSrc, Npp16s nValue, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary)signal divided by constant, 
 * scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_16sc_ISfs(Npp16sc nValue, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit integer complex number (16 bit real, 16 bit imaginary) signal divided by constant,
 * scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_16sc_Sfs(const Npp16sc * pSrc, Npp16sc nValue, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_32f_I_Ctx(Npp32f nValue, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_32f_I(Npp32f nValue, Npp32f * pSrcDst, int nLength);

/** 
 * 32-bit floating point signal divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
 * place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_32fc_I_Ctx(Npp32fc nValue, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_32fc_I(Npp32fc nValue, Npp32fc * pSrcDst, int nLength);

/** 
 * 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
 * divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_32fc_Ctx(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_32fc(const Npp32fc * pSrc, Npp32fc nValue, Npp32fc * pDst, int nLength);

/** 
 * 64-bit floating point in place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength Length of the vectors, number of items.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_64f_I_Ctx(Npp64f nValue, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_64f_I(Npp64f nValue, Npp64f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_64f_Ctx(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_64f(const Npp64f * pSrc, Npp64f nValue, Npp64f * pDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
 * place signal divided by constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_64fc_I_Ctx(Npp64fc nValue, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_64fc_I(Npp64fc nValue, Npp64fc * pSrcDst, int nLength);

/** 
 * 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
 * divided by constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided into each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivC_64fc_Ctx(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivC_64fc(const Npp64fc * pSrc, Npp64fc nValue, Npp64fc * pDst, int nLength);

/** @} signal_divc */

/** 
 * @defgroup signal_divcrev DivCRev
 *
 * Divides a constant by each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 16-bit unsigned short in place constant divided by signal, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivCRev_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivCRev_16u_I(Npp16u nValue, Npp16u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short signal divided by constant, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivCRev_16u_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivCRev_16u(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength);

/** 
 * 32-bit floating point in place constant divided by signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivCRev_32f_I_Ctx(Npp32f nValue, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivCRev_32f_I(Npp32f nValue, Npp32f * pSrcDst, int nLength);

/** 
 * 32-bit floating point constant divided by signal.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be divided by each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDivCRev_32f_Ctx(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDivCRev_32f(const Npp32f * pSrc, Npp32f nValue, Npp32f * pDst, int nLength);

/** @} signal_divcrev */

/** 
 * @defgroup signal_add Add
 *
 * Sample by sample addition of two signals.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength);

/** 
 * 16-bit unsigned short signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned int signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength);

/** 
 * 32-bit floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal add signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength);

/** 
 * 8-bit unsigned char signal add signal with 16-bit unsigned result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_8u16u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_8u16u(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp16u * pDst, int nLength);

/** 
 * 16-bit signed short signal add signal with 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be added to signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 8-bit unsigned char add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_8u_Sfs_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_8u_Sfs(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16u_Sfs_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16u_Sfs(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 64-bit signed integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_64s_Sfs_Ctx(const Npp64s * pSrc1, const Npp64s * pSrc2, Npp64s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_64s_Sfs(const Npp64s * pSrc1, const Npp64s * pSrc2, Npp64s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16sc_Sfs_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16sc_Sfs(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed complex integer add signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be added to signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32sc_Sfs_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32sc_Sfs(const Npp32sc * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16s_I_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16s_I(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32f_I_Ctx(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32f_I(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_64f_I_Ctx(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_64f_I(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32fc_I_Ctx(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32fc_I(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength);

/** 
 * 64-bit complex floating point in place signal add signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_64fc_I_Ctx(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_64fc_I(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength);

/** 
 * 16/32-bit signed short in place signal add signal with 32-bit signed integer results,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16s32s_I_Ctx(const Npp16s * pSrc, Npp32s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16s32s_I(const Npp16s * pSrc, Npp32s * pSrcDst, int nLength);

/** 
 * 8-bit unsigned char in place signal add signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_8u_ISfs_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_8u_ISfs(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal add signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16u_ISfs_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16u_ISfs(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16s_ISfs_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16s_ISfs(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32s_ISfs_Ctx(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32s_ISfs(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_16sc_ISfs_Ctx(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_16sc_ISfs(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal add signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be added to signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAdd_32sc_ISfs_Ctx(const Npp32sc * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAdd_32sc_ISfs(const Npp32sc * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_add */

/** 
 * @defgroup signal_addproduct AddProduct
 *
 * Adds sample by sample product of two signals to the destination signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal add product of source signal times destination signal to destination signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength);

/** 
 * 16-bit signed short signal add product of source signal1 times source signal2 to destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed short signal add product of source signal1 times source signal2 to destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal add product of source signal1 times source signal2 to 32-bit signed integer destination signal,
 * with scaling, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAddProduct_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsAddProduct_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** @} signal_addproduct */

/** 
 * @defgroup signal_mul Mul
 *
 * Sample by sample multiplication the samples of two signals.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength);

/** 
 * 32-bit floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal times signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength);

/** 
 * 8-bit unsigned char signal times signal with 16-bit unsigned result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_8u16u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_8u16u(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp16u * pDst, int nLength);

/** 
 * 16-bit signed short signal times signal with 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point signal times 32-bit complex floating point signal with complex 32-bit floating point result,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32f32fc_Ctx(const Npp32f * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32f32fc(const Npp32f * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength);

/** 
 * 8-bit unsigned char signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_8u_Sfs_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_8u_Sfs(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal time signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16u_Sfs_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16u_Sfs(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16sc_Sfs_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16sc_Sfs(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed complex integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32sc_Sfs_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32sc_Sfs(const Npp32sc * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal times 16-bit signed short signal, scale, then clamp to 16-bit signed saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16u16s_Sfs_Ctx(const Npp16u * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16u16s_Sfs(const Npp16u * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal times signal, scale, then clamp to 32-bit signed saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16s32s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16s32s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times 32-bit complex signed integer signal, scale, then clamp to 32-bit complex integer saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32s32sc_Sfs_Ctx(const Npp32s * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32s32sc_Sfs(const Npp32s * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal times signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal2 elements to be multiplied by signal1 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_Low_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_Low_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16s_I_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16s_I(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32f_I_Ctx(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32f_I(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_64f_I_Ctx(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_64f_I(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32fc_I_Ctx(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32fc_I(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength);

/** 
 * 64-bit complex floating point in place signal times signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_64fc_I_Ctx(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_64fc_I(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point in place signal times 32-bit floating point signal,
 * then clamp to 32-bit complex floating point saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32f32fc_I_Ctx(const Npp32f * pSrc, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32f32fc_I(const Npp32f * pSrc, Npp32fc * pSrcDst, int nLength);

/** 
 * 8-bit unsigned char in place signal times signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_8u_ISfs_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_8u_ISfs(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal times signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16u_ISfs_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16u_ISfs(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16s_ISfs_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16s_ISfs(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32s_ISfs_Ctx(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32s_ISfs(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_16sc_ISfs_Ctx(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_16sc_ISfs(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal times signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32sc_ISfs_Ctx(const Npp32sc * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32sc_ISfs(const Npp32sc * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal times 32-bit signed integer signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements
 * \param nLength \ref length_specification. 
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsMul_32s32sc_ISfs_Ctx(const Npp32s * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsMul_32s32sc_ISfs(const Npp32s * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_mul */

/** 
 * @defgroup signal_sub Sub
 *
 * Sample by sample subtraction of the samples of two signals.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16s_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16s(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength);

/** 
 * 32-bit floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength);

/** 
 * 16-bit signed short signal subtract 16-bit signed short signal,
 * then clamp and convert to 32-bit floating point saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16s32f_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16s32f(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 8-bit unsigned char signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_8u_Sfs_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_8u_Sfs(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16u_Sfs_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16u_Sfs(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16sc_Sfs_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16sc_Sfs(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed complex integer signal subtract signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 elements to be subtracted from signal2 elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32sc_Sfs_Ctx(const Npp32sc * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32sc_Sfs(const Npp32sc * pSrc1, const Npp32sc * pSrc2, Npp32sc * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16s_I_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16s_I(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32f_I_Ctx(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32f_I(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_64f_I_Ctx(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_64f_I(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32fc_I_Ctx(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32fc_I(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength);

/** 
 * 64-bit complex floating point in place signal subtract signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_64fc_I_Ctx(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_64fc_I(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength);

/** 
 * 8-bit unsigned char in place signal subtract signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_8u_ISfs_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_8u_ISfs(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal subtract signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16u_ISfs_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16u_ISfs(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16s_ISfs_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16s_ISfs(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32s_ISfs_Ctx(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32s_ISfs(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_16sc_ISfs_Ctx(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_16sc_ISfs(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit complex signed integer in place signal subtract signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSub_32sc_ISfs_Ctx(const Npp32sc * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSub_32sc_ISfs(const Npp32sc * pSrc, Npp32sc * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_sub */

/**
 * @defgroup signal_div Div
 *
 * Sample by sample division of the samples of two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_8u_Sfs_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_8u_Sfs(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_16u_Sfs_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_16u_Sfs(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32s_Sfs_Ctx(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32s_Sfs(const Npp32s * pSrc1, const Npp32s * pSrc2, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed complex short signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_16sc_Sfs_Ctx(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_16sc_Sfs(const Npp16sc * pSrc1, const Npp16sc * pSrc2, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal divided by 16-bit signed short signal, scale, then clamp to 16-bit signed short saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32s16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp32s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32s16s_Sfs(const Npp16s * pSrc1, const Npp32s * pSrc2, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32f_Ctx(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32f(const Npp32f * pSrc1, const Npp32f * pSrc2, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_64f_Ctx(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_64f(const Npp64f * pSrc1, const Npp64f * pSrc2, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32fc_Ctx(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32fc(const Npp32fc * pSrc1, const Npp32fc * pSrc2, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal divide signal,
 * then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_64fc_Ctx(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_64fc(const Npp64fc * pSrc1, const Npp64fc * pSrc2, Npp64fc * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal divide signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_8u_ISfs_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_8u_ISfs(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal divide signal, with scaling,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_16u_ISfs_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_16u_ISfs(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_16s_ISfs_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_16s_ISfs(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_16sc_ISfs_Ctx(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_16sc_ISfs(const Npp16sc * pSrc, Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer in place signal divide signal, with scaling, 
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32s_ISfs_Ctx(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32s_ISfs(const Npp32s * pSrc, Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32f_I_Ctx(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32f_I(const Npp32f * pSrc, Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_64f_I_Ctx(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_64f_I(const Npp64f * pSrc, Npp64f * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_32fc_I_Ctx(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_32fc_I(const Npp32fc * pSrc, Npp32fc * pSrcDst, int nLength);

/** 
 * 64-bit complex floating point in place signal divide signal,
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_64fc_I_Ctx(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_64fc_I(const Npp64fc * pSrc, Npp64fc * pSrcDst, int nLength);

/** @} signal_div */

/** 
 * @defgroup signal_divround Div_Round
 *
 * Sample by sample division of the samples of two signals with rounding.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_Round_8u_Sfs_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_Round_8u_Sfs(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, NppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit unsigned short signal divide signal, scale, round, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_Round_16u_Sfs_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_Round_16u_Sfs(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, NppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit signed short signal divide signal, scale, round, then clamp to saturated value.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_Round_16s_Sfs_Ctx(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_Round_16s_Sfs(const Npp16s * pSrc1, const Npp16s * pSrc2, Npp16s * pDst, int nLength, NppRoundMode nRndMode, int nScaleFactor);

/** 
 * 8-bit unsigned char in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_Round_8u_ISfs_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_Round_8u_ISfs(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit unsigned short in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_Round_16u_ISfs_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_Round_16u_ISfs(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppRoundMode nRndMode, int nScaleFactor);

/** 
 * 16-bit signed short in place signal divide signal, with scaling, rounding
 * then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements
 * \param nLength \ref length_specification.
 * \param nRndMode various rounding modes.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsDiv_Round_16s_ISfs_Ctx(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppRoundMode nRndMode, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsDiv_Round_16s_ISfs(const Npp16s * pSrc, Npp16s * pSrcDst, int nLength, NppRoundMode nRndMode, int nScaleFactor);

/** @} signal_divround */

/** 
 * @defgroup signal_abs Abs
 *
 * Absolute value of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 16-bit signed short signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength);

/** 
 * 32-bit signed integer signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_32s_Ctx(const Npp32s * pSrc, Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_32s(const Npp32s * pSrc, Npp32s * pDst, int nLength);

/** 
 * 32-bit floating point signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal absolute value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 16-bit signed short signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_16s_I_Ctx(Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_16s_I(Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit signed integer signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_32s_I_Ctx(Npp32s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_32s_I(Npp32s * pSrcDst, int nLength);

/** 
 * 32-bit floating point signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_32f_I_Ctx(Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_32f_I(Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal absolute value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAbs_64f_I_Ctx(Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAbs_64f_I(Npp64f * pSrcDst, int nLength);

/** @} signal_abs */

/** 
 * @defgroup signal_square Sqr
 *
 * Squares each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal squared.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength);

/** 
 * 32-bit floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_32f_I_Ctx(Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_32f_I(Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_64f_I_Ctx(Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_64f_I(Npp64f * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_32fc_I(Npp32fc * pSrcDst, int nLength);

/** 
 * 64-bit complex floating point signal squared.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_64fc_I(Npp64fc * pSrcDst, int nLength);

/** 
 * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_8u_Sfs(const Npp8u * pSrc, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_16u_Sfs(const Npp16u * pSrc, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_16s_Sfs(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_16sc_Sfs(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_8u_ISfs_Ctx(Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_8u_ISfs(Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_16u_ISfs_Ctx(Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_16u_ISfs(Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_16s_ISfs_Ctx(Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_16s_ISfs(Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal squared, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqr_16sc_ISfs_Ctx(Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqr_16sc_ISfs(Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_square */

/** 
 * @defgroup signal_sqrt Sqrt
 *
 * Square root of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 32-bit complex floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex floating point signal square root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength);

/** 
 * 32-bit floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_32f_I_Ctx(Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_32f_I(Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64f_I_Ctx(Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64f_I(Npp64f * pSrcDst, int nLength);

/** 
 * 32-bit complex floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_32fc_I_Ctx(Npp32fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_32fc_I(Npp32fc * pSrcDst, int nLength);

/** 
 * 64-bit complex floating point signal square root.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64fc_I_Ctx(Npp64fc * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64fc_I(Npp64fc * pSrcDst, int nLength);

/** 
 * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_8u_Sfs_Ctx(const Npp8u * pSrc, Npp8u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_8u_Sfs(const Npp8u * pSrc, Npp8u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_16u_Sfs_Ctx(const Npp16u * pSrc, Npp16u * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_16u_Sfs(const Npp16u * pSrc, Npp16u * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_16s_Sfs(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_16sc_Sfs(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal square root, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64s_Sfs_Ctx(const Npp64s * pSrc, Npp64s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64s_Sfs(const Npp64s * pSrc, Npp64s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_32s16s_Sfs_Ctx(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_32s16s_Sfs(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64s16s_Sfs_Ctx(const Npp64s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64s16s_Sfs(const Npp64s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 8-bit unsigned char signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_8u_ISfs_Ctx(Npp8u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_8u_ISfs(Npp8u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit unsigned short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_16u_ISfs_Ctx(Npp16u * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_16u_ISfs(Npp16u * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_16s_ISfs_Ctx(Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_16s_ISfs(Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 16-bit complex signed short signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_16sc_ISfs_Ctx(Npp16sc * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_16sc_ISfs(Npp16sc * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal square root, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSqrt_64s_ISfs_Ctx(Npp64s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsSqrt_64s_ISfs(Npp64s * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_sqrt */

/** 
 * @defgroup signal_cuberoot Cubrt
 *
 * Cube root of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal cube root.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCubrt_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCubrt_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 32-bit signed integer signal cube root, scale, then clamp to 16-bit signed integer saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCubrt_32s16s_Sfs_Ctx(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsCubrt_32s16s_Sfs(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** @} signal_cuberoot */

/** 
 * @defgroup signal_exp Exp
 *
 * E raised to the power of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal exponent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal exponent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 32-bit floating point signal exponent with 64-bit floating point result.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_32f64f_Ctx(const Npp32f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_32f64f(const Npp32f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 32-bit floating point signal exponent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_32f_I_Ctx(Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_32f_I(Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal exponent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_64f_I_Ctx(Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_64f_I(Npp64f * pSrcDst, int nLength);

/** 
 * 16-bit signed short signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_16s_Sfs(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_32s_Sfs(const Npp32s * pSrc, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_64s_Sfs_Ctx(const Npp64s * pSrc, Npp64s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_64s_Sfs(const Npp64s * pSrc, Npp64s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_16s_ISfs_Ctx(Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_16s_ISfs(Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_32s_ISfs_Ctx(Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_32s_ISfs(Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 64-bit signed integer signal exponent, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsExp_64s_ISfs_Ctx(Npp64s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsExp_64s_ISfs(Npp64s * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_exp */

/** 
 * @defgroup signal_ln Ln
 *
 * Natural logarithm of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 64-bit floating point signal natural logarithm with 32-bit floating point result.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_64f32f_Ctx(const Npp64f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_64f32f(const Npp64f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 32-bit floating point signal natural logarithm.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_32f_I_Ctx(Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_32f_I(Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal natural logarithm.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_64f_I_Ctx(Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_64f_I(Npp64f * pSrcDst, int nLength);

/** 
 * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_16s_Sfs(const Npp16s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_32s_Sfs(const Npp32s * pSrc, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to 16-bit signed short saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_32s16s_Sfs_Ctx(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_32s16s_Sfs(const Npp32s * pSrc, Npp16s * pDst, int nLength, int nScaleFactor);

/** 
 * 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_16s_ISfs_Ctx(Npp16s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_16s_ISfs(Npp16s * pSrcDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLn_32s_ISfs_Ctx(Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsLn_32s_ISfs(Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_ln */

/** 
 * @defgroup signal_10log10 10Log10
 *
 * Ten times the decimal logarithm of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
npps10Log10_32s_Sfs_Ctx(const Npp32s * pSrc, Npp32s * pDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
npps10Log10_32s_Sfs(const Npp32s * pSrc, Npp32s * pDst, int nLength, int nScaleFactor);

/** 
 * 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
npps10Log10_32s_ISfs_Ctx(Npp32s * pSrcDst, int nLength, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
npps10Log10_32s_ISfs(Npp32s * pSrcDst, int nLength, int nScaleFactor);

/** @} signal_10log10 */

/** 
 * @defgroup signal_sumln SumLn
 *
 * Sums up the natural logarithm of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * Device scratch buffer size (in bytes) for 32f SumLn.
 * This primitive provides the correct buffer size for nppsSumLn_32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumLnGetBufferSize_32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLnGetBufferSize_32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 32-bit floating point signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSumLn_32f_Ctx(const Npp32f * pSrc, int nLength, Npp32f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLn_32f(const Npp32f * pSrc, int nLength, Npp32f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for 64f SumLn.
 * This primitive provides the correct buffer size for nppsSumLn_64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumLnGetBufferSize_64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLnGetBufferSize_64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 64-bit floating point signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSumLn_64f_Ctx(const Npp64f * pSrc, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLn_64f(const Npp64f * pSrc, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for 32f64f SumLn.
 * This primitive provides the correct buffer size for nppsSumLn_32f64f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumLnGetBufferSize_32f64f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLnGetBufferSize_32f64f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 32-bit flaoting point input, 64-bit floating point output signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSumLn_32f64f_Ctx(const Npp32f * pSrc, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLn_32f64f(const Npp32f * pSrc, int nLength, Npp64f * pDst, Npp8u * pDeviceBuffer);

/** 
 * Device scratch buffer size (in bytes) for 16s32f SumLn.
 * This primitive provides the correct buffer size for nppsSumLn_16s32f.
 * \param nLength \ref length_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return NPP_SUCCESS
 */
NppStatus 
nppsSumLnGetBufferSize_16s32f_Ctx(int nLength, int * hpBufferSize /* host pointer */, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLnGetBufferSize_16s32f(int nLength, int * hpBufferSize /* host pointer */);

/** 
 * 16-bit signed short integer input, 32-bit floating point output signal sum natural logarithm.
 * \param pSrc \ref source_signal_pointer.
 * \param nLength \ref length_specification.
 * \param pDst Pointer to the output result.
 * \param pDeviceBuffer Pointer to the required device memory allocation.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSumLn_16s32f_Ctx(const Npp16s * pSrc, int nLength, Npp32f * pDst, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);

NppStatus 
nppsSumLn_16s32f(const Npp16s * pSrc, int nLength, Npp32f * pDst, Npp8u * pDeviceBuffer);

/** @} signal_sumln */

/** 
 * @defgroup signal_inversetan Arctan
 *
 * Inverse tangent of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal inverse tangent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsArctan_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsArctan_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit floating point signal inverse tangent.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsArctan_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsArctan_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength);

/** 
 * 32-bit floating point signal inverse tangent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsArctan_32f_I_Ctx(Npp32f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsArctan_32f_I(Npp32f * pSrcDst, int nLength);

/** 
 * 64-bit floating point signal inverse tangent.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsArctan_64f_I_Ctx(Npp64f * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsArctan_64f_I(Npp64f * pSrcDst, int nLength);

/** @} signal_inversetan */

/** 
 * @defgroup signal_normalize Normalize
 *
 * Normalize each sample of a real or complex signal using offset and division operations.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormalize_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f vSub, Npp32f vDiv, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormalize_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength, Npp32f vSub, Npp32f vDiv);

/** 
 * 32-bit complex floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormalize_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32fc vSub, Npp32f vDiv, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormalize_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, Npp32fc vSub, Npp32f vDiv);

/** 
 * 64-bit floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormalize_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f vSub, Npp64f vDiv, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormalize_64f(const Npp64f * pSrc, Npp64f * pDst, int nLength, Npp64f vSub, Npp64f vDiv);

/** 
 * 64-bit complex floating point signal normalize.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormalize_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64fc vSub, Npp64f vDiv, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormalize_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, Npp64fc vSub, Npp64f vDiv);

/** 
 * 16-bit signed short signal normalize, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormalize_16s_Sfs_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s vSub, int vDiv, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormalize_16s_Sfs(const Npp16s * pSrc, Npp16s * pDst, int nLength, Npp16s vSub, int vDiv, int nScaleFactor);

/** 
 * 16-bit complex signed short signal normalize, scale, then clamp to saturated value.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param vSub value subtracted from each signal element before division
 * \param vDiv divisor of post-subtracted signal element dividend
 * \param nScaleFactor \ref integer_result_scaling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNormalize_16sc_Sfs_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16sc vSub, int vDiv, int nScaleFactor, NppStreamContext nppStreamCtx);

NppStatus 
nppsNormalize_16sc_Sfs(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, Npp16sc vSub, int vDiv, int nScaleFactor);

/** @} signal_normalize */

/** 
 * @defgroup signal_cauchy Cauchy, CauchyD, and CauchyDD2
 *
 * Determine Cauchy robust error function and its first and second derivatives for each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 32-bit floating point signal Cauchy error calculation.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCauchy_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nParam, NppStreamContext nppStreamCtx);

NppStatus 
nppsCauchy_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nParam);

/** 
 * 32-bit floating point signal Cauchy first derivative.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCauchyD_32f_I_Ctx(Npp32f * pSrcDst, int nLength, Npp32f nParam, NppStreamContext nppStreamCtx);

NppStatus 
nppsCauchyD_32f_I(Npp32f * pSrcDst, int nLength, Npp32f nParam);

/** 
 * 32-bit floating point signal Cauchy first and second derivatives.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param pD2FVal \ref source_signal_pointer. This signal contains the second derivative
 *      of the source signal.
 * \param nLength \ref length_specification.
 * \param nParam constant used in Cauchy formula
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCauchyDD2_32f_I_Ctx(Npp32f * pSrcDst, Npp32f * pD2FVal, int nLength, Npp32f nParam, NppStreamContext nppStreamCtx);

NppStatus 
nppsCauchyDD2_32f_I(Npp32f * pSrcDst, Npp32f * pD2FVal, int nLength, Npp32f nParam);

/** @} signal_cauchy */

/** @} signal_arithmetic_operations */

/** 
 * @defgroup signal_logical_and_shift_operations Logical And Shift Operations
 * The set of logical and shift operations for signal processing available in the library.
 * @{
 *
 */

/** 
 * @defgroup signal_andc AndC
 *
 * Bitwise AND of a constant and each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAndC_8u_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAndC_8u(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAndC_16u_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAndC_16u(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal and with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAndC_32u_Ctx(const Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAndC_32u(const Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAndC_8u_I_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAndC_8u_I(Npp8u nValue, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAndC_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAndC_16u_I(Npp16u nValue, Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned signed integer in place signal and with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be anded with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAndC_32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAndC_32u_I(Npp32u nValue, Npp32u * pSrcDst, int nLength);

/** @} signal_andc */

/** 
 * @defgroup signal_and And
 *
 * Sample by sample bitwise AND of samples from two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAnd_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAnd_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAnd_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAnd_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal and with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAnd_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAnd_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAnd_8u_I_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAnd_8u_I(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAnd_16u_I_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAnd_16u_I(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned integer in place signal and with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be anded with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsAnd_32u_I_Ctx(const Npp32u * pSrc, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsAnd_32u_I(const Npp32u * pSrc, Npp32u * pSrcDst, int nLength);

/** @} signal_and */

/** 
 * @defgroup signal_orc OrC
 *
 * Bitwise OR of a constant and each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOrC_8u_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOrC_8u(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOrC_16u_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOrC_16u(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOrC_32u_Ctx(const Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOrC_32u(const Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOrC_8u_I_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOrC_8u_I(Npp8u nValue, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOrC_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOrC_16u_I(Npp16u nValue, Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned signed integer in place signal or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be ored with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOrC_32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOrC_32u_I(Npp32u nValue, Npp32u * pSrcDst, int nLength);

/** @} signal_orc */

/** 
 * @defgroup signal_or Or
 *
 * Sample by sample bitwise OR of the samples from two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOr_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOr_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOr_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOr_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOr_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOr_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOr_8u_I_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOr_8u_I(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOr_16u_I_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOr_16u_I(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned integer in place signal or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsOr_32u_I_Ctx(const Npp32u * pSrc, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsOr_32u_I(const Npp32u * pSrc, Npp32u * pSrcDst, int nLength);

/** @} signal_or */

/** 
 * @defgroup signal_xorc XorC
 *
 * Bitwise XOR of a constant and each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXorC_8u_Ctx(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXorC_8u(const Npp8u * pSrc, Npp8u nValue, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXorC_16u_Ctx(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXorC_16u(const Npp16u * pSrc, Npp16u nValue, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal exclusive or with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXorC_32u_Ctx(const Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXorC_32u(const Npp32u * pSrc, Npp32u nValue, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXorC_8u_I_Ctx(Npp8u nValue, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXorC_8u_I(Npp8u nValue, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXorC_16u_I_Ctx(Npp16u nValue, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXorC_16u_I(Npp16u nValue, Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned signed integer in place signal exclusive or with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be exclusive ored with each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXorC_32u_I_Ctx(Npp32u nValue, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXorC_32u_I(Npp32u nValue, Npp32u * pSrcDst, int nLength);

/** @} signal_xorc */

/**
 * @defgroup signal_xor Xor
 *
 * Sample by sample bitwise XOR of the samples from two signals.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXor_8u_Ctx(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXor_8u(const Npp8u * pSrc1, const Npp8u * pSrc2, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXor_16u_Ctx(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXor_16u(const Npp16u * pSrc1, const Npp16u * pSrc2, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal exclusive or with signal.
 * \param pSrc1 \ref source_signal_pointer.
 * \param pSrc2 \ref source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXor_32u_Ctx(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXor_32u(const Npp32u * pSrc1, const Npp32u * pSrc2, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXor_8u_I_Ctx(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXor_8u_I(const Npp8u * pSrc, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXor_16u_I_Ctx(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXor_16u_I(const Npp16u * pSrc, Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned integer in place signal exclusive or with signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pSrcDst \ref in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsXor_32u_I_Ctx(const Npp32u * pSrc, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsXor_32u_I(const Npp32u * pSrc, Npp32u * pSrcDst, int nLength);

/** @} signal_xor */

/** 
 * @defgroup signal_not Not
 *
 * Bitwise NOT of each sample of a signal.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNot_8u_Ctx(const Npp8u * pSrc, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsNot_8u(const Npp8u * pSrc, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNot_16u_Ctx(const Npp16u * pSrc, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsNot_16u(const Npp16u * pSrc, Npp16u * pDst, int nLength);

/** 
 * 32-bit unsigned integer not signal.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNot_32u_Ctx(const Npp32u * pSrc, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsNot_32u(const Npp32u * pSrc, Npp32u * pDst, int nLength);

/** 
 * 8-bit unsigned char in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNot_8u_I_Ctx(Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsNot_8u_I(Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNot_16u_I_Ctx(Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsNot_16u_I(Npp16u * pSrcDst, int nLength);

/** 
 * 32-bit unsigned signed integer in place not signal.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsNot_32u_I_Ctx(Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsNot_32u_I(Npp32u * pSrcDst, int nLength);

/** @} signal_not */

/** 
 * @defgroup signal_lshiftc LShiftC
 *
 * Left shifts the bits of each sample of a signal by a constant amount.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_8u_Ctx(const Npp8u * pSrc, int nValue, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_8u(const Npp8u * pSrc, int nValue, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_16u_Ctx(const Npp16u * pSrc, int nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_16u(const Npp16u * pSrc, int nValue, Npp16u * pDst, int nLength);

/** 
 * 16-bit signed short signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_16s_Ctx(const Npp16s * pSrc, int nValue, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_16s(const Npp16s * pSrc, int nValue, Npp16s * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_32u_Ctx(const Npp32u * pSrc, int nValue, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_32u(const Npp32u * pSrc, int nValue, Npp32u * pDst, int nLength);

/** 
 * 32-bit signed integer signal left shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_32s_Ctx(const Npp32s * pSrc, int nValue, Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_32s(const Npp32s * pSrc, int nValue, Npp32s * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_8u_I_Ctx(int nValue, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_8u_I(int nValue, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_16u_I_Ctx(int nValue, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_16u_I(int nValue, Npp16u * pSrcDst, int nLength);

/** 
 * 16-bit signed short in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_16s_I_Ctx(int nValue, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_16s_I(int nValue, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit unsigned signed integer in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_32u_I_Ctx(int nValue, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_32u_I(int nValue, Npp32u * pSrcDst, int nLength);

/** 
 * 32-bit signed signed integer in place signal left shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to left shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsLShiftC_32s_I_Ctx(int nValue, Npp32s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsLShiftC_32s_I(int nValue, Npp32s * pSrcDst, int nLength);

/** @} signal_lshiftc */

/** 
 * @defgroup signal_rshiftc RShiftC
 *
 * Right shifts the bits of each sample of a signal by a constant amount.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_8u_Ctx(const Npp8u * pSrc, int nValue, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_8u(const Npp8u * pSrc, int nValue, Npp8u * pDst, int nLength);

/** 
 * 16-bit unsigned short signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_16u_Ctx(const Npp16u * pSrc, int nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_16u(const Npp16u * pSrc, int nValue, Npp16u * pDst, int nLength);

/** 
 * 16-bit signed short signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_16s_Ctx(const Npp16s * pSrc, int nValue, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_16s(const Npp16s * pSrc, int nValue, Npp16s * pDst, int nLength);

/** 
 * 32-bit unsigned integer signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_32u_Ctx(const Npp32u * pSrc, int nValue, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_32u(const Npp32u * pSrc, int nValue, Npp32u * pDst, int nLength);

/** 
 * 32-bit signed integer signal right shift with constant.
 * \param pSrc \ref source_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_32s_Ctx(const Npp32s * pSrc, int nValue, Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_32s(const Npp32s * pSrc, int nValue, Npp32s * pDst, int nLength);

/** 
 * 8-bit unsigned char in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_8u_I_Ctx(int nValue, Npp8u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_8u_I(int nValue, Npp8u * pSrcDst, int nLength);

/** 
 * 16-bit unsigned short in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_16u_I_Ctx(int nValue, Npp16u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_16u_I(int nValue, Npp16u * pSrcDst, int nLength);

/** 
 * 16-bit signed short in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_16s_I_Ctx(int nValue, Npp16s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_16s_I(int nValue, Npp16s * pSrcDst, int nLength);

/** 
 * 32-bit unsigned signed integer in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_32u_I_Ctx(int nValue, Npp32u * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_32u_I(int nValue, Npp32u * pSrcDst, int nLength);

/** 
 * 32-bit signed signed integer in place signal right shift with constant.
 * \param pSrcDst \ref in_place_signal_pointer.
 * \param nValue Constant value to be used to right shift each vector element
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsRShiftC_32s_I_Ctx(int nValue, Npp32s * pSrcDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsRShiftC_32s_I(int nValue, Npp32s * pSrcDst, int nLength);

/** @} signal_rshiftc */

/** @} signal_logical_and_shift_operations */

/** @} signal_arithmetic_and_logical_operations */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPS_ARITHMETIC_AND_LOGICAL_OPERATIONS_H */
