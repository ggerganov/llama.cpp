 /* Copyright 2010-2021 NVIDIA Corporation.  All rights reserved. 
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
#ifndef NV_NPPS_INITIALIZATION_H
#define NV_NPPS_INITIALIZATION_H
 
/**
 * \file npps_initialization.h
 * NPP Signal Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @defgroup signal_initialization Initialization
 * @ingroup npps
 * Functions that provide functionality of initialization signal like: set, zero or copy other signal.
 * @{
 */

/** \defgroup signal_set Set
 * The set of set initialization operations available in the library.
 * @{
 *
 */

/** @name Set 
 * Set methods for 1D vectors of various types. The copy methods operate on vector data given
 * as a pointer to the underlying data-type (e.g. 8-bit vectors would
 * be passed as pointers to Npp8u type) and length of the vectors, i.e. the number of items.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_8u_Ctx(Npp8u nValue, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_8u(Npp8u nValue, Npp8u * pDst, int nLength);

/** 
 * 8-bit signed char, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_8s_Ctx(Npp8s nValue, Npp8s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_8s(Npp8s nValue, Npp8s * pDst, int nLength);

/** 
 * 16-bit unsigned integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_16u_Ctx(Npp16u nValue, Npp16u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_16u(Npp16u nValue, Npp16u * pDst, int nLength);

/** 
 * 16-bit signed integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_16s_Ctx(Npp16s nValue, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_16s(Npp16s nValue, Npp16s * pDst, int nLength);

/** 
 * 16-bit integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_16sc_Ctx(Npp16sc nValue, Npp16sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_16sc(Npp16sc nValue, Npp16sc * pDst, int nLength);

/** 
 * 32-bit unsigned integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_32u_Ctx(Npp32u nValue, Npp32u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_32u(Npp32u nValue, Npp32u * pDst, int nLength);

/** 
 * 32-bit signed integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_32s_Ctx(Npp32s nValue, Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_32s(Npp32s nValue, Npp32s * pDst, int nLength);

/** 
 * 32-bit integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_32sc_Ctx(Npp32sc nValue, Npp32sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_32sc(Npp32sc nValue, Npp32sc * pDst, int nLength);

/** 
 * 32-bit float, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_32f_Ctx(Npp32f nValue, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_32f(Npp32f nValue, Npp32f * pDst, int nLength);

/** 
 * 32-bit float complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_32fc_Ctx(Npp32fc nValue, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_32fc(Npp32fc nValue, Npp32fc * pDst, int nLength);

/** 
 * 64-bit long long integer, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_64s_Ctx(Npp64s nValue, Npp64s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_64s(Npp64s nValue, Npp64s * pDst, int nLength);

/** 
 * 64-bit long long integer complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_64sc_Ctx(Npp64sc nValue, Npp64sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_64sc(Npp64sc nValue, Npp64sc * pDst, int nLength);

/** 
 * 64-bit double, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_64f_Ctx(Npp64f nValue, Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_64f(Npp64f nValue, Npp64f * pDst, int nLength);

/** 
 * 64-bit double complex, vector set method.
 * \param nValue Value used to initialize the vector pDst.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsSet_64fc_Ctx(Npp64fc nValue, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsSet_64fc(Npp64fc nValue, Npp64fc * pDst, int nLength);

/** @} end of Signal Set */
/** @} signal_set */

/** \defgroup signal_zero Zero
 * The set of zero initialization operations available in the library.
 * @{
 *
 */

/** @name Zero
 * Set signals to zero.
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_8u_Ctx(Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_8u(Npp8u * pDst, int nLength);

/** 
 * 16-bit integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_16s_Ctx(Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_16s(Npp16s * pDst, int nLength);

/** 
 * 16-bit integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_16sc_Ctx(Npp16sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_16sc(Npp16sc * pDst, int nLength);

/** 
 * 32-bit integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_32s_Ctx(Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_32s(Npp32s * pDst, int nLength);

/** 
 * 32-bit integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_32sc_Ctx(Npp32sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_32sc(Npp32sc * pDst, int nLength);

/** 
 * 32-bit float, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_32f_Ctx(Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_32f(Npp32f * pDst, int nLength);

/** 
 * 32-bit float complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_32fc_Ctx(Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_32fc(Npp32fc * pDst, int nLength);

/** 
 * 64-bit long long integer, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_64s_Ctx(Npp64s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_64s(Npp64s * pDst, int nLength);

/** 
 * 64-bit long long integer complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_64sc_Ctx(Npp64sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_64sc(Npp64sc * pDst, int nLength);

/** 
 * 64-bit double, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_64f_Ctx(Npp64f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_64f(Npp64f * pDst, int nLength);

/** 
 * 64-bit double complex, vector zero method.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsZero_64fc_Ctx(Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsZero_64fc(Npp64fc * pDst, int nLength);

/** @} end of Zero */

/** @} signal_zero */

/** \defgroup signal_copy Copy
 * The set of copy initialization operations available in the library.
 * @{
 *
 */

/** @name Copy
 * Copy methods for various type signals. Copy methods operate on
 * signal data given as a pointer to the underlying data-type (e.g. 8-bit
 * vectors would be passed as pointers to Npp8u type) and length of the
 * vectors, i.e. the number of items. 
 *
 * @{
 *
 */

/** 
 * 8-bit unsigned char, vector copy method
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_8u_Ctx(const Npp8u * pSrc, Npp8u * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_8u(const Npp8u * pSrc, Npp8u * pDst, int nLength);

/** 
 * 16-bit signed short, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_16s(const Npp16s * pSrc, Npp16s * pDst, int nLength);

/** 
 * 32-bit signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_32s_Ctx(const Npp32s * pSrc, Npp32s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_32s(const Npp32s * pSrc, Npp32s * pDst, int nLength);

/** 
 * 32-bit float, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_32f(const Npp32f * pSrc, Npp32f * pDst, int nLength);

/** 
 * 64-bit signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_64s_Ctx(const Npp64s * pSrc, Npp64s * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_64s(const Npp64s * pSrc, Npp64s * pDst, int nLength);

/** 
 * 16-bit complex short, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_16sc(const Npp16sc * pSrc, Npp16sc * pDst, int nLength);

/** 
 * 32-bit complex signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_32sc_Ctx(const Npp32sc * pSrc, Npp32sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_32sc(const Npp32sc * pSrc, Npp32sc * pDst, int nLength);

/** 
 * 32-bit complex float, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_32fc(const Npp32fc * pSrc, Npp32fc * pDst, int nLength);

/** 
 * 64-bit complex signed integer, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_64sc_Ctx(const Npp64sc * pSrc, Npp64sc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_64sc(const Npp64sc * pSrc, Npp64sc * pDst, int nLength);

/** 
 * 64-bit complex double, vector copy method.
 * \param pSrc \ref source_signal_pointer.
 * \param pDst \ref destination_signal_pointer.
 * \param nLength \ref length_specification.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref signal_data_error_codes, \ref length_error_codes.
 */
NppStatus 
nppsCopy_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, int nLength, NppStreamContext nppStreamCtx);

NppStatus 
nppsCopy_64fc(const Npp64fc * pSrc, Npp64fc * pDst, int nLength);

/** @} end of Copy */

/** @} signal_copy */

/** @} signal_initialization */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPS_INITIALIZATION_H */
