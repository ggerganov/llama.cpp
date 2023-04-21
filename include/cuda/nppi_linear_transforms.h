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
#ifndef NV_NPPI_LINEAR_TRANSFORMS_H
#define NV_NPPI_LINEAR_TRANSFORMS_H
 
/**
 * \file nppi_linear_transforms.h
 * NPP Image Processing Functionality.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_linear_transforms Linear Transforms
 *  @ingroup nppi
 *
 * Linear image transformations.
 *
 * @{
 *
 * These functions can be found in the nppist library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
 *
 */

/** @defgroup image_fourier_transforms Fourier Transforms
 * The set of Fourier transform functions available in the library.
 * @{
 *
 */

/**
 * 32-bit floating point complex to 32-bit floating point magnitude.
 * 
 * Converts complex-number pixel image to single channel image computing
 * the result pixels as the magnitude of the complex values.
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiMagnitude_32fc32f_C1R_Ctx(const Npp32fc * pSrc, int nSrcStep,
                                    Npp32f  * pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
                                          
NppStatus
nppiMagnitude_32fc32f_C1R(const Npp32fc * pSrc, int nSrcStep,
                                Npp32f  * pDst, int nDstStep,
                          NppiSize oSizeROI);
                                          
/**
 * 32-bit floating point complex to 32-bit floating point squared magnitude.
 * 
 * Converts complex-number pixel image to single channel image computing
 * the result pixels as the squared magnitude of the complex values.
 * 
 * The squared magnitude is an itermediate result of magnitude computation and
 * can thus be computed faster than actual magnitude. If magnitudes are required
 * for sorting/comparing only, using this function instead of nppiMagnitude_32fc32f_C1R
 * can be a worthwhile performance optimization.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus
nppiMagnitudeSqr_32fc32f_C1R_Ctx(const Npp32fc * pSrc, int nSrcStep,
                                       Npp32f  * pDst, int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx);
                                          
NppStatus
nppiMagnitudeSqr_32fc32f_C1R(const Npp32fc * pSrc, int nSrcStep,
                                   Npp32f  * pDst, int nDstStep,
                             NppiSize oSizeROI);
                                          
/** @} image_fourier_transforms */

/** @} image_linear_transforms */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPI_LINEAR_TRANSFORMS_H */
