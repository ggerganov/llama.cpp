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
#ifndef NV_NPPI_GEOMETRY_TRANSFORMS_H
#define NV_NPPI_GEOMETRY_TRANSFORMS_H
 
/**
 * \file nppi_geometry_transforms.h
 * Image Geometry Transform Primitives.
 */
 
#include "nppdefs.h"


#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup image_geometry_transforms Geometry Transforms
 *  @ingroup nppi
 *
 * Routines manipulating an image geometry.
 *
 * These functions can be found in the nppig library. Linking to only the sub-libraries that you use can significantly
 * save link time, application load time, and CUDA runtime startup time when using dynamic libraries.
 *
 * \section geometric_transform_api Geometric Transform API Specifics
 *
 * This section covers some of the unique API features common to the
 * geometric transform primitives.
 *
 * \subsection geometric_transform_roi Geometric Transforms and ROIs
 *
 * Geometric transforms operate on source and destination ROIs. The way
 * these ROIs affect the processing of pixels differs from other (non
 * geometric) image-processing primitives: Only pixels in the intersection
 * of the destination ROI and the transformed source ROI are being
 * processed.
 *
 * The typical processing proceedes as follows:
 * -# Transform the rectangular source ROI (given in source image coordinates)
 *		into the destination image space. This yields a quadrilateral.
 * -# Write only pixels in the intersection of the transformed source ROI and
 *		the destination ROI.
 *
 * \subsection geometric_transforms_interpolation Pixel Interpolation
 *
 * The majority of image geometry transform operation need to perform a 
 * resampling of the source image as source and destination pixels are not
 * coincident.
 *
 * NPP supports the following pixel inerpolation modes (in order from fastest to 
 * slowest and lowest to highest quality):
 * - nearest neighbor
 * - linear interpolation
 * - cubic convolution
 * - supersampling
 * - interpolation using Lanczos window function
 *
 * @{
 *
 */

/** @defgroup image_resize_square_pixel ResizeSqrPixel
 *
 * ResizeSqrPixel functions attempt to choose source pixels that would approximately represent the center of the destination pixels.
 * It does so by using the following scaling formula to select source pixels for interpolation:
 *
 * \code
 *   nAdjustedXFactor = 1.0 / nXFactor;
 *   nAdjustedYFactor = 1.0 / nYFactor;
 *   nAdjustedXShift = nXShift * nAdjustedXFactor + ((1.0 - nAdjustedXFactor) * 0.5);
 *   nAdjustedYShift = nYShift * nAdjustedYFactor + ((1.0 - nAdjustedYFactor) * 0.5);
 *   nSrcX = nAdjustedXFactor * nDstX - nAdjustedXShift;
 *   nSrcY = nAdjustedYFactor * nDstY - nAdjustedYShift;
 * \endcode
 *
 * ResizeSqrPixel functions support the following interpolation modes:
 *
 * \code
 *   NPPI_INTER_NN
 *   NPPI_INTER_LINEAR
 *   NPPI_INTER_CUBIC
 *   NPPI_INTER_CUBIC2P_BSPLINE
 *   NPPI_INTER_CUBIC2P_CATMULLROM
 *   NPPI_INTER_CUBIC2P_B05C03
 *   NPPI_INTER_SUPER
 *   NPPI_INTER_LANCZOS
 * \endcode
 *
 * In the ResizeSqrPixel functions below source image clip checking is handled as follows:
 *
 * If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than oSizeROI.x + oSizeROI.width and
 * greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height then the source pixel is considered to be within
 * the source image clip rectangle and the source image is sampled.  Otherwise the source image is not sampled and a destination pixel is not
 * written to the destination image. 
 *
 * \section resize_error_codes Error Codes
 * The resize primitives return the following error codes:
 *
 *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
 *           srcROIRect has no intersection with the source image.
 *         - ::NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - ::NPP_RESIZE_FACTOR_ERROR Indicates an error condition if either nXFactor or
 *           nYFactor is less than or equal to zero or in the case of NPPI_INTER_SUPER are not both downscaling.
 *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
 *
 * @{
 *
 */

/** @name GetResizeRect
 * Returns NppiRect which represents the offset and size of the destination rectangle that would be generated by
 * resizing the source NppiRect by the requested scale factors and shifts.
 *                                    
 * @{
 *
 */

/**
 * \param oSrcROI Region of interest in the source image.
 * \param pDstRect User supplied host memory pointer to an NppiRect structure that will be filled in by this function with the region of interest in the destination image.
 * \param nXFactor Factor by which x dimension is changed. 
 * \param nYFactor Factor by which y dimension is changed. 
 * \param nXShift Source pixel shift in x-direction.
 * \param nYShift Source pixel shift in y-direction.
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 */
NppStatus
nppiGetResizeRect(NppiRect oSrcROI, NppiRect *pDstRect, 
                  double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/** @} */

/** @name ResizeSqrPixel
 * Resizes images.
 *                                    
 * <h3><a name="CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image.
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nXFactor Factor by which x dimension is changed. 
 * \param nYFactor Factor by which y dimension is changed. 
 * \param nXShift Source pixel shift in x-direction.
 * \param nYShift Source pixel shift in y-direction.
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 *
 * <h3><a name="CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image.
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nXFactor Factor by which x dimension is changed. 
 * \param nYFactor Factor by which y dimension is changed. 
 * \param nXShift Source pixel shift in x-direction.
 * \param nYShift Source pixel shift in y-direction.
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                              double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 8-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                              double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                              double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 8-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_8u_P3R_Ctx(const Npp8u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst[3], int nDstStep, NppiRect oDstROI,
                              double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_P3R(const Npp8u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst[3], int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 8-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_8u_P4R_Ctx(const Npp8u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst[4], int nDstStep, NppiRect oDstROI,
                              double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_P4R(const Npp8u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst[4], int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 1 channel 16-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 16-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                                double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                            double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 16-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16u_P3R_Ctx(const Npp16u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst[3], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16u_P3R(const Npp16u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst[3], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 16-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16u_P4R_Ctx(const Npp16u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst[4], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16u_P4R(const Npp16u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst[4], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 1 channel 16-bit signed image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16s_C1R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16s_C1R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 16-bit signed image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16s_C3R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16s_C3R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 16-bit signed image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16s_C4R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16s_C4R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 16-bit signed image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16s_AC4R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                                double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16s_AC4R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16s * pDst, int nDstStep, NppiRect oDstROI,
                            double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 16-bit signed planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16s_P3R_Ctx(const Npp16s * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16s * pDst[3], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16s_P3R(const Npp16s * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16s * pDst[3], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 16-bit signed planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_16s_P4R_Ctx(const Npp16s * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16s * pDst[4], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_16s_P4R(const Npp16s * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16s * pDst[4], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 1 channel 32-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 32-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 32-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 32-bit floating point image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                                double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                            double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 32-bit floating point planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_32f_P3R_Ctx(const Npp32f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst[3], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_32f_P3R(const Npp32f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst[3], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 32-bit floating point planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_32f_P4R_Ctx(const Npp32f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst[4], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_32f_P4R(const Npp32f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst[4], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 1 channel 64-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_64f_C1R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_64f_C1R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                          double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 64-bit floating point image resize.
 *
  * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
*/
NppStatus 
nppiResizeSqrPixel_64f_C3R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_64f_C3R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 64-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_64f_C4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_64f_C4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 64-bit floating point image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPackedPixelParameters">Common parameters for nppiResizeSqrPixel packed pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_64f_AC4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                                double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_64f_AC4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp64f * pDst, int nDstStep, NppiRect oDstROI,
                            double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 3 channel 64-bit floating point planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_64f_P3R_Ctx(const Npp64f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp64f * pDst[3], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_64f_P3R(const Npp64f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst[3], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * 4 channel 64-bit floating point planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizeSqrPlanarPixelParameters">Common parameters for nppiResizeSqrPixel planar pixel functions</a>.
 *
 */
NppStatus 
nppiResizeSqrPixel_64f_P4R_Ctx(const Npp64f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp64f * pDst[4], int nDstStep, NppiRect oDstROI,
                               double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_64f_P4R(const Npp64f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst[4], int nDstStep, NppiRect oDstROI,
                           double nXFactor, double nYFactor, double nXShift, double nYShift, int eInterpolation);

/**
 * Buffer size for \ref nppiResizeSqrPixel_8u_C1R_Advanced.
 * \param oSrcROI \ref roi_specification.
 * \param oDstROI \ref roi_specification.
 * \param hpBufferSize Required buffer size. Important: hpBufferSize is a 
 *        <em>host pointer.</em> \ref general_scratch_buffer.
 * \param eInterpolationMode The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.
 * \return NPP_NULL_POINTER_ERROR if hpBufferSize is 0 (NULL),  \ref roi_error_codes.
 */
NppStatus 
nppiResizeAdvancedGetBufferHostSize_8u_C1R(NppiSize oSrcROI, NppiSize oDstROI, int * hpBufferSize /* host pointer */, int eInterpolationMode);

/**
 * 1 channel 8-bit unsigned image resize. This primitive matches the behavior of GraphicsMagick++.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image.
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nXFactor Factor by which x dimension is changed. 
 * \param nYFactor Factor by which y dimension is changed. 
 * \param pBuffer Device buffer that is used during calculations.
 * \param eInterpolationMode The type of eInterpolation to perform resampling. Currently only supports NPPI_INTER_LANCZOS3_Advanced.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 */
NppStatus 
nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                             Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                                       double nXFactor, double nYFactor, Npp8u * pBuffer, int eInterpolationMode, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeSqrPixel_8u_C1R_Advanced(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                                   double nXFactor, double nYFactor, Npp8u * pBuffer, int eInterpolationMode);
/** @} */

/** @} image_resize_square_pixel */

/** @defgroup image_resize Resize
 *
 * Resize functions use scale factor automatically determined by the width and height ratios of input and output \ref roi_specification. 
 *
 * This simplified function replaces the previous version which was deprecated in an earlier release. In this function the resize
 * scale factor is automatically determined by the width and height ratios of oSrcRectROI and oDstRectROI.  If either of those 
 * parameters intersect their respective image sizes then pixels outside the image size width and height will not be processed.
 *
 * Resize supports the following interpolation modes:
 *
 * \code
 *   NPPI_INTER_NN
 *   NPPI_INTER_LINEAR
 *   NPPI_INTER_CUBIC
 *   NPPI_INTER_SUPER
 *   NPPI_INTER_LANCZOS
 * \endcode
 *
 * \section resize_error_codes Error Codes
 * The resize primitives return the following error codes:
 *
 *         - ::NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
 *
 * @{
 *
 */

/** @name GetResizeTiledSourceOffset
 * Helper function that can be used when tiling a destination image with a source image using multiple Resize calls.
 * oSrcRectROI and oDstRectROI widths and heights should remain unmodified even if they will overlap source and destination
 * image sizes.  oDstRectROI offsets should be set to the destination offset of the new tile. 
 * Resize function processing will stop when source or destination image sizes are reached, any unavailable source image pixels
 * beyond source image size will be border replicated. There is no particular association assumed between source and destination image locations.
 * The values of oSrcRectROI.x and oSrcRectROI.y are ignored during this function call.
 *                                    
 * @{
 *
 */

/**
 *
 * \param oSrcRectROI Region of interest in the source image (may overlap source image size width and height).
 * \param oDstRectROI Region of interest in the destination image (may overlap destination image size width and height).
 * \param pNewSrcRectOffset Pointer to host memory NppiPoint object that will contain the new source image ROI offset
 *                          to be used in the nppiResize call to generate that tile.
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 */
NppStatus 
nppiGetResizeTiledSourceOffset(NppiRect oSrcRectROI, NppiRect oDstRectROI, NppiPoint * pNewSrcRectOffset);

/** @} */

/** @name Resize
 * Resizes images.
 *                                    
 * <h3><a name="CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer to origin of source image.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the entire source image.
 * \param oSrcRectROI Region of interest in the source image (may overlap source image size width and height).
 * \param pDst \ref destination_image_pointer to origin of destination image.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSize Size in pixels of the entire destination image.
 * \param oDstRectROI Region of interest in the destination image (may overlap destination image size width and height).
 * \param eInterpolation The type of eInterpolation to perform resampling (16f versions do not support Lanczos interpolation).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 *
 * <h3><a name="CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane origin pointers).
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the entire source image.
 * \param oSrcRectROI Region of interest in the source image (may overlap source image size width and height).
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane origin pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSize Size in pixels of the entire destination image.
 * \param oDstRectROI Region of interest in the destination image (may overlap destination image size width and height).
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 * 
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                            Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                        Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 8-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                            Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_8u_C3R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                        Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                            Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_8u_C4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                        Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_8u_AC4R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 8-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_8u_P3R_Ctx(const Npp8u * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                            Npp8u * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_8u_P3R(const Npp8u * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                        Npp8u * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 8-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_8u_P4R_Ctx(const Npp8u * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                            Npp8u * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_8u_P4R(const Npp8u * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                        Npp8u * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 1 channel 16-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16u_C1R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 16-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16u_C3R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16u_C4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                              Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16u_AC4R(const Npp16u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                          Npp16u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 16-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16u_P3R_Ctx(const Npp16u * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16u * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16u_P3R(const Npp16u * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16u * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit unsigned planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16u_P4R_Ctx(const Npp16u * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16u * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16u_P4R(const Npp16u * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16u * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 1 channel 16-bit signed image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16s_C1R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 16-bit signed image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16s_C3R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit signed image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16s_C4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit signed image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                              Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16s_AC4R(const Npp16s * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                          Npp16s * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 16-bit signed planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16s_P3R_Ctx(const Npp16s * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16s * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16s_P3R(const Npp16s * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16s * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit signed planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16s_P4R_Ctx(const Npp16s * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16s * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16s_P4R(const Npp16s * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16s * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 1 channel 16-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16f_C1R(const Npp16f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 16-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16f_C3R(const Npp16f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 16-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp16f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_16f_C4R(const Npp16f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp16f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 1 channel 32-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_32f_C1R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 32-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_32f_C3R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 32-bit floating point image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_32f_C4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 32-bit floating point image resize not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizePackedPixelParameters">Common parameters for nppiResize packed pixel functions</a>.
 *
 */
NppStatus 
nppiResize_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                              Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_32f_AC4R(const Npp32f * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                          Npp32f * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 3 channel 32-bit floating point planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_32f_P3R_Ctx(const Npp32f * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp32f * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_32f_P3R(const Npp32f * pSrc[3], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp32f * pDst[3], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/**
 * 4 channel 32-bit floating point planar image resize.
 *
 * For common parameter descriptions, see <a href="#CommonResizePlanarPixelParameters">Common parameters for nppiResize planar pixel functions</a>.
 *
 */
NppStatus 
nppiResize_32f_P4R_Ctx(const Npp32f * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                             Npp32f * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResize_32f_P4R(const Npp32f * pSrc[4], int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, 
                         Npp32f * pDst[4], int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation);

/** @} */

/** @} image_resize */

/** @defgroup image_resize_batch ResizeBatch
 *
 * ResizeBatch functions use scale factor automatically determined by the width and height ratios for each pair of input / output images in provided batches.
 *
 * In this function as in nppiResize the resize scale factor is automatically determined by the width and height ratios of oSrcRectROI and oDstRectROI.  If either of those 
 * parameters intersect their respective image sizes then pixels outside the image size width and height will not be processed.
 * Details of the resize operation are described above in the Resize section. ResizeBatch generally takes the same parameter list as 
 * Resize except that there is a list of N instances of those parameters (N > 1) and that list is passed in device memory. A convenient
 * data structure is provided that allows for easy initialization of the parameter lists.  The only restriction on these functions is
 * that there is one single source ROI rectangle and one single destination ROI rectangle which are applied respectively to each image 
 * in the batch.  The primary purpose of this function is to provide improved performance for batches of smaller images as long as GPU 
 * resources are available.  Therefore it is recommended that the function not be used for very large images as there may not be resources 
 * available for processing several large images simultaneously.  
 * A single set of oSrcRectROI and oDstRectROI values are applied to each source image and destination image in the batch in the nppiResizeBatch 
 * version of the function while per image specific oSrcRectROI and oDstRectROI values can be used in the nppiResizeBatch_Advanced version of the function.
 * Source and destination image sizes may vary but oSmallestSrcSize and oSmallestDstSize must be set to the smallest
 * source and destination image sizes in the batch. The parameters in the NppiResizeBatchCXR structure represent the corresponding
 * per-image nppiResize parameters for each image in the nppiResizeBatch functions and the NppiImageDescriptor and NppiResizeBatchROI_Advanced structures represent 
 * the corresponding per-image nppiResize parameters for the nppiResizeBatch_Advanced functions.  The NppiResizeBatchCXR or 
 * NppiImageDescriptor and NppiResizeBatchROI_Advanced arrays must be in device memory.
 *
 * ResizeBatch supports the following interpolation modes:
 *
 * \code
 *   NPPI_INTER_NN
 *   NPPI_INTER_LINEAR
 *   NPPI_INTER_CUBIC
 *   NPPI_INTER_SUPER
 * \endcode
 *
 * Below is the diagram of batch resize functions for variable ROIs. Figure shows the flexibility that the API can handle.
 * The ROIs for source and destination images can be any rectangular width and height that reflects the needed resize factors, inside or beyond the image boundary.
 *
 * \image html resize.png
 *
 * \section resize_error_codes Error Codes
 * The resize primitives return the following error codes:
 *
 *         - ::NPP_RESIZE_NO_OPERATION_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
 *
 * <h3><a name="CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions include:</a></h3>
 *
 * \param oSmallestSrcSize Size in pixels of the entire smallest source image width and height, may be from different images.
 * \param oSrcRectROI Region of interest in the source images (may overlap source image size width and height).
 * \param oSmallestDstSize Size in pixels of the entire smallest destination image width and height, may be from different images.
 * \param oDstRectROI Region of interest in the destination images (may overlap destination image size width and height).
 * \param eInterpolation The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, NPPI_INTER_CUBIC, or NPPI_INTER_SUPER. 
 * \param pBatchList Device memory pointer to nBatchSize list of NppiResizeBatchCXR structures.
 * \param pBatchSrc Device pointer to NppiImageDescriptor list of source image descriptors. User needs to intialize this structure and copy it to device.
 * \param pBatchDst Device pointer to NppiImageDescriptor list of destination image descriptors. User needs to intialize this structure and copy it to device. 
 * \param pBatchROI Device pointer to NppiResizeBatchROI_Advanced list of per-image variable ROIs. User needs to initialize this structure and copy it to device. 
 * \param nBatchSize Number of NppiResizeBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 *
 * <h3><a name="CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions include:</a></h3>
 *
 * \param nMaxWidth The maximum width of all destination ROIs
 * \param nMaxHeight The maximum height of all destination ROIs
 * \param pBatchSrc Device pointer to NppiImageDescriptor list of source image descriptors. User needs to intialize this structure and copy it to device.
 * \param pBatchDst Device pointer to NppiImageDescriptor list of destination image descriptors. User needs to intialize this structure and copy it to device. 
 * \param pBatchROI Device pointer to NppiResizeBatchROI_Advanced list of per-image variable ROIs. User needs to initialize this structure and copy it to device. 
 * \param nBatchSize Number of images in a batch.
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref resize_error_codes
 *
 * @{
 *
 */

typedef struct
{
    const void * pSrc;  /* device memory pointer */
    int nSrcStep;
    void * pDst;        /* device memory pointer */
    int nDstStep;
} NppiResizeBatchCXR;

/**
 * Data structure for variable ROI image resizing.
 * 
 */
typedef struct
{
    NppiRect oSrcRectROI;    
    NppiRect oDstRectROI;
} NppiResizeBatchROI_Advanced; 
 
/**
 * 1 channel 8-bit image resize batch.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_8u_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                           int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_8u_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                       int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 8-bit image resize batch.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_8u_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                           int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_8u_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                       int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 8-bit image resize batch.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_8u_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                           int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_8u_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                       int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 8-bit image resize batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_8u_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                            int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);
                         
NppStatus 
nppiResizeBatch_8u_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                        int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);
                         
/**
 * 1 channel 32-bit floating point image resize batch.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_32f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                            int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);
NppStatus 
nppiResizeBatch_32f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                        int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 32-bit floating point image resize batch.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_32f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                            int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_32f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                        int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 32-bit floating point image resize batch.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_32f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                            int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_32f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                        int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 32-bit floating point image resize batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchParameters">Common parameters for nppiResizeBatch functions</a>.
 *
 */
NppStatus 
nppiResizeBatch_32f_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                             int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_32f_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize, NppiRect oDstRectROI, 
                         int eInterpolation, NppiResizeBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 8-bit image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */ 
NppStatus 
nppiResizeBatch_8u_C1R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                    NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_8u_C1R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);

/**
 * 3 channel 8-bit image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */ 
NppStatus 
nppiResizeBatch_8u_C3R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                    NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
                                 
NppStatus 
nppiResizeBatch_8u_C3R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);
                                 
/**
 * 4 channel 8-bit image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_8u_C4R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                    NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);                                

NppStatus 
nppiResizeBatch_8u_C4R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);                                

/**
 * 4 channel 8-bit image resize batch for variable ROI not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_8u_AC4R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
                                                                         
NppStatus 
nppiResizeBatch_8u_AC4R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);
                                                                         
/**
 * 1 channel 16-bit floating point image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_16f_C1R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_16f_C1R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);

/**
 * 3 channel 16-bit floating point image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_16f_C3R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
                                 
NppStatus 
nppiResizeBatch_16f_C3R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);
                                 
/**
 * 4 channel 16-bit floating point image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_16f_C4R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);                                 

NppStatus 
nppiResizeBatch_16f_C4R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);                                 

/**
 * 1 channel 32-bit floating point image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_32f_C1R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiResizeBatch_32f_C1R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);

/**
 * 3 channel 32-bit floating point image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_32f_C3R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
                                 
NppStatus 
nppiResizeBatch_32f_C3R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);
                                 
/**
 * 4 channel 32-bit floating point image resize batch for variable ROI.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_32f_C4R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                     NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);                                 

NppStatus 
nppiResizeBatch_32f_C4R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                 NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);                                 

/**
 * 4 channel 32-bit floating point image resize batch for variable ROI not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonResizeBatchAdvancedParameters">Common parameters for nppiResizeBatchAdvanced functions</a>.
 *  
 */
NppStatus 
nppiResizeBatch_32f_AC4R_Advanced_Ctx(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                      NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
                                                                                          
NppStatus 
nppiResizeBatch_32f_AC4R_Advanced(int nMaxWidth, int nMaxHeight, NppiImageDescriptor * pBatchSrc, NppiImageDescriptor * pBatchDst,
                                  NppiResizeBatchROI_Advanced * pBatchROI, unsigned int nBatchSize, int eInterpolation);
                                                                                          
/** @} image_resize_batch */

/** @defgroup image_remap Remap
 *
 * Routines providing remap functionality.
 *
 * Remap supports the following interpolation modes:
 *
 *   NPPI_INTER_NN
 *   NPPI_INTER_LINEAR
 *   NPPI_INTER_CUBIC
 *   NPPI_INTER_CUBIC2P_BSPLINE
 *   NPPI_INTER_CUBIC2P_CATMULLROM
 *   NPPI_INTER_CUBIC2P_B05C03
 *   NPPI_INTER_LANCZOS
 *
 * Remap chooses source pixels using pixel coordinates explicitely supplied in two 2D device memory image arrays pointed to by the pXMap and pYMap pointers.
 * The pXMap array contains the X coordinated and the pYMap array contains the Y coordinate of the corresponding source image pixel to
 * use as input. These coordinates are in floating point format so fraction pixel positions can be used. The coordinates of the source
 * pixel to sample are determined as follows:
 *
 *   nSrcX = pxMap[nDstX, nDstY]
 *   nSrcY = pyMap[nDstX, nDstY]
 *
 * In the Remap functions below source image clip checking is handled as follows:
 *
 * If the source pixel fractional x and y coordinates are greater than or equal to oSizeROI.x and less than oSizeROI.x + oSizeROI.width and
 * greater than or equal to oSizeROI.y and less than oSizeROI.y + oSizeROI.height then the source pixel is considered to be within
 * the source image clip rectangle and the source image is sampled.  Otherwise the source image is not sampled and a destination pixel is not
 * written to the destination image. 
 *
 * \section remap_error_codes Error Codes
 * The remap primitives return the following error codes:
 *
 *         - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
 *           srcROIRect has no intersection with the source image.
 *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 *
 * <h3><a name="CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image.
 * \param oSrcROI Region of interest in the source image.
 * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
 * \param nXMapStep pXMap image array line step in bytes.
 * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
 * \param nYMapStep pYMap image array line step in bytes.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Region of interest size in the destination image.
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
 *
 * <h3><a name="CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image.
 * \param oSrcROI Region of interest in the source image.
 * \param pXMap Device memory pointer to 2D image array of X coordinate values to be used when sampling source image. 
 * \param nXMapStep pXMap image array line step in bytes.
 * \param pYMap Device memory pointer to 2D image array of Y coordinate values to be used when sampling source image. 
 * \param nYMapStep pYMap image array line step in bytes.
 * \param pDst \ref destination_planar_image_pointer_array.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstSizeROI Region of interest size in the destination image.
 * \param eInterpolation The type of eInterpolation to perform resampling.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref remap_error_codes
 *
 * @{
 *
 */

/** @name Remap
 * Remaps images.
 *                                    
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_8u_C1R_Ctx(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_8u_C1R(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                 const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                       Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 8-bit unsigned image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_8u_C3R_Ctx(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_8u_C3R(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                 const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                       Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_8u_C4R_Ctx(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_8u_C4R(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                 const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                       Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image remap not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_8u_AC4R_Ctx(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_8u_AC4R(const Npp8u  * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp8u  * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 8-bit unsigned planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_8u_P3R_Ctx(const Npp8u  * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_8u_P3R(const Npp8u  * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                 const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                       Npp8u  * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 8-bit unsigned planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_8u_P4R_Ctx(const Npp8u  * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                     const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                           Npp8u  * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_8u_P4R(const Npp8u  * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                 const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                       Npp8u  * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 1 channel 16-bit unsigned image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 16-bit unsigned image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image remap not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                       const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                             Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                   const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                         Npp16u * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 16-bit unsigned planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16u_P3R_Ctx(const Npp16u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16u * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16u_P3R(const Npp16u * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16u * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 16-bit unsigned planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16u_P4R_Ctx(const Npp16u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16u * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16u_P4R(const Npp16u * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16u * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 1 channel 16-bit signed image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16s_C1R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16s_C1R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 16-bit signed image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16s_C3R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16s_C3R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 16-bit signed image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16s_C4R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16s_C4R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 16-bit signed image remap not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16s_AC4R_Ctx(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                       const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                             Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16s_AC4R(const Npp16s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                   const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                         Npp16s * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 16-bit signed planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16s_P3R_Ctx(const Npp16s * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16s * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16s_P3R(const Npp16s * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16s * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 16-bit signed planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_16s_P4R_Ctx(const Npp16s * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp16s * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_16s_P4R(const Npp16s * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp16s * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 1 channel 32-bit floating point image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 32-bit floating point image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 32-bit floating point image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 32-bit floating point image remap not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                       const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                             Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                   const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                         Npp32f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 32-bit floating point planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_32f_P3R_Ctx(const Npp32f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp32f * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_32f_P3R(const Npp32f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp32f * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 32-bit floating point planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_32f_P4R_Ctx(const Npp32f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                            Npp32f * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_32f_P4R(const Npp32f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp32f * pXMap, int nXMapStep, const Npp32f * pYMap, int nYMapStep,
                        Npp32f * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 1 channel 64-bit floating point image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_64f_C1R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                            Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_64f_C1R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                        Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 64-bit floating point image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_64f_C3R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                            Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_64f_C3R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                        Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 64-bit floating point image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_64f_C4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                            Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_64f_C4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                        Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 64-bit floating point image remap not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPackedPixelParameters">Common parameters for nppiRemap packed pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_64f_AC4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                       const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                             Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_64f_AC4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                    const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                          Npp64f * pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 3 channel 64-bit floating point planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_64f_P3R_Ctx(const Npp64f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                            Npp64f * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_64f_P3R(const Npp64f * const pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                        Npp64f * pDst[3], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/**
 * 4 channel 64-bit floating point planar image remap.
 *
 * For common parameter descriptions, see <a href="#CommonRemapPlanarPixelParameters">Common parameters for nppiRemap planar pixel functions</a>.
 *
 */
NppStatus 
nppiRemap_64f_P4R_Ctx(const Npp64f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                      const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                            Npp64f * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRemap_64f_P4R(const Npp64f * const pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                  const Npp64f * pXMap, int nXMapStep, const Npp64f * pYMap, int nYMapStep,
                        Npp64f * pDst[4], int nDstStep, NppiSize oDstSizeROI, int eInterpolation);

/** @} */

/** @} image_remap */

/** @defgroup image_rotate Rotate
 *
 *  Rotates an image around the origin (0,0) and then shifts it.
 *
 * \section rotate_error_codes Rotate Error Codes
 * - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 * - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *   the intersection of the oSrcROI and source image is less than or
 *   equal to 1.
 * - ::NPP_WRONG_INTERSECTION_ROI_ERROR indicates an error condition if
 *   srcROIRect has no intersection with the source image.
 * - ::NPP_WRONG_INTERSECTION_QUAD_WARNING indicates a warning that no
 *   operation is performed if the transformed source ROI does not
 *   intersect the destination ROI.
 *
 * @{
 *
 */

/** @defgroup rotate_utility_functions Rotate Utility Functions
 * The set of rotate utility functions.
 * @{
 *
 */

/**
 * Compute shape of rotated image.
 * 
 * \param oSrcROI Region-of-interest of the source image.
 * \param aQuad Array of 2D points. These points are the locations of the corners
 *      of the rotated ROI. 
 * \param nAngle The rotation nAngle.
 * \param nShiftX Post-rotation shift in x-direction
 * \param nShiftY Post-rotation shift in y-direction
 * \return \ref roi_error_codes.
 */
NppStatus
nppiGetRotateQuad(NppiRect oSrcROI, double aQuad[4][2], double nAngle, double nShiftX, double nShiftY);

/**
 * Compute bounding-box of rotated image.
 * \param oSrcROI Region-of-interest of the source image.
 * \param aBoundingBox Two 2D points representing the bounding-box of the rotated image. All four points
 *      from nppiGetRotateQuad are contained inside the axis-aligned rectangle spanned by the the two
 *      points of this bounding box.
 * \param nAngle The rotation angle.
 * \param nShiftX Post-rotation shift in x-direction.
 * \param nShiftY Post-rotation shift in y-direction.
 *
 * \return \ref roi_error_codes.
 */
NppStatus 
nppiGetRotateBound(NppiRect oSrcROI, double aBoundingBox[2][2], double nAngle, double nShiftX, double nShiftY);

/** @} rotate_utility_functions */

/** @defgroup rotate_ Rotate
 * The set of rotate functions available in the library.
 * 
 * <h3><a name="CommonRotateParameters">Common parameters for nppiRotate functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcSize Size in pixels of the source image
 * \param oSrcROI Region of interest in the source image.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Region of interest in the destination image.
 * \param nAngle The angle of rotation in degrees.
 * \param nShiftX Shift along horizontal axis 
 * \param nShiftY Shift along vertical axis 
 * \param eInterpolation The type of interpolation to perform resampling
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref rotate_error_codes
 *
 * @{
 *
 */

/**
 * 8-bit unsigned image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                      double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                        Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                  double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 3 channel 8-bit unsigned image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                      double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                        Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                  double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                      double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                        Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                  double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 8-bit unsigned image rotate ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp8u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 16-bit unsigned image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 3 channel 16-bit unsigned image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 16-bit unsigned image rotate ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                              Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                        double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                          Npp16u * pDst, int nDstStep, NppiRect oDstROI,
                    double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 32-bit float image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 3 channel 32-bit float image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 32-bit float image rotate.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                       double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                         Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                   double nAngle, double nShiftX, double nShiftY, int eInterpolation);

/**
 * 4 channel 32-bit float image rotate ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate functions</a>.
 *
 */
NppStatus 
nppiRotate_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                              Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                        double nAngle, double nShiftX, double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiRotate_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                          Npp32f * pDst, int nDstStep, NppiRect oDstROI,
                    double nAngle, double nShiftX, double nShiftY, int eInterpolation);
/** @} rotate */

/** @} image_rotate */

/** @defgroup image_mirror Mirror
 *  Mirrors images horizontally, vertically or diagonally.
 *
 * \section mirror_error_codes Mirror Error Codes
 *         - ::NPP_MIRROR_FLIP_ERROR if flip has an illegal value.
 *         - ::NPP_SIZE_ERROR if in_place ROI width or height are not even numbers.
 *
 * <h3><a name="CommonMirrorParameters">Common parameters for nppiMirror non-inplace and inplace functions include:</a></h3>
 *
 * \param pSrcDst \ref in_place_image_pointer for inplace functions.
 * \param nSrcDstStep \ref in_place_image_line_step for inplace functions.
 * \param pSrc \ref source_image_pointer for non-inplace functions.
 * \param nSrcStep \ref source_image_line_step for non-inplace functions.
 * \param pDst \ref destination_image_pointer for non-inplace functions.
 * \param nDstStep \ref destination_image_line_step for non-inplace functions.
 * \param oROI \ref roi_specification (in_place ROI widths and heights must be even numbers).
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 8-bit unsigned image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_C1R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_C1R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 1 channel 8-bit unsigned in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_C1IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirror_8u_C1IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);
                              
/**
 * 3 channel 8-bit unsigned image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 8-bit unsigned in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_C3IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 8-bit unsigned image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 8-bit unsigned in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_C4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_C4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 8-bit unsigned image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_AC4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_AC4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 8-bit unsigned in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_8u_AC4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_8u_AC4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 1 channel 16-bit unsigned image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_C1R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirror_16u_C1R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
                              
/**
 * 1 channel 16-bit unsigned in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_C1IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_C1IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 16-bit unsigned image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_C3R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 16-bit unsigned in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_C3IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_C3IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit unsigned image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_C4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_C4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit unsigned in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_C4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_C4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit unsigned image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_AC4R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_AC4R(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit unsigned in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16u_AC4IR_Ctx(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16u_AC4IR(Npp16u * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 1 channel 16-bit signed image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_C1R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirror_16s_C1R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
                              
/**
 * 1 channel 16-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_C1IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_C1IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 16-bit signed image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_C3R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_C3R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 16-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_C3IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_C3IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit signed image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_C4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_C4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_C4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_C4IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit signed image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_AC4R_Ctx(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_AC4R(const Npp16s * pSrc, int nSrcStep, Npp16s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 16-bit signed in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_16s_AC4IR_Ctx(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_16s_AC4IR(Npp16s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 1 channel 32-bit image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_C1R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirror_32s_C1R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
                              
/**
 * 1 channel 32-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_C1IR_Ctx(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_C1IR(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 32-bit image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_C3R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_C3R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 32-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_C3IR_Ctx(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_C3IR(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_C4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_C4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_C4IR_Ctx(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_C4IR(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_AC4R_Ctx(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_AC4R(const Npp32s * pSrc, int nSrcStep, Npp32s * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit signed in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32s_AC4IR_Ctx(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32s_AC4IR(Npp32s * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 1 channel 32-bit float image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirror_32f_C1R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);
                              
/**
 * 1 channel 32-bit float in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_C1IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_C1IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 32-bit float image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_C3R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_C3R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 3 channel 32-bit float in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_C3IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_C3IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit float image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit float in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_C4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_C4IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit float image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_AC4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_AC4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oROI, NppiAxis flip);

/**
 * 4 channel 32-bit float in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror functions</a>.
 *
 */
NppStatus 
nppiMirror_32f_AC4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirror_32f_AC4IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip);

/** @} image_mirror */

/** @defgroup mirror_batch MirrorBatch
 *  Mirrors batches of images horizontally, vertically or diagonally.
 *
 * MirrorBatch generally takes the same parameter list as Mirror except that there is a list of N instances of those parameters (N > 1) 
 * and that list is passed in device memory.  A convenient data structure is provided that allows for easy initialization of the 
 * parameter lists.  The only restriction on these functions is that there is one single ROI and a single mirror flag which are
 * applied respectively to each image in the batch.  The primary purpose of this function is to
 * provide improved performance for batches of smaller images as long as GPU resources are available.  Therefore it is recommended
 * that the function not be used for very large images as there may not be resources available for processing several large images
 * simultaneously.  
 *
 * <h3><a name="CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch non-inplace and inplace functions include:</a></h3>
 *
 * \param oSizeROI \ref roi_specification.
 * \param flip Specifies the axis about which the images are to be mirrored.
 * \param pBatchList Device memory pointer to nBatchSize list of NppiMirrorBatchCXR structures.
 * \param nBatchSize Number of NppiMirrorBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
 *
 * @{
 *
 */

typedef struct
{
    const void * pSrc;  /* device memory pointer, ignored for in place versions */
    int nSrcStep;
    void * pDst;        /* device memory pointer */
    int nDstStep;
} NppiMirrorBatchCXR;

/**
 * 1 channel 32-bit float image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_C1R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirrorBatch_32f_C1R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);
                              
/**
 * 1 channel 32-bit float in place image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_C1IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirrorBatch_32f_C1IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);

/**
 * 3 channel 32-bit float image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_C3R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirrorBatch_32f_C3R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);
                              
/**
 * 3 channel 32-bit float in place image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_C3IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirrorBatch_32f_C3IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit float image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_C4R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirrorBatch_32f_C4R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);
                              
/**
 * 4 channel 32-bit float in place image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_C4IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirrorBatch_32f_C4IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);

/**
 * 4 channel 32-bit float image mirror batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_AC4R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);
                              
NppStatus 
nppiMirrorBatch_32f_AC4R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);
                              
/**
 * 4 channel 32-bit float in place image mirror batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch functions</a>.
 *
 */
NppStatus 
nppiMirrorBatch_32f_AC4IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiMirrorBatch_32f_AC4IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR * pBatchList, int nBatchSize);

/** @} mirror_batch */

/** @defgroup image_affine_transform Affine Transforms
 * The set of affine transform functions available in the library.
 *
 * \section affine_transform_error_codes Affine Transform Error Codes
 *
 * - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *   the intersection of the oSrcROI and source image is less than or
 *   equal to 1
 * - ::NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *   oSrcROI has no intersection with the source image
 * - ::NPP_INTERPOLATION_ERROR Indicates an error condition if
 *   interpolation has an illegal value
 * - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *   are invalid
 * - ::NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *   operation is performed if the transformed source ROI has no
 *   intersection with the destination ROI
 *
 * @{
 *
 */

/** @defgroup affine_transform_utility_functions Affine Transform Utility Functions
 * The set of affine transform utility functions.
 * @{
 *
 */

/**
 * Computes affine transform coefficients based on source ROI and destination quadrilateral.
 *
 * The function computes the coefficients of an affine transformation that maps the
 * given source ROI (axis aligned rectangle with integer coordinates) to a quadrilateral
 * in the destination image.
 *
 * An affine transform in 2D is fully determined by the mapping of just three vertices.
 * This function's API allows for passing a complete quadrilateral effectively making the 
 * prolem overdetermined. What this means in practice is, that for certain quadrilaterals it is
 * not possible to find an affine transform that would map all four corners of the source
 * ROI to the four vertices of that quadrilateral.
 *
 * The function circumvents this problem by only looking at the first three vertices of
 * the destination image quadrilateral to determine the affine transformation's coefficients.
 * If the destination quadrilateral is indeed one that cannot be mapped using an affine
 * transformation the functions informs the user of this situation by returning a 
 * ::NPP_AFFINE_QUAD_INCORRECT_WARNING.
 *
 * \param oSrcROI The source ROI. This rectangle needs to be at least one pixel wide and
 *          high. If either width or hight are less than one an ::NPP_RECTANGLE_ERROR is returned.
 * \param aQuad The destination quadrilateral.
 * \param aCoeffs The resulting affine transform coefficients.
 * \return Error codes:
 *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *           the intersection of the oSrcROI and source image is less than or
 *           equal to 1
 *         - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *           are invalid
 *         - ::NPP_AFFINE_QUAD_INCORRECT_WARNING Indicates a warning when quad
 *           does not conform to the transform properties. Fourth vertex is
 *           ignored, internally computed coordinates are used instead
 */
NppStatus 
nppiGetAffineTransform(NppiRect oSrcROI, const double aQuad[4][2], double aCoeffs[2][3]);


/**
 * Compute shape of transformed image.
 *
 * This method computes the quadrilateral in the destination image that 
 * the source ROI is transformed into by the affine transformation expressed
 * by the coefficients array (aCoeffs).
 *
 * \param oSrcROI The source ROI.
 * \param aQuad The resulting destination quadrangle.
 * \param aCoeffs The afine transform coefficients.
 * \return Error codes:
 *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *           the intersection of the oSrcROI and source image is less than or
 *           equal to 1
 *         - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetAffineQuad(NppiRect oSrcROI, double aQuad[4][2], const double aCoeffs[2][3]);


/**
 * Compute bounding-box of transformed image.
 *
 * The method effectively computes the bounding box (axis aligned rectangle) of
 * the transformed source ROI (see nppiGetAffineQuad()). 
 *
 * \param oSrcROI The source ROI.
 * \param aBound The resulting bounding box.
 * \param aCoeffs The afine transform coefficients.
 * \return Error codes:
 *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *           the intersection of the oSrcROI and source image is less than or
 *           equal to 1
 *         - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetAffineBound(NppiRect oSrcROI, double aBound[2][2], const double aCoeffs[2][3]);

/** @} affine_transform_utility_functions */

/** @defgroup affine_transform Affine Transform
 * Transforms (warps) an image based on an affine transform. 
 *
 * The affine transform is given as a \f$2\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
 * source image is mapped to the location \f$(x', y')\f$ in the destination image.
 * The destination image coorodinates are computed as follows:
 * \f[
 * x' = c_{00} * x + c_{01} * y + c_{02} \qquad
 * y' = c_{10} * x + c_{11} * y + c_{12} \qquad
 * C = \left[ \matrix{c_{00} & c_{01} & c_{02} \cr 
                      c_{10} & c_{11} & c_{12} } \right]
 * \f]
 * Affine transforms can be understood as a linear transformation (traditional
 * matrix multiplication) and a shift operation. The \f$2\times 2\f$ matrix 
 * \f[
 *    L = \left[ \matrix{c_{00} & c_{01} \cr 
 *                       c_{10} & c_{11} } \right]
 * \f]
 * represents the linear transform portion of the affine transformation. The
 * vector
 * \f[
 *      v = \left( \matrix{c_{02} \cr
                           c_{12} } \right)
 * \f]
 * represents the post-transform shift, i.e. after the pixel location is transformed
 * by \f$L\f$ it is translated by \f$v\f$.
 * 
 * <h3><a name="CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Affine transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
 *
 * <h3><a name="CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_planar_image_pointer_array. (host memory array containing device memory image plane pointers)
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Affine transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                            Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                      const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 8-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                      const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                      const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 8-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                      const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 8-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                            Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                      const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 16-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 16-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 16-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 16-bit unsigned affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                              Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                        const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 16-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 32-bit signed affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 32-bit signed affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit signed affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit signed affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                              Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                        const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 32-bit signed affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 32-bit signed affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 16-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16f_C1R_Ctx(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16f_C1R(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 16-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16f_C3R_Ctx(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16f_C3R(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 16-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_16f_C4R_Ctx(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_16f_C4R(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 32-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 32-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                              Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                        const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 32-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 32-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 64-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_64f_C1R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_64f_C1R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 64-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_64f_C3R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_64f_C3R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 64-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_64f_C4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_64f_C4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                             Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 64-bit floating-point affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_64f_AC4R_Ctx(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_64f_AC4R(const Npp64f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                              Npp64f * pDst, int nDstStep, NppiRect oDstROI, 
                        const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 64-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_64f_P3R_Ctx(const Npp64f * aSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp64f * aDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_64f_P3R(const Npp64f * aSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             Npp64f * aDst[3], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 64-bit floating-point affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffine_64f_P4R_Ctx(const Npp64f * aSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp64f * aDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffine_64f_P4R(const Npp64f * aSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                             Npp64f * aDst[4], int nDstStep, NppiRect oDstROI, 
                       const double aCoeffs[2][3], int eInterpolation);


/** @} affine_transform */

/** @defgroup affine_transform_batch Affine Transform Batch
 *
 * Details of the warp affine operation are described above in the WarpAffine section. WarpAffineBatch generally takes the same parameter list as 
 * WarpAffine except that there is a list of N instances of those parameters (N > 1) and that list is passed in device memory. A convenient
 * data structure is provided that allows for easy initialization of the parameter lists.  The aTransformedCoeffs array is for internal use only
 * and should not be directly initialized by the application.  The only restriction on these functions is
 * that there is one single source ROI rectangle and one single destination ROI rectangle which are applied respectively to each image 
 * in the batch.  The primary purpose of this function is to provide improved performance for batches of smaller images as long as GPU 
 * resources are available.  Therefore it is recommended that the function not be used for very large images as there may not be resources 
 * available for processing several large images simultaneously.  
 * A single set of oSrcRectROI and oDstRectROI values are applied to each source image and destination image in the batch.
 * Source and destination image sizes may vary but oSmallestSrcSize must be set to the smallest
 * source and image size in the batch. The parameters in the NppiWarpAffineBatchCXR structure represent the corresponding
 * per-image nppiWarpAffine parameters for each image in the batch.  The NppiWarpAffineBatchCXR array must be in device memory.
 * The nppiWarpAffineBatchInit function MUST be called AFTER the application has initialized the array of NppiWarpAffineBatchCXR structures
 * and BEFORE calling any of the nppiWarpAffineBatch functions to so that the aTransformedCoeffs array can be internally pre-initialized
 * for each image in the batch. The batch size passed to nppiWarpAffineBatchInit must match the batch size passed to the corresponding
 * warp affine batch function.
 *
 *
 * WarpAffineBatch supports the following interpolation modes:
 *
 * \code
 *   NPPI_INTER_NN
 *   NPPI_INTER_LINEAR
 *   NPPI_INTER_CUBIC
 * \endcode
 *
 * \section Error Codes
 * The warp affine primitives return the following error codes:
 *
 *         - ::NPP_RECTANGLE_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
 *
 * <h3><a name="CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions include:</a></h3>
 *
 * \param oSmallestSrcSize Size in pixels of the entire smallest source image width and height, may be from different images.
 * \param oSrcRectROI Region of interest in the source images.
 * \param oDstRectROI Region of interest in the destination images.
 * \param eInterpolation The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, or NPPI_INTER_CUBIC. 
 * \param pBatchList Device memory pointer to nBatchSize list of NppiWarpAffineBatchCXR structures.
 * \param nBatchSize Number of NppiWarpAffineBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

typedef struct
{
    const void * pSrc;  /* device memory pointer */
    int nSrcStep;
    void * pDst;        /* device memory pointer */
    int nDstStep;
    Npp64f * pCoeffs;   /* device memory pointer to the tranformation matrix with double precision floating-point coefficient values to be used for this image */
    Npp64f aTransformedCoeffs[2][3]; /* FOR INTERNAL USE, DO NOT INITIALIZE  */
} NppiWarpAffineBatchCXR;


/**
 * Initializes the aTransformdedCoeffs array in pBatchList for each image in the list. 
 * MUST be called before calling the corresponding warp affine batch function whenever any of the transformation matrices in the list have changed.
 *
 * \param pBatchList Device memory pointer to nBatchSize list of NppiWarpAffineBatchCXR structures.
 * \param nBatchSize Number of NppiWarpAffineBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 */
NppStatus 
nppiWarpAffineBatchInit_Ctx(NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatchInit(NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 8-bit unsigned integer image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_8u_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                               int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_8u_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                           int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 8-bit unsigned integer image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_8u_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                               int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_8u_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                           int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_8u_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                               int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_8u_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                           int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer image warp affine batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_8u_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_8u_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 16-bit floating point image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_16f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_16f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 16-bit floating point image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_16f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_16f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 16-bit floating point image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_16f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_16f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 32-bit floating point image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_32f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_32f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 32-bit floating point image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_32f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_32f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 32-bit floating point image warp affine batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_32f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_32f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                            int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 32-bit floating point image warp affine batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBatchParameters">Common parameters for nppiWarpAffineBatch functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBatch_32f_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBatch_32f_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                             int eInterpolation, NppiWarpAffineBatchCXR * pBatchList, unsigned int nBatchSize);

/** @} affine_transform_batch */

/** @defgroup backwards_affine_transform Backwards Affine Transform
 * Transforms (warps) an image based on an affine transform. 
 *
 * The affine transform is given as a \f$2\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
 * source image is mapped to the location \f$(x', y')\f$ in the destination image.
 * The destination image coorodinates fullfil the following properties:
 * \f[
 * x = c_{00} * x' + c_{01} * y' + c_{02} \qquad
 * y = c_{10} * x' + c_{11} * y' + c_{12} \qquad
 * C = \left[ \matrix{c_{00} & c_{01} & c_{02} \cr 
                      c_{10} & c_{11} & c_{12} } \right]
 * \f]
 * In other words, given matrix \f$C\f$ the source image's shape is transfored to the destination image
 * using the inverse matrix \f$C^{-1}\f$:
 * \f[
 * M = C^{-1} = \left[ \matrix{m_{00} & m_{01} & m_{02} \cr 
                               m_{10} & m_{11} & m_{12} } \right]
 * x' = m_{00} * x + m_{01} * y + m_{02} \qquad
 * y' = m_{10} * x + m_{11} * y + m_{12} \qquad
 * \f]
 *
 * <h3><a name="CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Affine transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
 *
 * <h3><a name="CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Affine transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                              const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 8-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                              const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                              const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer backwards affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 8-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                              const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 8-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                    Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                              const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                          const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 16-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 16-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer backwards affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 16-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 32-bit signed integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 32-bit signed integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit signed integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit signed integer backwards affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 32-bit signed integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 32-bit signed integer backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Single-channel 32-bit floating-point backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel 32-bit floating-point backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point backwards affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[2][3], int eInterpolation);

/**
 * Three-channel planar 32-bit floating-point backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/**
 * Four-channel planar 32-bit floating-point backwards affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineBack_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[2][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineBack_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[2][3], int eInterpolation);

/** @} backwards_affine_transfrom */

/** @defgroup quad_based_affine_transform Quad-Based Affine Transform
 * Transforms (warps) an image based on an affine transform. 
 *
 * The affine transform is computed such that it maps a quadrilateral in source image space to a 
 * quadrilateral in destination image space. 
 *
 * An affine transform is fully determined by the mapping of 3 discrete points.
 * The following primitives compute an affine transformation matrix that maps 
 * the first three corners of the source quad are mapped to the first three 
 * vertices of the destination image quad. If the fourth vertices do not match
 * the transform, an ::NPP_AFFINE_QUAD_INCORRECT_WARNING is returned by the primitive.
 *
 * <h3><a name="CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param aSrcQuad Source quad.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aDstQuad Destination quad.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
 *
 * <h3><a name="CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI
 * \param aSrcQuad Source quad.
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI
 * \param aDstQuad Destination quad.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref affine_transform_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 32-bit floating-point quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                    Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                              int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                          int eInterpolation);


/**
 * Three-channel 8-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                    Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                              int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                          int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                    Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                              int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                          int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer quad-based affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp8u * pDst,                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Three-channel planar 8-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                    Npp8u * pDst[3],                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                              int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                Npp8u * pDst[3],                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                          int eInterpolation);

/**
 * Four-channel planar 8-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                    Npp8u * pDst[4],                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                              int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                Npp8u * pDst[4],                    int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                          int eInterpolation);

/**
 * Single-channel 16-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI,                    const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI,                    const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Three-channel 16-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer quad-based affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                                int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                            int eInterpolation);

/**
 * Three-channel planar 16-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Single-channel 32-bit signed integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Three-channel 32-bit signed integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel 32-bit signed integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel 32-bit signed integer quad-based affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                                int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                  Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                            int eInterpolation);

/**
 * Three-channel planar 32-bit signed integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel planar 32-bit signed integer quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Single-channel 32-bit floating-point quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Three-channel 32-bit floating-point quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel 32-bit floating-point quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2],
                                     Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2],
                                 Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel 32-bit floating-point quad-based affine warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPackedPixelParameters">Common parameters for nppiWarpAffineQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                                int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                            int eInterpolation);

/**
 * Three-channel planar 32-bit floating-point quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);

/**
 * Four-channel planar 32-bit floating-point quad-based affine warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpAffineQuadPlanarPixelParameters">Common parameters for nppiWarpAffineQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpAffineQuad_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                               int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpAffineQuad_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                 Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], 
                           int eInterpolation);


/** @} quad_based_affine_transform */

/** @} image_affine_transforms */

/** @defgroup image_perspective_transforms Perspective Transform
 * The set of perspective transform functions available in the library.
 *
 * \section perspective_transform_error_codes Perspective Transform Error Codes
 *
 * - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *   the intersection of the oSrcROI and source image is less than or
 *   equal to 1
 * - ::NPP_WRONG_INTERSECTION_ROI_ERROR Indicates an error condition if
 *   oSrcROI has no intersection with the source image
 * - ::NPP_INTERPOLATION_ERROR Indicates an error condition if
 *   interpolation has an illegal value
 * - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *   are invalid
 * - ::NPP_WRONG_INTERSECTION_QUAD_WARNING Indicates a warning that no
 *   operation is performed if the transformed source ROI has no
 *   intersection with the destination ROI
 *
 * @{
 *
 */

/** @defgroup perspective_transform_utility_functions Perspective Transform Utility Functions
 * The set of perspective transform utility functions.
 * @{
 *
 */

/**
 * Calculates perspective transform coefficients given source rectangular ROI
 * and its destination quadrangle projection
 *
 * \param oSrcROI Source ROI
 * \param quad Destination quadrangle
 * \param aCoeffs Perspective transform coefficients
 * \return Error codes:
 *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *           the intersection of the oSrcROI and source image is less than or
 *           equal to 1
 *         - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetPerspectiveTransform(NppiRect oSrcROI, const double quad[4][2], double aCoeffs[3][3]);


/**
 * Calculates perspective transform projection of given source rectangular
 * ROI
 *
 * \param oSrcROI Source ROI
 * \param quad Destination quadrangle
 * \param aCoeffs Perspective transform coefficients
 * \return Error codes:
 *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *           the intersection of the oSrcROI and source image is less than or
 *           equal to 1
 *         - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetPerspectiveQuad(NppiRect oSrcROI, double quad[4][2], const double aCoeffs[3][3]);


/**
 * Calculates bounding box of the perspective transform projection of the
 * given source rectangular ROI
 *
 * \param oSrcROI Source ROI
 * \param bound Bounding box of the transformed source ROI
 * \param aCoeffs Perspective transform coefficients
 * \return Error codes:
 *         - ::NPP_SIZE_ERROR Indicates an error condition if any image dimension
 *           has zero or negative value
 *         - ::NPP_RECTANGLE_ERROR Indicates an error condition if width or height of
 *           the intersection of the oSrcROI and source image is less than or
 *           equal to 1
 *         - ::NPP_COEFFICIENT_ERROR Indicates an error condition if coefficient values
 *           are invalid
 */
NppStatus 
nppiGetPerspectiveBound(NppiRect oSrcROI, double bound[2][2], const double aCoeffs[3][3]);

/** @} perspective_transform_utility_functions */

/** @defgroup perspective_transform Perspective Transform
 * Transforms (warps) an image based on a perspective transform. 
 *
 * The perspective transform is given as a \f$3\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
 * source image is mapped to the location \f$(x', y')\f$ in the destination image.
 * The destination image coorodinates are computed as follows:
 * \f[
 * x' = \frac{c_{00} * x + c_{01} * y + c_{02}}{c_{20} * x + c_{21} * y + c_{22}} \qquad
 * y' = \frac{c_{10} * x + c_{11} * y + c_{12}}{c_{20} * x + c_{21} * y + c_{22}}
 * \f]
 * \f[
 * C = \left[ \matrix{c_{00} & c_{01} & c_{02}   \cr 
                      c_{10} & c_{11} & c_{12}   \cr 
                      c_{20} & c_{21} & c_{22} } \right]
 * \f]
 *
 * <h3><a name="CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Perspective transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
 *
 * <h3><a name="CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Perspective transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                 Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                           const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3],int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3],int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer perspective warp, igoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                 const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                   Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                             const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit signed integer perspective warp, igoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                 const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                   Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                             const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 16-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16f_C1R_Ctx(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16f_C1R(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 16-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16f_C3R_Ctx(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16f_C3R(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 16-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_16f_C4R_Ctx(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_16f_C4R(const Npp16f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp16f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                 const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                   Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                             const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspective_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspective_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                  Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                            const double aCoeffs[3][3], int eInterpolation);

/** @} perspective_transform */

/** @defgroup perspective_transform_batch Perspective Transform Batch
 *
 * Details of the warp perspective operation are described above in the WarpPerspective section. WarpPerspectiveBatch generally takes the same parameter list as 
 * WarpPerspective except that there is a list of N instances of those parameters (N > 1) and that list is passed in device memory. A convenient
 * data structure is provided that allows for easy initialization of the parameter lists.  The aTransformedCoeffs array is for internal use only
 * and should not be directly initialized by the application.  The only restriction on these functions is
 * that there is one single source ROI rectangle and one single destination ROI rectangle which are applied respectively to each image 
 * in the batch.  The primary purpose of this function is to provide improved performance for batches of smaller images as long as GPU 
 * resources are available.  Therefore it is recommended that the function not be used for very large images as there may not be resources 
 * available for processing several large images simultaneously.  
 * A single set of oSrcRectROI and oDstRectROI values are applied to each source image and destination image in the batch.
 * Source and destination image sizes may vary but oSmallestSrcSize must be set to the smallest
 * source and image size in the batch. The parameters in the NppiWarpPerspectiveBatchCXR structure represent the corresponding
 * per-image nppiWarpPerspective parameters for each image in the batch.  The NppiWarpPerspectiveBatchCXR array must be in device memory.
 * The nppiWarpPerspectiveBatchInit function MUST be called AFTER the application has initialized the array of NppiWarpPerspectiveBatchCXR structures
 * and BEFORE calling any of the nppiWarpPerspectiveBatch functions to so that the aTransformedCoeffs array can be internally pre-initialized
 * for each image in the batch. The batch size passed to nppiWarpPerspectiveBatchInit must match the batch size passed to the corresponding
 * warp perspective batch function.
 *
 *
 * WarpPerspectiveBatch supports the following interpolation modes:
 *
 * \code
 *   NPPI_INTER_NN
 *   NPPI_INTER_LINEAR
 *   NPPI_INTER_CUBIC
 * \endcode
 *
 * \section Error Codes
 * The warp perspective primitives return the following error codes:
 *
 *         - ::NPP_RECTANGLE_ERROR if either destination ROI width or
 *           height is less than 1 pixel.
 *         - ::NPP_INTERPOLATION_ERROR if eInterpolation has an illegal value.
 *         - ::NPP_SIZE_ERROR if source size width or height is less than 2 pixels.
 *
 * <h3><a name="CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions include:</a></h3>
 *
 * \param oSmallestSrcSize Size in pixels of the entire smallest source image width and height, may be from different images.
 * \param oSrcRectROI Region of interest in the source images.
 * \param oDstRectROI Region of interest in the destination images.
 * \param eInterpolation The type of eInterpolation to perform resampling. Currently limited to NPPI_INTER_NN, NPPI_INTER_LINEAR, or NPPI_INTER_CUBIC. 
 * \param pBatchList Device memory pointer to nBatchSize list of NppiWarpPerspectiveBatchCXR structures.
 * \param nBatchSize Number of NppiWarpPerspectiveBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes
 *
 * @{
 *
 */

typedef struct
{
    const void * pSrc;  /* device memory pointer */
    int nSrcStep;
    void * pDst;        /* device memory pointer */
    int nDstStep;
    Npp64f * pCoeffs;   /* device memory pointer to the tranformation matrix with double precision floating-point coefficient values to be used for this image */
    Npp64f aTransformedCoeffs[3][3]; /* FOR INTERNAL USE, DO NOT INITIALIZE  */
} NppiWarpPerspectiveBatchCXR;


/**
 * Initializes the aTransformdedCoeffs array in pBatchList for each image in the list. 
 * MUST be called before calling the corresponding warp perspective batch function whenever any of the transformation matrices in the list have changed.
 *
 * \param pBatchList Device memory pointer to nBatchSize list of NppiWarpPerspectiveBatchCXR structures.
 * \param nBatchSize Number of NppiWarpPerspectiveBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context. 
 */
NppStatus 
nppiWarpPerspectiveBatchInit_Ctx(NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatchInit(NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 8-bit unsigned integer image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_8u_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                    int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_8u_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 8-bit unsigned integer image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_8u_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                    int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_8u_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_8u_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                    int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_8u_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 8-bit unsigned integer image warp perspective batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_8u_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_8u_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 16-bit floating point image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_16f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_16f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 16-bit floating point image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_16f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_16f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 16-bit floating point image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_16f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_16f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 1 channel 32-bit floating point image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_32f_C1R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_32f_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 3 channel 32-bit floating point image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_32f_C3R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_32f_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 32-bit floating point image warp perspective batch.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_32f_C4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                     int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_32f_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                 int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/**
 * 4 channel 32-bit floating point image warp perspective batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBatchParameters">Common parameters for nppiWarpPerspectiveBatch functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBatch_32f_AC4R_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                      int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBatch_32f_AC4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI, 
                                  int eInterpolation, NppiWarpPerspectiveBatchCXR * pBatchList, unsigned int nBatchSize);

/** @} perspective_transform_batch */


/** @defgroup backwards_perspective_transform Backwards Perspective Transform
 * Transforms (warps) an image based on a perspective transform. 
 *
 * The perspective transform is given as a \f$3\times 3\f$ matrix C. A pixel location \f$(x, y)\f$ in the
 * source image is mapped to the location \f$(x', y')\f$ in the destination image.
 * The destination image coorodinates fullfil the following properties:
 * \f[
 * x = \frac{c_{00} * x' + c_{01} * y' + c_{02}}{c_{20} * x' + c_{21} * y' + c_{22}} \qquad
 * y = \frac{c_{10} * x' + c_{11} * y' + c_{12}}{c_{20} * x' + c_{21} * y' + c_{22}}
 * \f]
 * \f[
 * C = \left[ \matrix{c_{00} & c_{01} & c_{02}   \cr 
                      c_{10} & c_{11} & c_{12}   \cr 
                      c_{20} & c_{21} & c_{22} } \right]
 * \f]
 * In other words, given matrix \f$C\f$ the source image's shape is transfored to the destination image
 * using the inverse matrix \f$C^{-1}\f$:
 * \f[
 * M = C^{-1} = \left[ \matrix{m_{00} & m_{01} & m_{02} \cr 
                               m_{10} & m_{11} & m_{12} \cr 
                               m_{20} & m_{21} & m_{22} } \right]
 * x' = \frac{c_{00} * x + c_{01} * y + c_{02}}{c_{20} * x + c_{21} * y + c_{22}} \qquad
 * y' = \frac{c_{10} * x + c_{11} * y + c_{12}}{c_{20} * x + c_{21} * y + c_{22}}
 * \f]
 *
 * <h3><a name="CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Perspective transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
 *
 * <h3><a name="CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aCoeffs Perspective transform coefficients.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
 *
 * @{
 *
 */


/**
 * Single-channel 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                                   const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                                   const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                                   const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer backwards perspective warp, igoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp8u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                         Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                                   const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                         Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                                   const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                     Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, 
                               const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);
                                            
NppStatus 
nppiWarpPerspectiveBack_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);
                                            
/**
 * Four-channel 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer backwards perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                           Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                     const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp16u * pDst, int nDstStep, NppiRect oDstROI, 
                                 const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit signed integer backwards perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                           Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                     const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp32s * pDst, int nDstStep, NppiRect oDstROI, 
                                 const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Single-channel 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel 32-bit floating-point backwards perspective warp, ignorning alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                           Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                     const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                       Npp32f * pDst, int nDstStep, NppiRect oDstROI, 
                                 const double aCoeffs[3][3], int eInterpolation);

/**
 * Three-channel planar 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                          Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, 
                                      Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/**
 * Four-channel planar 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveBack_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                                    const double aCoeffs[3][3], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveBack_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, 
                                const double aCoeffs[3][3], int eInterpolation);

/** @} backward_perspective_transform */

/** @defgroup quad_based_perspective_transform Quad-Based Perspective Transform
 * Transforms (warps) an image based on an perspective transform. 
 *
 * The perspective transform is computed such that it maps a quadrilateral in source image space to a 
 * quadrilateral in destination image space. 
 *
 * <h3><a name="CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_image_pointer.
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param aSrcQuad Source quad.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aDstQuad Destination quad.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
 *
 * <h3><a name="CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions include:</a></h3>
 *
 * \param pSrc \ref source_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param oSrcSize Size of source image in pixels.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSrcROI Source ROI.
 * \param aSrcQuad Source quad.
 * \param pDst \ref destination_planar_image_pointer_array (host memory array containing device memory image plane pointers).
 * \param nDstStep \ref destination_image_line_step.
 * \param oDstROI Destination ROI.
 * \param aDstQuad Destination quad.
 * \param eInterpolation Interpolation mode: can be NPPI_INTER_NN,
 *        NPPI_INTER_LINEAR or NPPI_INTER_CUBIC.
 * \param nppStreamCtx \ref application_managed_stream_context. 
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref perspective_transform_error_codes
 *
 * @{
 *
 */

/**
 * Single-channel 8-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_C1R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_8u_C1R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel 8-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_C3R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI,  const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI,  const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_C4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                         Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_8u_C4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 8-bit unsigned integer quad-based perspective warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_AC4R_Ctx(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_8u_AC4R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp8u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel planar 8-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_P3R_Ctx(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                         Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_8u_P3R(const Npp8u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp8u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel planar 8-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_8u_P4R_Ctx(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                         Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_8u_P4R(const Npp8u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                     Npp8u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Single-channel 16-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_C1R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_16u_C1R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel 16-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_C3R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_16u_C3R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_C4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_16u_C4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 16-bit unsigned integer quad-based perspective warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_AC4R_Ctx(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                           Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_16u_AC4R(const Npp16u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                       Npp16u * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel planar 16-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_P3R_Ctx(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_16u_P3R(const Npp16u * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp16u * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel planar 16-bit unsigned integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_16u_P4R_Ctx(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_16u_P4R(const Npp16u * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp16u * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Single-channel 32-bit signed integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_C1R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32s_C1R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel 32-bit signed integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_C3R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32s_C3R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 32-bit signed integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_C4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32s_C4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 32-bit signed integer quad-based perspective warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_AC4R_Ctx(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                           Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32s_AC4R(const Npp32s * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                       Npp32s * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel planar 32-bit signed integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_P3R_Ctx(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32s_P3R(const Npp32s * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32s * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel planar 32-bit signed integer quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32s_P4R_Ctx(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);
                                            
NppStatus 
nppiWarpPerspectiveQuad_32s_P4R(const Npp32s * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32s * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
                                            
/**
 * Single-channel 32-bit floating-point quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_C1R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32f_C1R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel 32-bit floating-point quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_C3R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32f_C3R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 32-bit floating-point quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_C4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32f_C4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel 32-bit floating-point quad-based perspective warp, ignoring alpha channel.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPackedPixelParameters">Common parameters for nppiWarpPerspectiveQuad packed pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_AC4R_Ctx(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                           Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32f_AC4R(const Npp32f * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                       Npp32f * pDst, int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Three-channel planar 32-bit floating-point quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_P3R_Ctx(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32f_P3R(const Npp32f * pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32f * pDst[3], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);

/**
 * Four-channel planar 32-bit floating-point quad-based perspective warp.
 * 
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveQuadPlanarPixelParameters">Common parameters for nppiWarpPerspectiveQuad planar pixel functions</a>.
 *
 */
NppStatus 
nppiWarpPerspectiveQuad_32f_P4R_Ctx(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                          Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus 
nppiWarpPerspectiveQuad_32f_P4R(const Npp32f * pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const double aSrcQuad[4][2], 
                                      Npp32f * pDst[4], int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);


/** @} quad_based_perspective_transform */

/** @} image_perspective_transforms */

/** @} image_geometry_transforms */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NV_NPPI_GEOMETRY_TRANSFORMS_H */
