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
#ifndef NV_NPPIDEFS_H
#define NV_NPPIDEFS_H

#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * \file nppdefs.h
 * Typedefinitions and macros for NPP library.
 */
 
#ifdef __cplusplus
extern "C" {
#endif

// Workaround for cuda_fp16.h C incompatibility
typedef 
struct
{
   short fp16;
}
Npp16f;

typedef
struct
{
   short fp16_0;
   short fp16_1;
}
Npp16f_2;

#define NPP_HALF_TO_NPP16F(pHalf) (* reinterpret_cast<Npp16f *>((void *)(pHalf)))

// If this is a 32-bit Windows compile, don't align to 16-byte at all
        // and use a "union-trick" to create 8-byte alignment.
#if defined(_WIN32) && !defined(_WIN64)

            // On 32-bit Windows platforms, do not force 8-byte alignment.
            //   This is a consequence of a limitation of that platform.
    #define NPP_ALIGN_8
            // On 32-bit Windows platforms, do not force 16-byte alignment.
            //   This is a consequence of a limitation of that platform.
    #define NPP_ALIGN_16

#else /* _WIN32 && !_WIN64 */

    #define NPP_ALIGN_8     __align__(8)
    #define NPP_ALIGN_16    __align__(16)

#endif /* !__CUDACC__ && _WIN32 && !_WIN64 */


/** \defgroup typedefs_npp NPP Type Definitions and Constants
 * Definitions of types, structures, enumerations and constants available in the library.
 * @{
 */

/** 
 * Filtering methods.
 */
typedef enum 
{
    NPPI_INTER_UNDEFINED         = 0,
    NPPI_INTER_NN                = 1,        /**<  Nearest neighbor filtering. */
    NPPI_INTER_LINEAR            = 2,        /**<  Linear interpolation. */
    NPPI_INTER_CUBIC             = 4,        /**<  Cubic interpolation. */
    NPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
    NPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
    NPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
    NPPI_INTER_SUPER             = 8,        /**<  Super sampling. */
    NPPI_INTER_LANCZOS           = 16,       /**<  Lanczos filtering. */
    NPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
    NPPI_SMOOTH_EDGE             = (int)0x8000000 /**<  Smooth edge filtering. */
} NppiInterpolationMode; 

/** 
 * Bayer Grid Position Registration.
 */
typedef enum 
{
    NPPI_BAYER_BGGR         = 0,             /**<  Default registration position. */
    NPPI_BAYER_RGGB         = 1,
    NPPI_BAYER_GBRG         = 2,
    NPPI_BAYER_GRBG         = 3
} NppiBayerGridPosition; 

/**
 * Fixed filter-kernel sizes.
 */
typedef enum
{
    NPP_MASK_SIZE_1_X_3,
    NPP_MASK_SIZE_1_X_5,
    NPP_MASK_SIZE_3_X_1 = 100, // leaving space for more 1 X N type enum values 
    NPP_MASK_SIZE_5_X_1,
    NPP_MASK_SIZE_3_X_3 = 200, // leaving space for more N X 1 type enum values
    NPP_MASK_SIZE_5_X_5,
    NPP_MASK_SIZE_7_X_7 = 400,
    NPP_MASK_SIZE_9_X_9 = 500,
    NPP_MASK_SIZE_11_X_11 = 600,
    NPP_MASK_SIZE_13_X_13 = 700,
    NPP_MASK_SIZE_15_X_15 = 800
} NppiMaskSize;

/** 
 * Differential Filter types
 */
 
typedef enum
{
    NPP_FILTER_SOBEL,
    NPP_FILTER_SCHARR,
} NppiDifferentialKernel;

/**
 * Error Status Codes
 *
 * Almost all NPP function return error-status information using
 * these return codes.
 * Negative return codes indicate errors, positive return codes indicate
 * warnings, a return code of 0 indicates success.
 */
typedef enum 
{
    /* negative return-codes indicate errors */
    NPP_NOT_SUPPORTED_MODE_ERROR            = -9999,  
    
    NPP_INVALID_HOST_POINTER_ERROR          = -1032,
    NPP_INVALID_DEVICE_POINTER_ERROR        = -1031,
    NPP_LUT_PALETTE_BITSIZE_ERROR           = -1030,
    NPP_ZC_MODE_NOT_SUPPORTED_ERROR         = -1028,      /**<  ZeroCrossing mode not supported  */
    NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY   = -1027,
    NPP_TEXTURE_BIND_ERROR                  = -1024,
    NPP_WRONG_INTERSECTION_ROI_ERROR        = -1020,
    NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR   = -1006,
    NPP_MEMFREE_ERROR                       = -1005,
    NPP_MEMSET_ERROR                        = -1004,
    NPP_MEMCPY_ERROR                        = -1003,
    NPP_ALIGNMENT_ERROR                     = -1002,
    NPP_CUDA_KERNEL_EXECUTION_ERROR         = -1000,

    NPP_ROUND_MODE_NOT_SUPPORTED_ERROR      = -213,     /**< Unsupported round mode*/
    
    NPP_QUALITY_INDEX_ERROR                 = -210,     /**< Image pixels are constant for quality index */

    NPP_RESIZE_NO_OPERATION_ERROR           = -201,     /**< One of the output image dimensions is less than 1 pixel */

    NPP_OVERFLOW_ERROR                      = -109,     /**< Number overflows the upper or lower limit of the data type */
    NPP_NOT_EVEN_STEP_ERROR                 = -108,     /**< Step value is not pixel multiple */
    NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR    = -107,     /**< Number of levels for histogram is less than 2 */
    NPP_LUT_NUMBER_OF_LEVELS_ERROR          = -106,     /**< Number of levels for LUT is less than 2 */

    NPP_CORRUPTED_DATA_ERROR                = -61,      /**< Processed data is corrupted */
    NPP_CHANNEL_ORDER_ERROR                 = -60,      /**< Wrong order of the destination channels */
    NPP_ZERO_MASK_VALUE_ERROR               = -59,      /**< All values of the mask are zero */
    NPP_QUADRANGLE_ERROR                    = -58,      /**< The quadrangle is nonconvex or degenerates into triangle, line or point */
    NPP_RECTANGLE_ERROR                     = -57,      /**< Size of the rectangle region is less than or equal to 1 */
    NPP_COEFFICIENT_ERROR                   = -56,      /**< Unallowable values of the transformation coefficients   */

    NPP_NUMBER_OF_CHANNELS_ERROR            = -53,      /**< Bad or unsupported number of channels */
    NPP_COI_ERROR                           = -52,      /**< Channel of interest is not 1, 2, or 3 */
    NPP_DIVISOR_ERROR                       = -51,      /**< Divisor is equal to zero */

    NPP_CHANNEL_ERROR                       = -47,      /**< Illegal channel index */
    NPP_STRIDE_ERROR                        = -37,      /**< Stride is less than the row length */
    
    NPP_ANCHOR_ERROR                        = -34,      /**< Anchor point is outside mask */
    NPP_MASK_SIZE_ERROR                     = -33,      /**< Lower bound is larger than upper bound */

    NPP_RESIZE_FACTOR_ERROR                 = -23,
    NPP_INTERPOLATION_ERROR                 = -22,
    NPP_MIRROR_FLIP_ERROR                   = -21,
    NPP_MOMENT_00_ZERO_ERROR                = -20,
    NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR      = -19,
    NPP_THRESHOLD_ERROR                     = -18,
    NPP_CONTEXT_MATCH_ERROR                 = -17,
    NPP_FFT_FLAG_ERROR                      = -16,
    NPP_FFT_ORDER_ERROR                     = -15,
    NPP_STEP_ERROR                          = -14,       /**<  Step is less or equal zero */
    NPP_SCALE_RANGE_ERROR                   = -13,
    NPP_DATA_TYPE_ERROR                     = -12,
    NPP_OUT_OFF_RANGE_ERROR                 = -11,
    NPP_DIVIDE_BY_ZERO_ERROR                = -10,
    NPP_MEMORY_ALLOCATION_ERR               = -9,
    NPP_NULL_POINTER_ERROR                  = -8,
    NPP_RANGE_ERROR                         = -7,
    NPP_SIZE_ERROR                          = -6,
    NPP_BAD_ARGUMENT_ERROR                  = -5,
    NPP_NO_MEMORY_ERROR                     = -4,
    NPP_NOT_IMPLEMENTED_ERROR               = -3,
    NPP_ERROR                               = -2,
    NPP_ERROR_RESERVED                      = -1,
    
    /* success */
    NPP_NO_ERROR                            = 0,        /**<  Error free operation */
    NPP_SUCCESS = NPP_NO_ERROR,                         /**<  Successful operation (same as NPP_NO_ERROR) */

    /* positive return-codes indicate warnings */
    NPP_NO_OPERATION_WARNING                = 1,        /**<  Indicates that no operation was performed */
    NPP_DIVIDE_BY_ZERO_WARNING              = 6,        /**<  Divisor is zero however does not terminate the execution */
    NPP_AFFINE_QUAD_INCORRECT_WARNING       = 28,       /**<  Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded. */
    NPP_WRONG_INTERSECTION_ROI_WARNING      = 29,       /**<  The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed. */
    NPP_WRONG_INTERSECTION_QUAD_WARNING     = 30,       /**<  The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed. */
    NPP_DOUBLE_SIZE_WARNING                 = 35,       /**<  Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing. */
    
    NPP_MISALIGNED_DST_ROI_WARNING          = 10000,    /**<  Speed reduction due to uncoalesced memory accesses warning. */
   
} NppStatus;

typedef struct 
{
    int    major;   /**<  Major version number */
    int    minor;   /**<  Minor version number */
    int    build;   /**<  Build number. This reflects the nightly build this release was made from. */
} NppLibraryVersion;

/** \defgroup npp_basic_types Basic NPP Data Types
 * Definitions of basic types available in the library.
 * @{
 */


typedef unsigned char       Npp8u;     /**<  8-bit unsigned chars */
typedef signed char         Npp8s;     /**<  8-bit signed chars */
typedef unsigned short      Npp16u;    /**<  16-bit unsigned integers */
typedef short               Npp16s;    /**<  16-bit signed integers */
typedef unsigned int        Npp32u;    /**<  32-bit unsigned integers */
typedef int                 Npp32s;    /**<  32-bit signed integers */
typedef unsigned long long  Npp64u;    /**<  64-bit unsigned integers */
typedef long long           Npp64s;    /**<  64-bit signed integers */
typedef float               Npp32f;    /**<  32-bit (IEEE) floating-point numbers */
typedef double              Npp64f;    /**<  64-bit floating-point numbers */


/**
 * Complex Number
 * This struct represents an unsigned char complex number.
 */
typedef struct __align__(2)
{
    Npp8u  re;     /**<  Real part */
    Npp8u  im;     /**<  Imaginary part */
} Npp8uc;

/**
 * Complex Number
 * This struct represents an unsigned short complex number.
 */
typedef struct __align__(4)
{
    Npp16u  re;     /**<  Real part */
    Npp16u  im;     /**<  Imaginary part */
} Npp16uc;

/**
 * Complex Number
 * This struct represents a short complex number.
 */
typedef struct __align__(4)
{
    Npp16s  re;     /**<  Real part */
    Npp16s  im;     /**<  Imaginary part */
} Npp16sc;

/**
 * Complex Number
 * This struct represents an unsigned int complex number.
 */
typedef struct NPP_ALIGN_8
{
    Npp32u  re;     /**<  Real part */
    Npp32u  im;     /**<  Imaginary part */
} Npp32uc;

/**
 * Complex Number
 * This struct represents a signed int complex number.
 */
typedef struct NPP_ALIGN_8
{
    Npp32s  re;     /**<  Real part */
    Npp32s  im;     /**<  Imaginary part */
} Npp32sc;

/**
 * Complex Number
 * This struct represents a single floating-point complex number.
 */
typedef struct NPP_ALIGN_8
{
    Npp32f  re;     /**<  Real part */
    Npp32f  im;     /**<  Imaginary part */
} Npp32fc;

/**
 * Complex Number
 * This struct represents a long long complex number.
 */
typedef struct NPP_ALIGN_16
{
    Npp64s  re;     /**<  Real part */
    Npp64s  im;     /**<  Imaginary part */
} Npp64sc;

/**
 * Complex Number
 * This struct represents a double floating-point complex number.
 */
typedef struct NPP_ALIGN_16
{
    Npp64f  re;     /**<  Real part */
    Npp64f  im;     /**<  Imaginary part */
} Npp64fc;

/*@}*/

#define NPP_MIN_8U      ( 0 )                        /**<  Minimum 8-bit unsigned integer */
#define NPP_MAX_8U      ( 255 )                      /**<  Maximum 8-bit unsigned integer */
#define NPP_MIN_16U     ( 0 )                        /**<  Minimum 16-bit unsigned integer */
#define NPP_MAX_16U     ( 65535 )                    /**<  Maximum 16-bit unsigned integer */
#define NPP_MIN_32U     ( 0 )                        /**<  Minimum 32-bit unsigned integer */
#define NPP_MAX_32U     ( 4294967295U )              /**<  Maximum 32-bit unsigned integer */
#define NPP_MIN_64U     ( 0 )                        /**<  Minimum 64-bit unsigned integer */
#define NPP_MAX_64U     ( 18446744073709551615ULL )  /**<  Maximum 64-bit unsigned integer */

#define NPP_MIN_8S      (-127 - 1 )                  /**<  Minimum 8-bit signed integer */
#define NPP_MAX_8S      ( 127 )                      /**<  Maximum 8-bit signed integer */
#define NPP_MIN_16S     (-32767 - 1 )                /**<  Minimum 16-bit signed integer */
#define NPP_MAX_16S     ( 32767 )                    /**<  Maximum 16-bit signed integer */
#define NPP_MIN_32S     (-2147483647 - 1 )           /**<  Minimum 32-bit signed integer */
#define NPP_MAX_32S     ( 2147483647 )               /**<  Maximum 32-bit signed integer */
#define NPP_MAX_64S     ( 9223372036854775807LL )    /**<  Maximum 64-bit signed integer */
#define NPP_MIN_64S     (-9223372036854775807LL - 1) /**<  Minimum 64-bit signed integer */

#define NPP_MINABS_32F  ( 1.175494351e-38f )         /**<  Smallest positive 32-bit floating point value */
#define NPP_MAXABS_32F  ( 3.402823466e+38f )         /**<  Largest  positive 32-bit floating point value */
#define NPP_MINABS_64F  ( 2.2250738585072014e-308 )  /**<  Smallest positive 64-bit floating point value */
#define NPP_MAXABS_64F  ( 1.7976931348623158e+308 )  /**<  Largest  positive 64-bit floating point value */


/** 
 * 2D Point
 */
typedef struct 
{
    int x;      /**<  x-coordinate. */
    int y;      /**<  y-coordinate. */
} NppiPoint;

/** 
 * 2D Npp32f Point
 */
typedef struct 
{
    Npp32f x;      /**<  x-coordinate. */
    Npp32f y;      /**<  y-coordinate. */
} NppiPoint32f;

/** 
 * 2D Npp64f Point
 */
typedef struct 
{
    Npp64f x;      /**<  x-coordinate. */
    Npp64f y;      /**<  y-coordinate. */
} NppiPoint64f;

/** 
 * 2D Polar Point
 */
typedef struct {
    Npp32f rho;
    Npp32f theta;
} NppPointPolar;

/**
 * 2D Size
 * This struct typically represents the size of a a rectangular region in
 * two space.
 */
typedef struct 
{
    int width;  /**<  Rectangle width. */
    int height; /**<  Rectangle height. */
} NppiSize;

/**
 * 2D Rectangle
 * This struct contains position and size information of a rectangle in 
 * two space.
 * The rectangle's position is usually signified by the coordinate of its
 * upper-left corner.
 */
typedef struct
{
    int x;          /**<  x-coordinate of upper left corner (lowest memory address). */
    int y;          /**<  y-coordinate of upper left corner (lowest memory address). */
    int width;      /**<  Rectangle width. */
    int height;     /**<  Rectangle height. */
} NppiRect;

typedef enum 
{
    NPP_HORIZONTAL_AXIS,
    NPP_VERTICAL_AXIS,
    NPP_BOTH_AXIS
} NppiAxis;

typedef enum 
{
    NPP_CMP_LESS,
    NPP_CMP_LESS_EQ,
    NPP_CMP_EQ,
    NPP_CMP_GREATER_EQ,
    NPP_CMP_GREATER
} NppCmpOp;

/**
 * Rounding Modes
 *
 * The enumerated rounding modes are used by a large number of NPP primitives
 * to allow the user to specify the method by which fractional values are converted
 * to integer values. Also see \ref rounding_modes.
 *
 * For NPP release 5.5 new names for the three rounding modes are introduced that are
 * based on the naming conventions for rounding modes set forth in the IEEE-754
 * floating-point standard. Developers are encouraged to use the new, longer names
 * to be future proof as the legacy names will be deprecated in subsequent NPP releases.
 *
 */
typedef enum 
{
    /** 
     * Round to the nearest even integer.
     * All fractional numbers are rounded to their nearest integer. The ambiguous
     * cases (i.e. \<integer\>.5) are rounded to the closest even integer.
     * E.g.
     * - roundNear(0.5) = 0
     * - roundNear(0.6) = 1
     * - roundNear(1.5) = 2
     * - roundNear(-1.5) = -2
     */
    NPP_RND_NEAR,
    NPP_ROUND_NEAREST_TIES_TO_EVEN = NPP_RND_NEAR, ///< Alias name for ::NPP_RND_NEAR.
    /** 
     * Round according to financial rule.
     * All fractional numbers are rounded to their nearest integer. The ambiguous
     * cases (i.e. \<integer\>.5) are rounded away from zero.
     * E.g.
     * - roundFinancial(0.4)  = 0
     * - roundFinancial(0.5)  = 1
     * - roundFinancial(-1.5) = -2
     */
    NPP_RND_FINANCIAL,
    NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO = NPP_RND_FINANCIAL, ///< Alias name for ::NPP_RND_FINANCIAL. 
    /**
     * Round towards zero (truncation). 
     * All fractional numbers of the form \<integer\>.\<decimals\> are truncated to
     * \<integer\>.
     * - roundZero(1.5) = 1
     * - roundZero(1.9) = 1
     * - roundZero(-2.5) = -2
     */
    NPP_RND_ZERO,
    NPP_ROUND_TOWARD_ZERO = NPP_RND_ZERO, ///< Alias name for ::NPP_RND_ZERO. 
    
    /*
     * Other rounding modes supported by IEEE-754 (2008) floating-point standard:
     *
     * - NPP_ROUND_TOWARD_INFINITY // ceiling
     * - NPP_ROUND_TOWARD_NEGATIVE_INFINITY // floor
     *
     */
} NppRoundMode;

typedef enum  
{
    NPP_BORDER_UNDEFINED        = 0,
    NPP_BORDER_NONE             = NPP_BORDER_UNDEFINED, 
    NPP_BORDER_CONSTANT         = 1,
    NPP_BORDER_REPLICATE        = 2,
    NPP_BORDER_WRAP             = 3,
    NPP_BORDER_MIRROR           = 4
} NppiBorderType;


typedef enum {
    NPP_ALG_HINT_NONE,
    NPP_ALG_HINT_FAST,
    NPP_ALG_HINT_ACCURATE
} NppHintAlgorithm;

/* 
 * Alpha composition controls. 
 */

typedef enum {
    NPPI_OP_ALPHA_OVER,
    NPPI_OP_ALPHA_IN,
    NPPI_OP_ALPHA_OUT,
    NPPI_OP_ALPHA_ATOP,
    NPPI_OP_ALPHA_XOR,
    NPPI_OP_ALPHA_PLUS,
    NPPI_OP_ALPHA_OVER_PREMUL,
    NPPI_OP_ALPHA_IN_PREMUL,
    NPPI_OP_ALPHA_OUT_PREMUL,
    NPPI_OP_ALPHA_ATOP_PREMUL,
    NPPI_OP_ALPHA_XOR_PREMUL,
    NPPI_OP_ALPHA_PLUS_PREMUL,
    NPPI_OP_ALPHA_PREMUL
} NppiAlphaOp;


/** 
 * The NppiHOGConfig structure defines the configuration parameters for the HOG descriptor: 
 */
 
typedef struct 
{
    int      cellSize;             /**<  square cell size (pixels). */
    int      histogramBlockSize;   /**<  square histogram block size (pixels). */
    int      nHistogramBins;       /**<  required number of histogram bins. */
    NppiSize detectionWindowSize;  /**<  detection window size (pixels). */
} NppiHOGConfig;

#define NPP_HOG_MAX_CELL_SIZE                          (16) /**< max horizontal/vertical pixel size of cell.   */
#define NPP_HOG_MAX_BLOCK_SIZE                         (64) /**< max horizontal/vertical pixel size of block.  */
#define NPP_HOG_MAX_BINS_PER_CELL                      (16) /**< max number of histogram bins. */
#define NPP_HOG_MAX_CELLS_PER_DESCRIPTOR              (256) /**< max number of cells in a descriptor window.   */
#define NPP_HOG_MAX_OVERLAPPING_BLOCKS_PER_DESCRIPTOR (256) /**< max number of overlapping blocks in a descriptor window.   */
#define NPP_HOG_MAX_DESCRIPTOR_LOCATIONS_PER_CALL     (128) /**< max number of descriptor window locations per function call.   */

typedef struct
{
    int      numClassifiers;    /**<  number of classifiers */
    Npp32s * classifiers;       /**<  packed classifier data 40 bytes each */
    size_t   classifierStep;
    NppiSize classifierSize;
    Npp32s * counterDevice;
} NppiHaarClassifier_32f;

typedef struct
{
    int      haarBufferSize;    /**<  size of the buffer */
    Npp32s * haarBuffer;        /**<  buffer */
    
} NppiHaarBuffer;

typedef enum {
    nppZCR,    /**<  sign change */
    nppZCXor,  /**<  sign change XOR */
    nppZCC     /**<  sign change count_0 */
} NppsZCType;

typedef enum {
    nppiDCTable,    /**<  DC Table */
    nppiACTable,    /**<  AC Table */
} NppiHuffmanTableType;

typedef enum {
    nppiNormInf = 0, /**<  maximum */ 
    nppiNormL1 = 1,  /**<  sum */
    nppiNormL2 = 2   /**<  square root of sum of squares */
} NppiNorm;

typedef struct
{
	NppiRect oBoundingBox; 		 /**<  x, y, width, height == left, top, right, and bottom pixel coordinates */
	Npp32u nConnectedPixelCount; /**< total number of pixels in connected region */
	Npp32u aSeedPixelValue[3];	 /**< original pixel value of seed pixel, 1 or 3 channel */
} NppiConnectedRegion;

/**
 * General image descriptor. Defines the basic parameters of an image,
 * including data pointer, step, and image size.
 * This can be used by both source and destination images.
 */
typedef struct
{
    void *     pData;  /**< device memory pointer to the image */
    int        nStep;  /**< step size */
    NppiSize  oSize;   /**< width and height of the image */
} NppiImageDescriptor;


typedef struct
{
    void *     pData;        /**< per image device memory pointer to the corresponding buffer */
    int        nBufferSize;  /**< allocated buffer size */
} NppiBufferDescriptor;


/**
 * Provides details of uniquely labeled pixel regions of interest returned 
 * by CompressedLabelMarkersUF function. 
 */

typedef struct
{
    Npp32u nMarkerLabelPixelCount;           /**< total number of pixels in this connected pixel region */
    Npp32u nContourPixelCount;               /**< total number of pixels in this connected pixel region contour */
    Npp32u nContourPixelsFound;              /**< total number of pixels in this connected pixel region contour found during geometry search */
    NppiPoint oContourFirstPixelLocation;    /**< image geometric x and y location of the first pixel in the contour */
    NppiRect oMarkerLabelBoundingBox;        /**< bounding box of this connected pixel region */
} NppiCompressedMarkerLabelsInfo;


/**
 * Provides details of contour pixel grid map location and association 
 */

typedef struct
{
    Npp32u nMarkerLabelID;                   /**< this connected pixel region contour ID */
    Npp32u nContourPixelCount;               /**< total number of pixels in this connected pixel region contour */
    Npp32u nContourStartingPixelOffset;      /**< base offset of starting pixel in the overall contour pixel list */
    Npp32u nSegmentNum;                      /**< relative block segment number within this particular contour ID */
} NppiContourBlockSegment;


/**
 * Provides contour (boundary) geometry info of uniquely labeled pixel regions returned 
 * by nppiCompressedMarkerLabelsUFInfo function in host memory in counterclockwise order relative to contour interiors. 
 */

typedef struct
{
    NppiPoint oContourOrderedGeometryLocation;                 /**< image geometry X and Y location in requested output order */
    NppiPoint oContourPrevPixelLocation;                       /**< image geometry X and Y location of previous contour pixel */
    NppiPoint oContourCenterPixelLocation;                     /**< image geometry X and Y location of center contour pixel */
    NppiPoint oContourNextPixelLocation;                       /**< image geometry X and Y location of next contour pixel */
    Npp32s nOrderIndex;                                        /**< contour pixel counterclockwise order index in geometry list */
    Npp32s nReverseOrderIndex;                                 /**< contour pixel clockwise order index in geometry list */
    Npp32u nFirstIndex;                                        /**< index of first ordered contour pixel in this subgroup */
    Npp32u nLastIndex;                                         /**< index of last ordered contour pixel in this subgroup */
    Npp32u nNextContourPixelIndex;                             /**< index of next ordered contour pixel in NppiContourPixelGeometryInfo list */
    Npp32u nPrevContourPixelIndex;                             /**< index of previous ordered contour pixel in NppiContourPixelGeometryInfo list */
    Npp8u nPixelAlreadyUsed;                                   /**< this test pixel is has already been used */
    Npp8u nAlreadyLinked;                                      /**< this test pixel is already linked to center pixel */
    Npp8u nAlreadyOutput;                                      /**< this pixel has already been output in geometry list */
    Npp8u nContourInteriorDirection;                           /**< direction of contour region interior */
} NppiContourPixelGeometryInfo;

/**
 * Provides contour (boundary) direction info of uniquely labeled pixel regions returned 
 * by CompressedLabelMarkersUF function. 
 */

#define NPP_CONTOUR_DIRECTION_SOUTH_EAST    1
#define NPP_CONTOUR_DIRECTION_SOUTH         2
#define NPP_CONTOUR_DIRECTION_SOUTH_WEST    4
#define NPP_CONTOUR_DIRECTION_WEST          8
#define NPP_CONTOUR_DIRECTION_EAST         16
#define NPP_CONTOUR_DIRECTION_NORTH_EAST   32
#define NPP_CONTOUR_DIRECTION_NORTH        64
#define NPP_CONTOUR_DIRECTION_NORTH_WEST  128

#define NPP_CONTOUR_DIRECTION_ANY_NORTH  NPP_CONTOUR_DIRECTION_NORTH_EAST | NPP_CONTOUR_DIRECTION_NORTH | NPP_CONTOUR_DIRECTION_NORTH_WEST
#define NPP_CONTOUR_DIRECTION_ANY_WEST   NPP_CONTOUR_DIRECTION_NORTH_WEST | NPP_CONTOUR_DIRECTION_WEST | NPP_CONTOUR_DIRECTION_SOUTH_WEST
#define NPP_CONTOUR_DIRECTION_ANY_SOUTH  NPP_CONTOUR_DIRECTION_SOUTH_EAST | NPP_CONTOUR_DIRECTION_SOUTH | NPP_CONTOUR_DIRECTION_SOUTH_WEST
#define NPP_CONTOUR_DIRECTION_ANY_EAST   NPP_CONTOUR_DIRECTION_NORTH_EAST | NPP_CONTOUR_DIRECTION_EAST | NPP_CONTOUR_DIRECTION_SOUTH_EAST

typedef struct
{
    Npp32u  nMarkerLabelID;                         /**< MarkerLabelID of contour interior connected region */
    Npp8u nContourDirectionCenterPixel;             /**< provides current center contour pixel input and output direction info */
    Npp8u nContourInteriorDirectionCenterPixel;     /**< provides current center contour pixel region interior direction info */
    Npp8u nConnected;                               /**< direction to directly connected contour pixels, 0 if not connected */
    Npp8u nGeometryInfoIsValid;
    NppiContourPixelGeometryInfo oContourPixelGeometryInfo;
    NppiPoint nEast1, nNorthEast1, nNorth1, nNorthWest1, nWest1, nSouthWest1, nSouth1, nSouthEast1;
    Npp8u nTest1EastConnected;
    Npp8u nTest1NorthEastConnected;
    Npp8u nTest1NorthConnected;
    Npp8u nTest1NorthWestConnected;
    Npp8u nTest1WestConnected;
    Npp8u nTest1SouthWestConnected;
    Npp8u nTest1SouthConnected;
    Npp8u nTest1SouthEastConnected;
} NppiContourPixelDirectionInfo;

typedef struct
{
    Npp32u nTotalImagePixelContourCount;    /**< total number of contour pixels in image */
    Npp32u nLongestImageContourPixelCount;  /**< longest per contour pixel count in image */
} NppiContourTotalsInfo;


/**
 * Provides control of the type of segment boundaries, if any, added 
 * to the image generated by the watershed segmentation function. 
 */

typedef enum 
{
    NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE,
    NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK,
    NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE,
    NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST,
    NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY
} NppiWatershedSegmentBoundaryType;

    
/** 
 * NPP stream context structure must be filled in by application. 
 * Application should not initialize or alter reserved fields. 
 * 
 */
typedef struct
{
    cudaStream_t hStream;
    int nCudaDeviceId; /* From cudaGetDevice() */
    int nMultiProcessorCount; /* From cudaGetDeviceProperties() */
    int nMaxThreadsPerMultiProcessor; /* From cudaGetDeviceProperties() */
    int nMaxThreadsPerBlock; /* From cudaGetDeviceProperties() */
    size_t nSharedMemPerBlock; /* From cudaGetDeviceProperties */
    int nCudaDevAttrComputeCapabilityMajor; /* From cudaGetDeviceAttribute() */
    int nCudaDevAttrComputeCapabilityMinor; /* From cudaGetDeviceAttribute() */
    unsigned int nStreamFlags; /* From cudaStreamGetFlags() */
    int nReserved0;
} NppStreamContext;

#ifdef __cplusplus
} /* extern "C" */
#endif

/*@}*/
 
#endif /* NV_NPPIDEFS_H */

