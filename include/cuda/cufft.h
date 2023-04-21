 /* Copyright 2005-2021 NVIDIA Corporation.  All rights reserved.
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

/*!
* \file cufft.h
* \brief Public header file for the NVIDIA CUDA FFT library (CUFFT)
*/

#ifndef _CUFFT_H_
#define _CUFFT_H_


#include "cuComplex.h"
#include "driver_types.h"
#include "library_types.h"

#ifndef CUFFTAPI
#ifdef _WIN32
#define CUFFTAPI __stdcall
#elif __GNUC__ >= 4
#define CUFFTAPI __attribute__ ((visibility ("default")))
#else
#define CUFFTAPI
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CUFFT_VER_MAJOR 10
#define CUFFT_VER_MINOR 7
#define CUFFT_VER_PATCH 1
#define CUFFT_VER_BUILD 0

// cuFFT library version
//
// CUFFT_VERSION / 1000 - major version
// CUFFT_VERSION / 100 % 100 - minor version
// CUFFT_VERSION % 100 - patch level
#define CUFFT_VERSION 10701

// CUFFT API function return values
typedef enum cufftResult_t {
  CUFFT_SUCCESS        = 0x0,
  CUFFT_INVALID_PLAN   = 0x1,
  CUFFT_ALLOC_FAILED   = 0x2,
  CUFFT_INVALID_TYPE   = 0x3,
  CUFFT_INVALID_VALUE  = 0x4,
  CUFFT_INTERNAL_ERROR = 0x5,
  CUFFT_EXEC_FAILED    = 0x6,
  CUFFT_SETUP_FAILED   = 0x7,
  CUFFT_INVALID_SIZE   = 0x8,
  CUFFT_UNALIGNED_DATA = 0x9,
  CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
  CUFFT_INVALID_DEVICE = 0xB,
  CUFFT_PARSE_ERROR = 0xC,
  CUFFT_NO_WORKSPACE = 0xD,
  CUFFT_NOT_IMPLEMENTED = 0xE,
  CUFFT_LICENSE_ERROR = 0x0F,
  CUFFT_NOT_SUPPORTED = 0x10

} cufftResult;

#define MAX_CUFFT_ERROR 0x11


// CUFFT defines and supports the following data types


// cufftReal is a single-precision, floating-point real data type.
// cufftDoubleReal is a double-precision, real data type.
typedef float cufftReal;
typedef double cufftDoubleReal;

// cufftComplex is a single-precision, floating-point complex data type that
// consists of interleaved real and imaginary components.
// cufftDoubleComplex is the double-precision equivalent.
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

// CUFFT transform directions
#define CUFFT_FORWARD -1 // Forward FFT
#define CUFFT_INVERSE  1 // Inverse FFT

// CUFFT supports the following transform types
typedef enum cufftType_t {
  CUFFT_R2C = 0x2a,     // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c,     // Complex (interleaved) to Real
  CUFFT_C2C = 0x29,     // Complex to Complex, interleaved
  CUFFT_D2Z = 0x6a,     // Double to Double-Complex
  CUFFT_Z2D = 0x6c,     // Double-Complex to Double
  CUFFT_Z2Z = 0x69      // Double-Complex to Double-Complex
} cufftType;

// CUFFT supports the following data layouts
typedef enum cufftCompatibility_t {
    CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01    // The default value
} cufftCompatibility;

#define CUFFT_COMPATIBILITY_DEFAULT   CUFFT_COMPATIBILITY_FFTW_PADDING

//
// structure definition used by the shim between old and new APIs
//
#define MAX_SHIM_RANK 3

// cufftHandle is a handle type used to store and access CUFFT plans.
typedef int cufftHandle;


cufftResult CUFFTAPI cufftPlan1d(cufftHandle *plan,
                                 int nx,
                                 cufftType type,
                                 int batch);

cufftResult CUFFTAPI cufftPlan2d(cufftHandle *plan,
                                 int nx, int ny,
                                 cufftType type);

cufftResult CUFFTAPI cufftPlan3d(cufftHandle *plan,
                                 int nx, int ny, int nz,
                                 cufftType type);

cufftResult CUFFTAPI cufftPlanMany(cufftHandle *plan,
                                   int rank,
                                   int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   cufftType type,
                                   int batch);

cufftResult CUFFTAPI cufftMakePlan1d(cufftHandle plan,
                                     int nx,
                                     cufftType type,
                                     int batch,
                                     size_t *workSize);

cufftResult CUFFTAPI cufftMakePlan2d(cufftHandle plan,
                                     int nx, int ny,
                                     cufftType type,
                                     size_t *workSize);

cufftResult CUFFTAPI cufftMakePlan3d(cufftHandle plan,
                                     int nx, int ny, int nz,
                                     cufftType type,
                                     size_t *workSize);

cufftResult CUFFTAPI cufftMakePlanMany(cufftHandle plan,
                                       int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       cufftType type,
                                       int batch,
                                       size_t *workSize);

cufftResult CUFFTAPI cufftMakePlanMany64(cufftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         long long int *onembed,
                                         long long int ostride, long long int odist,
                                         cufftType type,
                                         long long int batch,
                                         size_t * workSize);

cufftResult CUFFTAPI cufftGetSizeMany64(cufftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride, long long int idist,
                                        long long int *onembed,
                                        long long int ostride, long long int odist,
                                        cufftType type,
                                        long long int batch,
                                        size_t *workSize);




cufftResult CUFFTAPI cufftEstimate1d(int nx,
                                     cufftType type,
                                     int batch,
                                     size_t *workSize);

cufftResult CUFFTAPI cufftEstimate2d(int nx, int ny,
                                     cufftType type,
                                     size_t *workSize);

cufftResult CUFFTAPI cufftEstimate3d(int nx, int ny, int nz,
                                     cufftType type,
                                     size_t *workSize);

cufftResult CUFFTAPI cufftEstimateMany(int rank,
                                       int *n,
                                       int *inembed, int istride, int idist,
                                       int *onembed, int ostride, int odist,
                                       cufftType type,
                                       int batch,
                                       size_t *workSize);

cufftResult CUFFTAPI cufftCreate(cufftHandle * handle);

cufftResult CUFFTAPI cufftGetSize1d(cufftHandle handle,
                                    int nx,
                                    cufftType type,
                                    int batch,
                                    size_t *workSize );

cufftResult CUFFTAPI cufftGetSize2d(cufftHandle handle,
                                    int nx, int ny,
                                    cufftType type,
                                    size_t *workSize);

cufftResult CUFFTAPI cufftGetSize3d(cufftHandle handle,
                                    int nx, int ny, int nz,
                                    cufftType type,
                                    size_t *workSize);

cufftResult CUFFTAPI cufftGetSizeMany(cufftHandle handle,
                                      int rank, int *n,
                                      int *inembed, int istride, int idist,
                                      int *onembed, int ostride, int odist,
                                      cufftType type, int batch, size_t *workArea);

cufftResult CUFFTAPI cufftGetSize(cufftHandle handle, size_t *workSize);

cufftResult CUFFTAPI cufftSetWorkArea(cufftHandle plan, void *workArea);

cufftResult CUFFTAPI cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);

cufftResult CUFFTAPI cufftExecC2C(cufftHandle plan,
                                  cufftComplex *idata,
                                  cufftComplex *odata,
                                  int direction);

cufftResult CUFFTAPI cufftExecR2C(cufftHandle plan,
                                  cufftReal *idata,
                                  cufftComplex *odata);

cufftResult CUFFTAPI cufftExecC2R(cufftHandle plan,
                                  cufftComplex *idata,
                                  cufftReal *odata);

cufftResult CUFFTAPI cufftExecZ2Z(cufftHandle plan,
                                  cufftDoubleComplex *idata,
                                  cufftDoubleComplex *odata,
                                  int direction);

cufftResult CUFFTAPI cufftExecD2Z(cufftHandle plan,
                                  cufftDoubleReal *idata,
                                  cufftDoubleComplex *odata);

cufftResult CUFFTAPI cufftExecZ2D(cufftHandle plan,
                                  cufftDoubleComplex *idata,
                                  cufftDoubleReal *odata);


// utility functions
cufftResult CUFFTAPI cufftSetStream(cufftHandle plan,
                                    cudaStream_t stream);

cufftResult CUFFTAPI cufftDestroy(cufftHandle plan);

cufftResult CUFFTAPI cufftGetVersion(int *version);

cufftResult CUFFTAPI cufftGetProperty(libraryPropertyType type,
                                      int *value);

#ifdef __cplusplus
}
#endif

#endif /* _CUFFT_H_ */
