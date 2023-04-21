/*
 * Copyright 1993-2021 NVIDIA Corporation. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

/*
 * This is the public header file for the CUBLAS library, defining the API
 *
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines)
 * on top of the CUDA runtime.
 */

#if !defined(CUBLAS_API_H_)
#define CUBLAS_API_H_

#ifndef CUBLASWINAPI
#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI
#endif
#endif

#ifndef CUBLASAPI
#error "This file should not be included without defining CUBLASAPI"
#endif

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "library_types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define CUBLAS_VER_MAJOR 11
#define CUBLAS_VER_MINOR 9
#define CUBLAS_VER_PATCH 2
#define CUBLAS_VER_BUILD 110
#define CUBLAS_VERSION (CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH)

/* CUBLAS status type returns */
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1, CUBLAS_FILL_MODE_FULL = 2 } cublasFillMode_t;

typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;

typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2,
  CUBLAS_OP_HERMITAN = 2, /* synonym if CUBLAS_OP_C */
  CUBLAS_OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
} cublasOperation_t;

typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;

typedef enum { CUBLAS_ATOMICS_NOT_ALLOWED = 0, CUBLAS_ATOMICS_ALLOWED = 1 } cublasAtomicsMode_t;

/*For different GEMM algorithm */
typedef enum {
  CUBLAS_GEMM_DFALT = -1,
  CUBLAS_GEMM_DEFAULT = -1,
  CUBLAS_GEMM_ALGO0 = 0,
  CUBLAS_GEMM_ALGO1 = 1,
  CUBLAS_GEMM_ALGO2 = 2,
  CUBLAS_GEMM_ALGO3 = 3,
  CUBLAS_GEMM_ALGO4 = 4,
  CUBLAS_GEMM_ALGO5 = 5,
  CUBLAS_GEMM_ALGO6 = 6,
  CUBLAS_GEMM_ALGO7 = 7,
  CUBLAS_GEMM_ALGO8 = 8,
  CUBLAS_GEMM_ALGO9 = 9,
  CUBLAS_GEMM_ALGO10 = 10,
  CUBLAS_GEMM_ALGO11 = 11,
  CUBLAS_GEMM_ALGO12 = 12,
  CUBLAS_GEMM_ALGO13 = 13,
  CUBLAS_GEMM_ALGO14 = 14,
  CUBLAS_GEMM_ALGO15 = 15,
  CUBLAS_GEMM_ALGO16 = 16,
  CUBLAS_GEMM_ALGO17 = 17,
  CUBLAS_GEMM_ALGO18 = 18,  // sliced 32x32
  CUBLAS_GEMM_ALGO19 = 19,  // sliced 64x32
  CUBLAS_GEMM_ALGO20 = 20,  // sliced 128x32
  CUBLAS_GEMM_ALGO21 = 21,  // sliced 32x32  -splitK
  CUBLAS_GEMM_ALGO22 = 22,  // sliced 64x32  -splitK
  CUBLAS_GEMM_ALGO23 = 23,  // sliced 128x32 -splitK
  CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
  CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
  CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
  CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
  CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
  CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
  CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
  CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
  CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
  CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
  CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
  CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
  CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
  CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
  CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
  CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
  CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
  CUBLAS_GEMM_ALGO15_TENSOR_OP = 115
} cublasGemmAlgo_t;

/*Enum for default math mode/tensor operation*/
typedef enum {
  CUBLAS_DEFAULT_MATH = 0,

  /* deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
  CUBLAS_TENSOR_OP_MATH = 1,

  /* same as using matching _PEDANTIC compute type when using cublas<T>routine calls or cublasEx() calls with
     cudaDataType as compute type */
  CUBLAS_PEDANTIC_MATH = 2,

  /* allow accelerating single precision routines using TF32 tensor cores */
  CUBLAS_TF32_TENSOR_OP_MATH = 3,

  /* flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines
     with lower size output type */
  CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
} cublasMath_t;

/* For backward compatibility purposes */
typedef cudaDataType cublasDataType_t;

/* Enum for compute type
 *
 * - default types provide best available performance using all available hardware features
 *   and guarantee internal storage precision with at least the same precision and range;
 * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
 * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
 */
typedef enum {
  CUBLAS_COMPUTE_16F = 64,           /* half - default */
  CUBLAS_COMPUTE_16F_PEDANTIC = 65,  /* half - pedantic */
  CUBLAS_COMPUTE_32F = 68,           /* float - default */
  CUBLAS_COMPUTE_32F_PEDANTIC = 69,  /* float - pedantic */
  CUBLAS_COMPUTE_32F_FAST_16F = 74,  /* float - fast, allows down-converting inputs to half or TF32 */
  CUBLAS_COMPUTE_32F_FAST_16BF = 75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
  CUBLAS_COMPUTE_32F_FAST_TF32 = 77, /* float - fast, allows down-converting inputs to TF32 */
  CUBLAS_COMPUTE_64F = 70,           /* double - default */
  CUBLAS_COMPUTE_64F_PEDANTIC = 71,  /* double - pedantic */
  CUBLAS_COMPUTE_32I = 72,           /* signed 32-bit int - default */
  CUBLAS_COMPUTE_32I_PEDANTIC = 73,  /* signed 32-bit int - pedantic */
} cublasComputeType_t;

/* Opaque structure holding CUBLAS library context */
struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int* version);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetProperty(libraryPropertyType type, int* value);
CUBLASAPI size_t CUBLASWINAPI cublasGetCudartVersion(void);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetWorkspace_v2(cublasHandle_t handle,
                                                            void* workspace,
                                                            size_t workspaceSizeInBytes);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget);

CUBLASAPI const char* CUBLASWINAPI cublasGetStatusName(cublasStatus_t status);
CUBLASAPI const char* CUBLASWINAPI cublasGetStatusString(cublasStatus_t status);

/* Cublas logging */
typedef void (*cublasLogCallback)(const char* msg);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLoggerConfigure(int logIsOn,
                                                            int logToStdOut,
                                                            int logToStdErr,
                                                            const char* logFileName);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetLoggerCallback(cublasLogCallback userCallback);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetLoggerCallback(cublasLogCallback* userCallback);

/*
 * cublasStatus_t
 * cublasSetVector (int n, int elemSize, const void *x, int incx,
 *                  void *y, int incy)
 *
 * copies n elements from a vector x in CPU memory space to a vector y
 * in GPU memory space. Elements in both vectors are assumed to have a
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, y points to an object, or part of an object, allocated
 * via cublasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector
 * is equal to 1, this access a column vector while using an increment
 * equal to the leading dimension of the respective matrix accesses a
 * row vector.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy);

/*
 * cublasStatus_t
 * cublasGetVector (int n, int elemSize, const void *x, int incx,
 *                  void *y, int incy)
 *
 * copies n elements from a vector x in GPU memory space to a vector y
 * in CPU memory space. Elements in both vectors are assumed to have a
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, x points to an object, or part of an object, allocated
 * via cublasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector
 * is equal to 1, this access a column vector while using an increment
 * equal to the leading dimension of the respective matrix accesses a
 * row vector.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);

/*
 * cublasStatus_t
 * cublasSetMatrix (int rows, int cols, int elemSize, const void *A,
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in CPU memory
 * space to a matrix B in GPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column
 * major format, with the leading dimension (i.e. number of rows) of
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, B points to an object, or part of an
 * object, that was allocated via cublasAlloc().
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
 *                                ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

/*
 * cublasStatus_t
 * cublasGetMatrix (int rows, int cols, int elemSize, const void *A,
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in GPU memory
 * space to a matrix B in CPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column
 * major format, with the leading dimension (i.e. number of rows) of
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, A points to an object, or part of an
 * object, that was allocated via cublasAlloc().
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

/*
 * cublasStatus
 * cublasSetVectorAsync ( int n, int elemSize, const void *x, int incx,
 *                       void *y, int incy, cudaStream_t stream );
 *
 * cublasSetVectorAsync has the same functionnality as cublasSetVector
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasSetVectorAsync(
    int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream);
/*
 * cublasStatus
 * cublasGetVectorAsync( int n, int elemSize, const void *x, int incx,
 *                       void *y, int incy, cudaStream_t stream)
 *
 * cublasGetVectorAsync has the same functionnality as cublasGetVector
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI cublasGetVectorAsync(
    int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream);

/*
 * cublasStatus_t
 * cublasSetMatrixAsync (int rows, int cols, int elemSize, const void *A,
 *                       int lda, void *B, int ldb, cudaStream_t stream)
 *
 * cublasSetMatrixAsync has the same functionnality as cublasSetMatrix
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or
 *                                ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI
cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);

/*
 * cublasStatus_t
 * cublasGetMatrixAsync (int rows, int cols, int elemSize, const void *A,
 *                       int lda, void *B, int ldb, cudaStream_t stream)
 *
 * cublasGetMatrixAsync has the same functionnality as cublasGetMatrix
 * but the transfer is done asynchronously within the CUDA stream passed
 * in parameter.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus_t CUBLASWINAPI
cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);

CUBLASAPI void CUBLASWINAPI cublasXerbla(const char* srName, int info);
/* ---------------- CUBLAS BLAS1 functions ---------------- */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* result,
                                                   cudaDataType resultType,
                                                   cudaDataType executionType); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle,
                                                  int n,
                                                  const void* x,
                                                  cudaDataType xType,
                                                  int incx,
                                                  const void* y,
                                                  cudaDataType yType,
                                                  int incy,
                                                  void* result,
                                                  cudaDataType resultType,
                                                  cudaDataType executionType);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   const void* y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   void* result,
                                                   cudaDataType resultType,
                                                   cudaDataType executionType);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle,
                                                    int n,
                                                    const float* x,
                                                    int incx,
                                                    const float* y,
                                                    int incy,
                                                    float* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle,
                                                    int n,
                                                    const double* x,
                                                    int incx,
                                                    const double* y,
                                                    int incy,
                                                    double* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* alpha, /* host or device pointer */
                                                   cudaDataType alphaType,
                                                   void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   cudaDataType executionType);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle,
                                                      int n,
                                                      const float* alpha, /* host or device pointer */
                                                      cuComplex* x,
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     cuDoubleComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle,
                                                      int n,
                                                      const double* alpha, /* host or device pointer */
                                                      cuDoubleComplex* x,
                                                      int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* alpha, /* host or device pointer */
                                                   cudaDataType alphaType,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   cudaDataType executiontype);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     float* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     double* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     cuDoubleComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCopyEx(
    cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCswap_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSwapEx(
    cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIdamax_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIamaxEx(
    cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result /* host or device pointer */
);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIsamin_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIdamin_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIaminEx(
    cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result /* host or device pointer */
);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAsumEx(cublasHandle_t handle,
                                                   int n,
                                                   const void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* result,
                                                   cudaDataType resultType, /* host or device pointer */
                                                   cudaDataType executiontype);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(
    cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    float* x,
                                                    int incx,
                                                    float* y,
                                                    int incy,
                                                    const float* c,  /* host or device pointer */
                                                    const float* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    double* x,
                                                    int incx,
                                                    double* y,
                                                    int incy,
                                                    const double* c,  /* host or device pointer */
                                                    const double* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    cuComplex* x,
                                                    int incx,
                                                    cuComplex* y,
                                                    int incy,
                                                    const float* c,      /* host or device pointer */
                                                    const cuComplex* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle,
                                                     int n,
                                                     cuComplex* x,
                                                     int incx,
                                                     cuComplex* y,
                                                     int incy,
                                                     const float* c,  /* host or device pointer */
                                                     const float* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle,
                                                    int n,
                                                    cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* y,
                                                    int incy,
                                                    const double* c,           /* host or device pointer */
                                                    const cuDoubleComplex* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex* x,
                                                     int incx,
                                                     cuDoubleComplex* y,
                                                     int incy,
                                                     const double* c,  /* host or device pointer */
                                                     const double* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx(cublasHandle_t handle,
                                                  int n,
                                                  void* x,
                                                  cudaDataType xType,
                                                  int incx,
                                                  void* y,
                                                  cudaDataType yType,
                                                  int incy,
                                                  const void* c, /* host or device pointer */
                                                  const void* s,
                                                  cudaDataType csType,
                                                  cudaDataType executiontype);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle,
                                                     float* a,  /* host or device pointer */
                                                     float* b,  /* host or device pointer */
                                                     float* c,  /* host or device pointer */
                                                     float* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle,
                                                     double* a,  /* host or device pointer */
                                                     double* b,  /* host or device pointer */
                                                     double* c,  /* host or device pointer */
                                                     double* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle,
                                                     cuComplex* a,  /* host or device pointer */
                                                     cuComplex* b,  /* host or device pointer */
                                                     float* c,      /* host or device pointer */
                                                     cuComplex* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle,
                                                     cuDoubleComplex* a,  /* host or device pointer */
                                                     cuDoubleComplex* b,  /* host or device pointer */
                                                     double* c,           /* host or device pointer */
                                                     cuDoubleComplex* s); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotgEx(cublasHandle_t handle,
                                                   void* a, /* host or device pointer */
                                                   void* b, /* host or device pointer */
                                                   cudaDataType abType,
                                                   void* c, /* host or device pointer */
                                                   void* s, /* host or device pointer */
                                                   cudaDataType csType,
                                                   cudaDataType executiontype);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle,
                                                     int n,
                                                     float* x,
                                                     int incx,
                                                     float* y,
                                                     int incy,
                                                     const float* param); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle,
                                                     int n,
                                                     double* x,
                                                     int incx,
                                                     double* y,
                                                     int incy,
                                                     const double* param); /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmEx(cublasHandle_t handle,
                                                   int n,
                                                   void* x,
                                                   cudaDataType xType,
                                                   int incx,
                                                   void* y,
                                                   cudaDataType yType,
                                                   int incy,
                                                   const void* param, /* host or device pointer */
                                                   cudaDataType paramType,
                                                   cudaDataType executiontype);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle,
                                                      float* d1,       /* host or device pointer */
                                                      float* d2,       /* host or device pointer */
                                                      float* x1,       /* host or device pointer */
                                                      const float* y1, /* host or device pointer */
                                                      float* param);   /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle,
                                                      double* d1,       /* host or device pointer */
                                                      double* d2,       /* host or device pointer */
                                                      double* x1,       /* host or device pointer */
                                                      const double* y1, /* host or device pointer */
                                                      double* param);   /* host or device pointer */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotmgEx(cublasHandle_t handle,
                                                    void* d1, /* host or device pointer */
                                                    cudaDataType d1Type,
                                                    void* d2, /* host or device pointer */
                                                    cudaDataType d2Type,
                                                    void* x1, /* host or device pointer */
                                                    cudaDataType x1Type,
                                                    const void* y1, /* host or device pointer */
                                                    cudaDataType y1Type,
                                                    void* param, /* host or device pointer */
                                                    cudaDataType paramType,
                                                    cudaDataType executiontype);
/* --------------- CUBLAS BLAS2 functions  ---------------- */

/* GEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);
/* GBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle,
                                                     cublasOperation_t trans,
                                                     int m,
                                                     int n,
                                                     int kl,
                                                     int ku,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* TRMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* AP,
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* AP,
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* AP,
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* AP,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TRSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* TPSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const float* AP,
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const double* AP,
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuComplex* AP,
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     const cuDoubleComplex* AP,
                                                     cuDoubleComplex* x,
                                                     int incx);
/* TBSV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const float* A,
                                                     int lda,
                                                     float* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const double* A,
                                                     int lda,
                                                     double* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* x,
                                                     int incx);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* x,
                                                     int incx);

/* SYMV/HEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* SBMV/HBMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* SPMV/HPMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* AP,
                                                     const float* x,
                                                     int incx,
                                                     const float* beta, /* host or device pointer */
                                                     float* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* AP,
                                                     const double* x,
                                                     int incx,
                                                     const double* beta, /* host or device pointer */
                                                     double* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* AP,
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* y,
                                                     int incy);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* AP,
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* y,
                                                     int incy);

/* GER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle,
                                                    int m,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    const float* y,
                                                    int incy,
                                                    float* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle,
                                                    int m,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    const double* y,
                                                    int incy,
                                                    double* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

/* SYR/HER */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    float* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    double* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const cuDoubleComplex* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* A,
                                                    int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* A,
                                                    int lda);

/* SPR/HPR */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const float* x,
                                                    int incx,
                                                    float* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const double* x,
                                                    int incx,
                                                    double* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const float* alpha, /* host or device pointer */
                                                    const cuComplex* x,
                                                    int incx,
                                                    cuComplex* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    int n,
                                                    const double* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* x,
                                                    int incx,
                                                    cuDoubleComplex* AP);

/* SYR2/HER2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     const float* y,
                                                     int incy,
                                                     float* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     const double* y,
                                                     int incy,
                                                     double* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* A,
                                                     int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* A,
                                                     int lda);

/* SPR2/HPR2 */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* x,
                                                     int incx,
                                                     const float* y,
                                                     int incy,
                                                     float* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* x,
                                                     int incx,
                                                     const double* y,
                                                     int incy,
                                                     double* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* x,
                                                     int incx,
                                                     const cuComplex* y,
                                                     int incy,
                                                     cuComplex* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* x,
                                                     int incx,
                                                     const cuDoubleComplex* y,
                                                     int incy,
                                                     cuDoubleComplex* AP);
/* BATCH GEMV */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemvBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const float* alpha, /* host or device pointer */
                                                         const float* const Aarray[],
                                                         int lda,
                                                         const float* const xarray[],
                                                         int incx,
                                                         const float* beta, /* host or device pointer */
                                                         float* const yarray[],
                                                         int incy,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemvBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const double* alpha, /* host or device pointer */
                                                         const double* const Aarray[],
                                                         int lda,
                                                         const double* const xarray[],
                                                         int incx,
                                                         const double* beta, /* host or device pointer */
                                                         double* const yarray[],
                                                         int incy,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemvBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const cuComplex* alpha, /* host or device pointer */
                                                         const cuComplex* const Aarray[],
                                                         int lda,
                                                         const cuComplex* const xarray[],
                                                         int incx,
                                                         const cuComplex* beta, /* host or device pointer */
                                                         cuComplex* const yarray[],
                                                         int incy,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemvBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         const cuDoubleComplex* alpha, /* host or device pointer */
                                                         const cuDoubleComplex* const Aarray[],
                                                         int lda,
                                                         const cuDoubleComplex* const xarray[],
                                                         int incx,
                                                         const cuDoubleComplex* beta, /* host or device pointer */
                                                         cuDoubleComplex* const yarray[],
                                                         int incy,
                                                         int batchCount);

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHSHgemvBatched(cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __half* const Aarray[],
                                                           int lda,
                                                           const __half* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           __half* const yarray[],
                                                           int incy,
                                                           int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHSSgemvBatched(cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __half* const Aarray[],
                                                           int lda,
                                                           const __half* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           float* const yarray[],
                                                           int incy,
                                                           int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasTSTgemvBatched(cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __nv_bfloat16* const Aarray[],
                                                           int lda,
                                                           const __nv_bfloat16* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           __nv_bfloat16* const yarray[],
                                                           int incy,
                                                           int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasTSSgemvBatched(cublasHandle_t handle,
                                                           cublasOperation_t trans,
                                                           int m,
                                                           int n,
                                                           const float* alpha, /* host or device pointer */
                                                           const __nv_bfloat16* const Aarray[],
                                                           int lda,
                                                           const __nv_bfloat16* const xarray[],
                                                           int incx,
                                                           const float* beta, /* host or device pointer */
                                                           float* const yarray[],
                                                           int incy,
                                                           int batchCount);
#endif

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemvStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t trans,
                                                                int m,
                                                                int n,
                                                                const float* alpha, /* host or device pointer */
                                                                const float* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const float* x,
                                                                int incx,
                                                                long long int stridex,
                                                                const float* beta, /* host or device pointer */
                                                                float* y,
                                                                int incy,
                                                                long long int stridey,
                                                                int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemvStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t trans,
                                                                int m,
                                                                int n,
                                                                const double* alpha, /* host or device pointer */
                                                                const double* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const double* x,
                                                                int incx,
                                                                long long int stridex,
                                                                const double* beta, /* host or device pointer */
                                                                double* y,
                                                                int incy,
                                                                long long int stridey,
                                                                int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemvStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t trans,
                                                                int m,
                                                                int n,
                                                                const cuComplex* alpha, /* host or device pointer */
                                                                const cuComplex* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const cuComplex* x,
                                                                int incx,
                                                                long long int stridex,
                                                                const cuComplex* beta, /* host or device pointer */
                                                                cuComplex* y,
                                                                int incy,
                                                                long long int stridey,
                                                                int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgemvStridedBatched(cublasHandle_t handle,
                          cublasOperation_t trans,
                          int m,
                          int n,
                          const cuDoubleComplex* alpha, /* host or device pointer */
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* x,
                          int incx,
                          long long int stridex,
                          const cuDoubleComplex* beta, /* host or device pointer */
                          cuDoubleComplex* y,
                          int incy,
                          long long int stridey,
                          int batchCount);

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHSHgemvStridedBatched(cublasHandle_t handle,
                                                                  cublasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __half* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __half* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  __half* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHSSgemvStridedBatched(cublasHandle_t handle,
                                                                  cublasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __half* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __half* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  float* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasTSTgemvStridedBatched(cublasHandle_t handle,
                                                                  cublasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __nv_bfloat16* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __nv_bfloat16* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  __nv_bfloat16* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasTSSgemvStridedBatched(cublasHandle_t handle,
                                                                  cublasOperation_t trans,
                                                                  int m,
                                                                  int n,
                                                                  const float* alpha, /* host or device pointer */
                                                                  const __nv_bfloat16* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const __nv_bfloat16* x,
                                                                  int incx,
                                                                  long long int stridex,
                                                                  const float* beta, /* host or device pointer */
                                                                  float* y,
                                                                  int incy,
                                                                  long long int stridey,
                                                                  int batchCount);
#endif
/* ---------------- CUBLAS BLAS3 functions ---------------- */

/* GEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3m(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const cuComplex* A,
                                                    int lda,
                                                    const cuComplex* B,
                                                    int ldb,
                                                    const cuComplex* beta, /* host or device pointer */
                                                    cuComplex* C,
                                                    int ldc);
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mEx(cublasHandle_t handle,
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha,
                                                      const void* A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void* B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      const cuComplex* beta,
                                                      void* C,
                                                      cudaDataType Ctype,
                                                      int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm3m(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex* alpha, /* host or device pointer */
                                                    const cuDoubleComplex* A,
                                                    int lda,
                                                    const cuDoubleComplex* B,
                                                    int ldb,
                                                    const cuDoubleComplex* beta, /* host or device pointer */
                                                    cuDoubleComplex* C,
                                                    int ldc);

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  int k,
                                                  const __half* alpha, /* host or device pointer */
                                                  const __half* A,
                                                  int lda,
                                                  const __half* B,
                                                  int ldb,
                                                  const __half* beta, /* host or device pointer */
                                                  __half* C,
                                                  int ldc);
#endif
/* IO in FP16/FP32, computation in float */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmEx(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const float* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const void* B,
                                                    cudaDataType Btype,
                                                    int ldb,
                                                    const float* beta, /* host or device pointer */
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle,
                                                   cublasOperation_t transa,
                                                   cublasOperation_t transb,
                                                   int m,
                                                   int n,
                                                   int k,
                                                   const void* alpha, /* host or device pointer */
                                                   const void* A,
                                                   cudaDataType Atype,
                                                   int lda,
                                                   const void* B,
                                                   cudaDataType Btype,
                                                   int ldb,
                                                   const void* beta, /* host or device pointer */
                                                   void* C,
                                                   cudaDataType Ctype,
                                                   int ldc,
                                                   cublasComputeType_t computeType,
                                                   cublasGemmAlgo_t algo);

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmEx(cublasHandle_t handle,
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb,
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha,
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const void* B,
                                                    cudaDataType Btype,
                                                    int ldb,
                                                    const cuComplex* beta,
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasUint8gemmBias(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          cublasOperation_t transc,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const unsigned char* A,
                                                          int A_bias,
                                                          int lda,
                                                          const unsigned char* B,
                                                          int B_bias,
                                                          int ldb,
                                                          unsigned char* C,
                                                          int C_bias,
                                                          int ldc,
                                                          int C_mult,
                                                          int C_shift);

/* SYRK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);
/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkEx(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const cuComplex* beta, /* host or device pointer */
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc);

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha,
                                                      const void* A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const cuComplex* beta,
                                                      void* C,
                                                      cudaDataType Ctype,
                                                      int ldc);

/* HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const float* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const float* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     int n,
                                                     int k,
                                                     const double* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const double* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

/* IO in Int8 complex/cuComplex, computation in cuComplex */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkEx(cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t trans,
                                                    int n,
                                                    int k,
                                                    const float* alpha, /* host or device pointer */
                                                    const void* A,
                                                    cudaDataType Atype,
                                                    int lda,
                                                    const float* beta, /* host or device pointer */
                                                    void* C,
                                                    cudaDataType Ctype,
                                                    int ldc);

/* IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float* alpha,
                                                      const void* A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const float* beta,
                                                      void* C,
                                                      cudaDataType Ctype,
                                                      int ldc);

/* SYR2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const float* alpha, /* host or device pointer */
                                                      const float* A,
                                                      int lda,
                                                      const float* B,
                                                      int ldb,
                                                      const float* beta, /* host or device pointer */
                                                      float* C,
                                                      int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const double* alpha, /* host or device pointer */
                                                      const double* A,
                                                      int lda,
                                                      const double* B,
                                                      int ldb,
                                                      const double* beta, /* host or device pointer */
                                                      double* C,
                                                      int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha, /* host or device pointer */
                                                      const cuComplex* A,
                                                      int lda,
                                                      const cuComplex* B,
                                                      int ldb,
                                                      const cuComplex* beta, /* host or device pointer */
                                                      cuComplex* C,
                                                      int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex* alpha, /* host or device pointer */
                                                      const cuDoubleComplex* A,
                                                      int lda,
                                                      const cuDoubleComplex* B,
                                                      int ldb,
                                                      const cuDoubleComplex* beta, /* host or device pointer */
                                                      cuDoubleComplex* C,
                                                      int ldc);
/* HER2K */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex* alpha, /* host or device pointer */
                                                      const cuComplex* A,
                                                      int lda,
                                                      const cuComplex* B,
                                                      int ldb,
                                                      const float* beta, /* host or device pointer */
                                                      cuComplex* C,
                                                      int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex* alpha, /* host or device pointer */
                                                      const cuDoubleComplex* A,
                                                      int lda,
                                                      const cuDoubleComplex* B,
                                                      int ldb,
                                                      const double* beta, /* host or device pointer */
                                                      cuDoubleComplex* C,
                                                      int ldc);
/* SYRKX : eXtended SYRK*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const float* alpha, /* host or device pointer */
                                                   const float* A,
                                                   int lda,
                                                   const float* B,
                                                   int ldb,
                                                   const float* beta, /* host or device pointer */
                                                   float* C,
                                                   int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const double* alpha, /* host or device pointer */
                                                   const double* A,
                                                   int lda,
                                                   const double* B,
                                                   int ldb,
                                                   const double* beta, /* host or device pointer */
                                                   double* C,
                                                   int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex* alpha, /* host or device pointer */
                                                   const cuComplex* A,
                                                   int lda,
                                                   const cuComplex* B,
                                                   int ldb,
                                                   const cuComplex* beta, /* host or device pointer */
                                                   cuComplex* C,
                                                   int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuDoubleComplex* alpha, /* host or device pointer */
                                                   const cuDoubleComplex* A,
                                                   int lda,
                                                   const cuDoubleComplex* B,
                                                   int ldb,
                                                   const cuDoubleComplex* beta, /* host or device pointer */
                                                   cuDoubleComplex* C,
                                                   int ldc);
/* HERKX : eXtended HERK */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuComplex* alpha, /* host or device pointer */
                                                   const cuComplex* A,
                                                   int lda,
                                                   const cuComplex* B,
                                                   int ldb,
                                                   const float* beta, /* host or device pointer */
                                                   cuComplex* C,
                                                   int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx(cublasHandle_t handle,
                                                   cublasFillMode_t uplo,
                                                   cublasOperation_t trans,
                                                   int n,
                                                   int k,
                                                   const cuDoubleComplex* alpha, /* host or device pointer */
                                                   const cuDoubleComplex* A,
                                                   int lda,
                                                   const cuDoubleComplex* B,
                                                   int ldb,
                                                   const double* beta, /* host or device pointer */
                                                   cuDoubleComplex* C,
                                                   int ldc);
/* SYMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta, /* host or device pointer */
                                                     float* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     const double* beta, /* host or device pointer */
                                                     double* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

/* HEMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     const cuComplex* beta, /* host or device pointer */
                                                     cuComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     const cuDoubleComplex* beta, /* host or device pointer */
                                                     cuDoubleComplex* C,
                                                     int ldc);

/* TRSM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     float* B,
                                                     int ldb);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     double* B,
                                                     int ldb);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     cuComplex* B,
                                                     int ldb);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     cuDoubleComplex* B,
                                                     int ldb);

/* TRMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const float* alpha, /* host or device pointer */
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     float* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const double* alpha, /* host or device pointer */
                                                     const double* A,
                                                     int lda,
                                                     const double* B,
                                                     int ldb,
                                                     double* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex* alpha, /* host or device pointer */
                                                     const cuComplex* A,
                                                     int lda,
                                                     const cuComplex* B,
                                                     int ldb,
                                                     cuComplex* C,
                                                     int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t trans,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex* alpha, /* host or device pointer */
                                                     const cuDoubleComplex* A,
                                                     int lda,
                                                     const cuDoubleComplex* B,
                                                     int ldb,
                                                     cuDoubleComplex* C,
                                                     int ldc);
/* BATCH GEMM */
#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const __half* alpha, /* host or device pointer */
                                                         const __half* const Aarray[],
                                                         int lda,
                                                         const __half* const Barray[],
                                                         int ldb,
                                                         const __half* beta, /* host or device pointer */
                                                         __half* const Carray[],
                                                         int ldc,
                                                         int batchCount);
#endif
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const float* alpha, /* host or device pointer */
                                                         const float* const Aarray[],
                                                         int lda,
                                                         const float* const Barray[],
                                                         int ldb,
                                                         const float* beta, /* host or device pointer */
                                                         float* const Carray[],
                                                         int ldc,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const double* alpha, /* host or device pointer */
                                                         const double* const Aarray[],
                                                         int lda,
                                                         const double* const Barray[],
                                                         int ldb,
                                                         const double* beta, /* host or device pointer */
                                                         double* const Carray[],
                                                         int ldc,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const cuComplex* alpha, /* host or device pointer */
                                                         const cuComplex* const Aarray[],
                                                         int lda,
                                                         const cuComplex* const Barray[],
                                                         int ldb,
                                                         const cuComplex* beta, /* host or device pointer */
                                                         cuComplex* const Carray[],
                                                         int ldc,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mBatched(cublasHandle_t handle,
                                                           cublasOperation_t transa,
                                                           cublasOperation_t transb,
                                                           int m,
                                                           int n,
                                                           int k,
                                                           const cuComplex* alpha, /* host or device pointer */
                                                           const cuComplex* const Aarray[],
                                                           int lda,
                                                           const cuComplex* const Barray[],
                                                           int ldb,
                                                           const cuComplex* beta, /* host or device pointer */
                                                           cuComplex* const Carray[],
                                                           int ldc,
                                                           int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle,
                                                         cublasOperation_t transa,
                                                         cublasOperation_t transb,
                                                         int m,
                                                         int n,
                                                         int k,
                                                         const cuDoubleComplex* alpha, /* host or device pointer */
                                                         const cuDoubleComplex* const Aarray[],
                                                         int lda,
                                                         const cuDoubleComplex* const Barray[],
                                                         int ldb,
                                                         const cuDoubleComplex* beta, /* host or device pointer */
                                                         cuDoubleComplex* const Carray[],
                                                         int ldc,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle,
                                                          cublasOperation_t transa,
                                                          cublasOperation_t transb,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const void* alpha, /* host or device pointer */
                                                          const void* const Aarray[],
                                                          cudaDataType Atype,
                                                          int lda,
                                                          const void* const Barray[],
                                                          cudaDataType Btype,
                                                          int ldb,
                                                          const void* beta, /* host or device pointer */
                                                          void* const Carray[],
                                                          cudaDataType Ctype,
                                                          int ldc,
                                                          int batchCount,
                                                          cublasComputeType_t computeType,
                                                          cublasGemmAlgo_t algo);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                                                 cublasOperation_t transa,
                                                                 cublasOperation_t transb,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const void* alpha, /* host or device pointer */
                                                                 const void* A,
                                                                 cudaDataType Atype,
                                                                 int lda,
                                                                 long long int strideA, /* purposely signed */
                                                                 const void* B,
                                                                 cudaDataType Btype,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const void* beta, /* host or device pointer */
                                                                 void* C,
                                                                 cudaDataType Ctype,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount,
                                                                 cublasComputeType_t computeType,
                                                                 cublasGemmAlgo_t algo);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const float* alpha, /* host or device pointer */
                                                                const float* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const float* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const float* beta, /* host or device pointer */
                                                                float* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const double* alpha, /* host or device pointer */
                                                                const double* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const double* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const double* beta, /* host or device pointer */
                                                                double* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const cuComplex* alpha, /* host or device pointer */
                                                                const cuComplex* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const cuComplex* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const cuComplex* beta, /* host or device pointer */
                                                                cuComplex* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm3mStridedBatched(cublasHandle_t handle,
                                                                  cublasOperation_t transa,
                                                                  cublasOperation_t transb,
                                                                  int m,
                                                                  int n,
                                                                  int k,
                                                                  const cuComplex* alpha, /* host or device pointer */
                                                                  const cuComplex* A,
                                                                  int lda,
                                                                  long long int strideA, /* purposely signed */
                                                                  const cuComplex* B,
                                                                  int ldb,
                                                                  long long int strideB,
                                                                  const cuComplex* beta, /* host or device pointer */
                                                                  cuComplex* C,
                                                                  int ldc,
                                                                  long long int strideC,
                                                                  int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgemmStridedBatched(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const cuDoubleComplex* alpha, /* host or device pointer */
                          const cuDoubleComplex* A,
                          int lda,
                          long long int strideA, /* purposely signed */
                          const cuDoubleComplex* B,
                          int ldb,
                          long long int strideB,
                          const cuDoubleComplex* beta, /* host or device poi */
                          cuDoubleComplex* C,
                          int ldc,
                          long long int strideC,
                          int batchCount);

#if defined(__cplusplus)
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa,
                                                                cublasOperation_t transb,
                                                                int m,
                                                                int n,
                                                                int k,
                                                                const __half* alpha, /* host or device pointer */
                                                                const __half* A,
                                                                int lda,
                                                                long long int strideA, /* purposely signed */
                                                                const __half* B,
                                                                int ldb,
                                                                long long int strideB,
                                                                const __half* beta, /* host or device pointer */
                                                                __half* C,
                                                                int ldc,
                                                                long long int strideC,
                                                                int batchCount);
#endif
/* ---------------- CUBLAS BLAS-like extension ---------------- */
/* GEAM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const float* alpha, /* host or device pointer */
                                                  const float* A,
                                                  int lda,
                                                  const float* beta, /* host or device pointer */
                                                  const float* B,
                                                  int ldb,
                                                  float* C,
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const double* alpha, /* host or device pointer */
                                                  const double* A,
                                                  int lda,
                                                  const double* beta, /* host or device pointer */
                                                  const double* B,
                                                  int ldb,
                                                  double* C,
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuComplex* alpha, /* host or device pointer */
                                                  const cuComplex* A,
                                                  int lda,
                                                  const cuComplex* beta, /* host or device pointer */
                                                  const cuComplex* B,
                                                  int ldb,
                                                  cuComplex* C,
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex* alpha, /* host or device pointer */
                                                  const cuDoubleComplex* A,
                                                  int lda,
                                                  const cuDoubleComplex* beta, /* host or device pointer */
                                                  const cuDoubleComplex* B,
                                                  int ldb,
                                                  cuDoubleComplex* C,
                                                  int ldc);

/* Batched LU - GETRF*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          float* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          double* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          cuComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle,
                                                          int n,
                                                          cuDoubleComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          int* P,    /*Device Pointer*/
                                                          int* info, /*Device Pointer*/
                                                          int batchSize);

/* Batched inversion based on LU factorization from getrf */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const float* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,     /*Device pointer*/
                                                          float* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const double* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,      /*Device pointer*/
                                                          double* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,         /*Device pointer*/
                                                          cuComplex* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuDoubleComplex* const A[], /*Device pointer*/
                                                          int lda,
                                                          const int* P,               /*Device pointer*/
                                                          cuDoubleComplex* const C[], /*Device pointer*/
                                                          int ldc,
                                                          int* info,
                                                          int batchSize);

/* Batched solver based on LU factorization from getrf */

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const float* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          float* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const double* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          double* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const cuComplex* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          cuComplex* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrsBatched(cublasHandle_t handle,
                                                          cublasOperation_t trans,
                                                          int n,
                                                          int nrhs,
                                                          const cuDoubleComplex* const Aarray[],
                                                          int lda,
                                                          const int* devIpiv,
                                                          cuDoubleComplex* const Barray[],
                                                          int ldb,
                                                          int* info,
                                                          int batchSize);

/* TRSM - Batched Triangular Solver */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const float* alpha, /*Host or Device Pointer*/
                                                         const float* const A[],
                                                         int lda,
                                                         float* const B[],
                                                         int ldb,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const double* alpha, /*Host or Device Pointer*/
                                                         const double* const A[],
                                                         int lda,
                                                         double* const B[],
                                                         int ldb,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const cuComplex* alpha, /*Host or Device Pointer*/
                                                         const cuComplex* const A[],
                                                         int lda,
                                                         cuComplex* const B[],
                                                         int ldb,
                                                         int batchCount);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched(cublasHandle_t handle,
                                                         cublasSideMode_t side,
                                                         cublasFillMode_t uplo,
                                                         cublasOperation_t trans,
                                                         cublasDiagType_t diag,
                                                         int m,
                                                         int n,
                                                         const cuDoubleComplex* alpha, /*Host or Device Pointer*/
                                                         const cuDoubleComplex* const A[],
                                                         int lda,
                                                         cuDoubleComplex* const B[],
                                                         int ldb,
                                                         int batchCount);

/* Batched - MATINV*/
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const float* const A[], /*Device pointer*/
                                                           int lda,
                                                           float* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const double* const A[], /*Device pointer*/
                                                           int lda,
                                                           double* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const cuComplex* const A[], /*Device pointer*/
                                                           int lda,
                                                           cuComplex* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZmatinvBatched(cublasHandle_t handle,
                                                           int n,
                                                           const cuDoubleComplex* const A[], /*Device pointer*/
                                                           int lda,
                                                           cuDoubleComplex* const Ainv[], /*Device pointer*/
                                                           int lda_inv,
                                                           int* info, /*Device Pointer*/
                                                           int batchSize);

/* Batch QR Factorization */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          float* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          float* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          double* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          double* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          cuComplex* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          cuComplex* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeqrfBatched(cublasHandle_t handle,
                                                          int m,
                                                          int n,
                                                          cuDoubleComplex* const Aarray[], /*Device pointer*/
                                                          int lda,
                                                          cuDoubleComplex* const TauArray[], /*Device pointer*/
                                                          int* info,
                                                          int batchSize);
/* Least Square Min only m >= n and Non-transpose supported */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         float* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         float* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray, /*Device pointer*/
                                                         int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         double* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         double* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray, /*Device pointer*/
                                                         int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         cuComplex* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         cuComplex* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray,
                                                         int batchSize);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgelsBatched(cublasHandle_t handle,
                                                         cublasOperation_t trans,
                                                         int m,
                                                         int n,
                                                         int nrhs,
                                                         cuDoubleComplex* const Aarray[], /*Device pointer*/
                                                         int lda,
                                                         cuDoubleComplex* const Carray[], /*Device pointer*/
                                                         int ldc,
                                                         int* info,
                                                         int* devInfoArray,
                                                         int batchSize);
/* DGMM */
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const float* A,
                                                  int lda,
                                                  const float* x,
                                                  int incx,
                                                  float* C,
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const double* A,
                                                  int lda,
                                                  const double* x,
                                                  int incx,
                                                  double* C,
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuComplex* A,
                                                  int lda,
                                                  const cuComplex* x,
                                                  int incx,
                                                  cuComplex* C,
                                                  int ldc);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex* A,
                                                  int lda,
                                                  const cuDoubleComplex* x,
                                                  int incx,
                                                  cuDoubleComplex* C,
                                                  int ldc);

/* TPTTR : Triangular Pack format to Triangular format */
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* AP, float* A, int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* AP, double* A, int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpttr(
    cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda);
/* TRTTP : Triangular format to Triangular Pack format */
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, float* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, double* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP);

CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrttp(
    cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP);

#if defined(__cplusplus)
}

static inline cublasStatus_t cublasMigrateComputeType(cublasHandle_t handle,
                                                      cudaDataType_t dataType,
                                                      cublasComputeType_t* computeType) {
  cublasMath_t mathMode = CUBLAS_DEFAULT_MATH;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  status = cublasGetMathMode(handle, &mathMode);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  bool isPedantic = ((mathMode & 0xf) == CUBLAS_PEDANTIC_MATH);

  switch (dataType) {
    case CUDA_R_32F:
    case CUDA_C_32F:
      *computeType = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
      return CUBLAS_STATUS_SUCCESS;
    case CUDA_R_64F:
    case CUDA_C_64F:
      *computeType = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
      return CUBLAS_STATUS_SUCCESS;
    case CUDA_R_16F:
      *computeType = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
      return CUBLAS_STATUS_SUCCESS;
    case CUDA_R_32I:
      *computeType = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
      return CUBLAS_STATUS_SUCCESS;
    default:
      return CUBLAS_STATUS_NOT_SUPPORTED;
  }
}
/* wrappers to accept old code with cudaDataType computeType when referenced from c++ code */
static inline cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m,
                                          int n,
                                          int k,
                                          const void* alpha, /* host or device pointer */
                                          const void* A,
                                          cudaDataType Atype,
                                          int lda,
                                          const void* B,
                                          cudaDataType Btype,
                                          int ldb,
                                          const void* beta, /* host or device pointer */
                                          void* C,
                                          cudaDataType Ctype,
                                          int ldc,
                                          cudaDataType computeType,
                                          cublasGemmAlgo_t algo) {
  cublasComputeType_t migratedComputeType = CUBLAS_COMPUTE_32F;
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  return cublasGemmEx(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      alpha,
                      A,
                      Atype,
                      lda,
                      B,
                      Btype,
                      ldb,
                      beta,
                      C,
                      Ctype,
                      ldc,
                      migratedComputeType,
                      algo);
}

static inline cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                                 cublasOperation_t transa,
                                                 cublasOperation_t transb,
                                                 int m,
                                                 int n,
                                                 int k,
                                                 const void* alpha, /* host or device pointer */
                                                 const void* const Aarray[],
                                                 cudaDataType Atype,
                                                 int lda,
                                                 const void* const Barray[],
                                                 cudaDataType Btype,
                                                 int ldb,
                                                 const void* beta, /* host or device pointer */
                                                 void* const Carray[],
                                                 cudaDataType Ctype,
                                                 int ldc,
                                                 int batchCount,
                                                 cudaDataType computeType,
                                                 cublasGemmAlgo_t algo) {
  cublasComputeType_t migratedComputeType;
  cublasStatus_t status;
  status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  return cublasGemmBatchedEx(handle,
                             transa,
                             transb,
                             m,
                             n,
                             k,
                             alpha,
                             Aarray,
                             Atype,
                             lda,
                             Barray,
                             Btype,
                             ldb,
                             beta,
                             Carray,
                             Ctype,
                             ldc,
                             batchCount,
                             migratedComputeType,
                             algo);
}

static inline cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                                        cublasOperation_t transa,
                                                        cublasOperation_t transb,
                                                        int m,
                                                        int n,
                                                        int k,
                                                        const void* alpha, /* host or device pointer */
                                                        const void* A,
                                                        cudaDataType Atype,
                                                        int lda,
                                                        long long int strideA, /* purposely signed */
                                                        const void* B,
                                                        cudaDataType Btype,
                                                        int ldb,
                                                        long long int strideB,
                                                        const void* beta, /* host or device pointer */
                                                        void* C,
                                                        cudaDataType Ctype,
                                                        int ldc,
                                                        long long int strideC,
                                                        int batchCount,
                                                        cudaDataType computeType,
                                                        cublasGemmAlgo_t algo) {
  cublasComputeType_t migratedComputeType;
  cublasStatus_t status;
  status = cublasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  return cublasGemmStridedBatchedEx(handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    Atype,
                                    lda,
                                    strideA,
                                    B,
                                    Btype,
                                    ldb,
                                    strideB,
                                    beta,
                                    C,
                                    Ctype,
                                    ldc,
                                    strideC,
                                    batchCount,
                                    migratedComputeType,
                                    algo);
}
#endif /* __cplusplus */

#endif /* !defined(CUBLAS_API_H_) */
