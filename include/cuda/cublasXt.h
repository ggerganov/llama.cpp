/*
 * Copyright 1993-2019 NVIDIA Corporation. All rights reserved.
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

/*   cublasXt : Host API, Out of Core and Multi-GPU BLAS Library

*/

#if !defined(CUBLAS_XT_H_)
#define CUBLAS_XT_H_

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#include "cublas_v2.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct cublasXtContext;
typedef struct cublasXtContext* cublasXtHandle_t;

cublasStatus_t CUBLASWINAPI cublasXtCreate(cublasXtHandle_t* handle);
cublasStatus_t CUBLASWINAPI cublasXtDestroy(cublasXtHandle_t handle);
cublasStatus_t CUBLASWINAPI cublasXtGetNumBoards(int nbDevices, int deviceId[], int* nbBoards);
cublasStatus_t CUBLASWINAPI cublasXtMaxBoards(int* nbGpuBoards);
/* This routine selects the Gpus that the user want to use for CUBLAS-XT */
cublasStatus_t CUBLASWINAPI cublasXtDeviceSelect(cublasXtHandle_t handle, int nbDevices, int deviceId[]);

/* This routine allows to change the dimension of the tiles ( blockDim x blockDim ) */
cublasStatus_t CUBLASWINAPI cublasXtSetBlockDim(cublasXtHandle_t handle, int blockDim);
cublasStatus_t CUBLASWINAPI cublasXtGetBlockDim(cublasXtHandle_t handle, int* blockDim);

typedef enum { CUBLASXT_PINNING_DISABLED = 0, CUBLASXT_PINNING_ENABLED = 1 } cublasXtPinnedMemMode_t;
/* This routine allows to CUBLAS-XT to pin the Host memory if it find out that some of the matrix passed
   are not pinned : Pinning/Unpinning the Host memory is still a costly operation
   It is better if the user controls the memory on its own (by pinning/unpinning oly when necessary)
*/
cublasStatus_t CUBLASWINAPI cublasXtGetPinningMemMode(cublasXtHandle_t handle, cublasXtPinnedMemMode_t* mode);
cublasStatus_t CUBLASWINAPI cublasXtSetPinningMemMode(cublasXtHandle_t handle, cublasXtPinnedMemMode_t mode);

/* This routines is to provide a CPU Blas routines, used for too small sizes or hybrid computation */
typedef enum {
  CUBLASXT_FLOAT = 0,
  CUBLASXT_DOUBLE = 1,
  CUBLASXT_COMPLEX = 2,
  CUBLASXT_DOUBLECOMPLEX = 3,
} cublasXtOpType_t;

typedef enum {
  CUBLASXT_GEMM = 0,
  CUBLASXT_SYRK = 1,
  CUBLASXT_HERK = 2,
  CUBLASXT_SYMM = 3,
  CUBLASXT_HEMM = 4,
  CUBLASXT_TRSM = 5,
  CUBLASXT_SYR2K = 6,
  CUBLASXT_HER2K = 7,

  CUBLASXT_SPMM = 8,
  CUBLASXT_SYRKX = 9,
  CUBLASXT_HERKX = 10,
  CUBLASXT_TRMM = 11,
  CUBLASXT_ROUTINE_MAX = 12,
} cublasXtBlasOp_t;

/* Currently only 32-bit integer BLAS routines are supported */
cublasStatus_t CUBLASWINAPI cublasXtSetCpuRoutine(cublasXtHandle_t handle,
                                                  cublasXtBlasOp_t blasOp,
                                                  cublasXtOpType_t type,
                                                  void* blasFunctor);

/* Specified the percentage of work that should done by the CPU, default is 0 (no work) */
cublasStatus_t CUBLASWINAPI cublasXtSetCpuRatio(cublasXtHandle_t handle,
                                                cublasXtBlasOp_t blasOp,
                                                cublasXtOpType_t type,
                                                float ratio);

/* GEMM */
cublasStatus_t CUBLASWINAPI cublasXtSgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZgemm(cublasXtHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          size_t m,
                                          size_t n,
                                          size_t k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* ------------------------------------------------------- */
/* SYRK */
cublasStatus_t CUBLASWINAPI cublasXtSsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZsyrk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* -------------------------------------------------------------------- */
/* HERK */
cublasStatus_t CUBLASWINAPI cublasXtCherk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const float* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const float* beta,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZherk(cublasXtHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          size_t n,
                                          size_t k,
                                          const double* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const double* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* -------------------------------------------------------------------- */
/* SYR2K */
cublasStatus_t CUBLASWINAPI cublasXtSsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const float* alpha,
                                           const float* A,
                                           size_t lda,
                                           const float* B,
                                           size_t ldb,
                                           const float* beta,
                                           float* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const double* alpha,
                                           const double* A,
                                           size_t lda,
                                           const double* B,
                                           size_t ldb,
                                           const double* beta,
                                           double* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const cuComplex* beta,
                                           cuComplex* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZsyr2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const cuDoubleComplex* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);
/* -------------------------------------------------------------------- */
/* HERKX : variant extension of HERK */
cublasStatus_t CUBLASWINAPI cublasXtCherkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const float* beta,
                                           cuComplex* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZherkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const double* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);

/* -------------------------------------------------------------------- */
/* TRSM */
cublasStatus_t CUBLASWINAPI cublasXtStrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          float* B,
                                          size_t ldb);

cublasStatus_t CUBLASWINAPI cublasXtDtrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          double* B,
                                          size_t ldb);

cublasStatus_t CUBLASWINAPI cublasXtCtrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          cuComplex* B,
                                          size_t ldb);

cublasStatus_t CUBLASWINAPI cublasXtZtrsm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          cuDoubleComplex* B,
                                          size_t ldb);
/* -------------------------------------------------------------------- */
/* SYMM : Symmetric Multiply Matrix*/
cublasStatus_t CUBLASWINAPI cublasXtSsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZsymm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);
/* -------------------------------------------------------------------- */
/* HEMM : Hermitian Matrix Multiply */
cublasStatus_t CUBLASWINAPI cublasXtChemm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZhemm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);

/* -------------------------------------------------------------------- */
/* SYRKX : variant extension of SYRK  */
cublasStatus_t CUBLASWINAPI cublasXtSsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const float* alpha,
                                           const float* A,
                                           size_t lda,
                                           const float* B,
                                           size_t ldb,
                                           const float* beta,
                                           float* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const double* alpha,
                                           const double* A,
                                           size_t lda,
                                           const double* B,
                                           size_t ldb,
                                           const double* beta,
                                           double* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const cuComplex* beta,
                                           cuComplex* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZsyrkx(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const cuDoubleComplex* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);
/* -------------------------------------------------------------------- */
/* HER2K : variant extension of HERK  */
cublasStatus_t CUBLASWINAPI cublasXtCher2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuComplex* alpha,
                                           const cuComplex* A,
                                           size_t lda,
                                           const cuComplex* B,
                                           size_t ldb,
                                           const float* beta,
                                           cuComplex* C,
                                           size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZher2k(cublasXtHandle_t handle,
                                           cublasFillMode_t uplo,
                                           cublasOperation_t trans,
                                           size_t n,
                                           size_t k,
                                           const cuDoubleComplex* alpha,
                                           const cuDoubleComplex* A,
                                           size_t lda,
                                           const cuDoubleComplex* B,
                                           size_t ldb,
                                           const double* beta,
                                           cuDoubleComplex* C,
                                           size_t ldc);

/* -------------------------------------------------------------------- */
/* SPMM : Symmetric Packed Multiply Matrix*/
cublasStatus_t CUBLASWINAPI cublasXtSspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* AP,
                                          const float* B,
                                          size_t ldb,
                                          const float* beta,
                                          float* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* AP,
                                          const double* B,
                                          size_t ldb,
                                          const double* beta,
                                          double* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* AP,
                                          const cuComplex* B,
                                          size_t ldb,
                                          const cuComplex* beta,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZspmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* AP,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C,
                                          size_t ldc);

/* -------------------------------------------------------------------- */
/* TRMM */
cublasStatus_t CUBLASWINAPI cublasXtStrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const float* alpha,
                                          const float* A,
                                          size_t lda,
                                          const float* B,
                                          size_t ldb,
                                          float* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtDtrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const double* alpha,
                                          const double* A,
                                          size_t lda,
                                          const double* B,
                                          size_t ldb,
                                          double* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtCtrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuComplex* alpha,
                                          const cuComplex* A,
                                          size_t lda,
                                          const cuComplex* B,
                                          size_t ldb,
                                          cuComplex* C,
                                          size_t ldc);

cublasStatus_t CUBLASWINAPI cublasXtZtrmm(cublasXtHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          size_t m,
                                          size_t n,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A,
                                          size_t lda,
                                          const cuDoubleComplex* B,
                                          size_t ldb,
                                          cuDoubleComplex* C,
                                          size_t ldc);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(CUBLAS_XT_H_) */
