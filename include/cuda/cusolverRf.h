/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
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

#if !defined(CUSOLVERRF_H_)
#define CUSOLVERRF_H_

#include "driver_types.h"
#include "cuComplex.h"   
#include "cusolver_common.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* CUSOLVERRF mode */
typedef enum { 
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0, //default   
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1        
} cusolverRfResetValuesFastMode_t;

/* CUSOLVERRF matrix format */
typedef enum { 
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0, //default   
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1        
} cusolverRfMatrixFormat_t;

/* CUSOLVERRF unit diagonal */
typedef enum { 
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0, //default   
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1, 
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,        
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3        
} cusolverRfUnitDiagonal_t;

/* CUSOLVERRF factorization algorithm */
typedef enum {
    CUSOLVERRF_FACTORIZATION_ALG0 = 0, // default
    CUSOLVERRF_FACTORIZATION_ALG1 = 1,
    CUSOLVERRF_FACTORIZATION_ALG2 = 2,
} cusolverRfFactorization_t;

/* CUSOLVERRF triangular solve algorithm */
typedef enum {
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1, // default
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
} cusolverRfTriangularSolve_t;

/* CUSOLVERRF numeric boost report */
typedef enum {
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0, //default
    CUSOLVERRF_NUMERIC_BOOST_USED = 1
} cusolverRfNumericBoostReport_t;

/* Opaque structure holding CUSOLVERRF library common */
struct cusolverRfCommon;
typedef struct cusolverRfCommon *cusolverRfHandle_t;

/* CUSOLVERRF create (allocate memory) and destroy (free memory) in the handle */
cusolverStatus_t CUSOLVERAPI cusolverRfCreate(cusolverRfHandle_t *handle);
cusolverStatus_t CUSOLVERAPI cusolverRfDestroy(cusolverRfHandle_t handle);

/* CUSOLVERRF set and get input format */
cusolverStatus_t CUSOLVERAPI cusolverRfGetMatrixFormat(cusolverRfHandle_t handle, 
                                                       cusolverRfMatrixFormat_t *format, 
                                                       cusolverRfUnitDiagonal_t *diag);

cusolverStatus_t CUSOLVERAPI cusolverRfSetMatrixFormat(cusolverRfHandle_t handle, 
                                                       cusolverRfMatrixFormat_t format, 
                                                       cusolverRfUnitDiagonal_t diag);
    
/* CUSOLVERRF set and get numeric properties */
cusolverStatus_t CUSOLVERAPI cusolverRfSetNumericProperties(cusolverRfHandle_t handle, 
                                                            double zero,
                                                            double boost);
											 
cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericProperties(cusolverRfHandle_t handle, 
                                                            double* zero,
                                                            double* boost);
											 
cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericBoostReport(cusolverRfHandle_t handle, 
                                                             cusolverRfNumericBoostReport_t *report);

/* CUSOLVERRF choose the triangular solve algorithm */
cusolverStatus_t CUSOLVERAPI cusolverRfSetAlgs(cusolverRfHandle_t handle,
                                               cusolverRfFactorization_t factAlg,
                                               cusolverRfTriangularSolve_t solveAlg);

cusolverStatus_t CUSOLVERAPI cusolverRfGetAlgs(cusolverRfHandle_t handle, 
                                               cusolverRfFactorization_t* factAlg,
                                               cusolverRfTriangularSolve_t* solveAlg);

/* CUSOLVERRF set and get fast mode */
cusolverStatus_t CUSOLVERAPI cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle, 
                                                              cusolverRfResetValuesFastMode_t *fastMode);

cusolverStatus_t CUSOLVERAPI cusolverRfSetResetValuesFastMode(cusolverRfHandle_t handle, 
                                                              cusolverRfResetValuesFastMode_t fastMode);

/*** Non-Batched Routines ***/
/* CUSOLVERRF setup of internal structures from host or device memory */
cusolverStatus_t CUSOLVERAPI cusolverRfSetupHost(/* Input (in the host memory) */
                                                 int n,
                                                 int nnzA,
                                                 int* h_csrRowPtrA,
                                                 int* h_csrColIndA,
                                                 double* h_csrValA,
                                                 int nnzL,
                                                 int* h_csrRowPtrL,
                                                 int* h_csrColIndL,
                                                 double* h_csrValL,
                                                 int nnzU,
                                                 int* h_csrRowPtrU,
                                                 int* h_csrColIndU,
                                                 double* h_csrValU,
                                                 int* h_P,
                                                 int* h_Q,
                                                 /* Output */
                                                 cusolverRfHandle_t handle);
    
cusolverStatus_t CUSOLVERAPI cusolverRfSetupDevice(/* Input (in the device memory) */
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA,
                                                   int* csrColIndA,
                                                   double* csrValA,
                                                   int nnzL,
                                                   int* csrRowPtrL,
                                                   int* csrColIndL,
                                                   double* csrValL,
                                                   int nnzU,
                                                   int* csrRowPtrU,
                                                   int* csrColIndU,
                                                   double* csrValU,
                                                   int* P,
                                                   int* Q,
                                                   /* Output */
                                                   cusolverRfHandle_t handle);

/* CUSOLVERRF update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
cusolverStatus_t CUSOLVERAPI cusolverRfResetValues(/* Input (in the device memory) */
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA, 
                                                   int* csrColIndA, 
                                                   double* csrValA,
                                                   int* P,
                                                   int* Q,
                                                   /* Output */
                                                   cusolverRfHandle_t handle);

/* CUSOLVERRF analysis (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfAnalyze(cusolverRfHandle_t handle);

/* CUSOLVERRF re-factorization (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfRefactor(cusolverRfHandle_t handle);

/* CUSOLVERRF extraction: Get L & U packed into a single matrix M */
cusolverStatus_t CUSOLVERAPI cusolverRfAccessBundledFactorsDevice(/* Input */
                                                                  cusolverRfHandle_t handle,
                                                                  /* Output (in the host memory) */
                                                                  int* nnzM, 
                                                                  /* Output (in the device memory) */
                                                                  int** Mp, 
                                                                  int** Mi, 
                                                                  double** Mx);

cusolverStatus_t CUSOLVERAPI cusolverRfExtractBundledFactorsHost(/* Input */
                                                                 cusolverRfHandle_t handle, 
                                                                 /* Output (in the host memory) */
                                                                 int* h_nnzM,
                                                                 int** h_Mp, 
                                                                 int** h_Mi, 
                                                                 double** h_Mx);

/* CUSOLVERRF extraction: Get L & U individually */
cusolverStatus_t CUSOLVERAPI cusolverRfExtractSplitFactorsHost(/* Input */
                                                               cusolverRfHandle_t handle, 
                                                               /* Output (in the host memory) */
                                                               int* h_nnzL, 
                                                               int** h_csrRowPtrL, 
                                                               int** h_csrColIndL, 
                                                               double** h_csrValL, 
                                                               int* h_nnzU, 
                                                               int** h_csrRowPtrU, 
                                                               int** h_csrColIndU, 
                                                               double** h_csrValU);

/* CUSOLVERRF (forward and backward triangular) solves */
cusolverStatus_t CUSOLVERAPI cusolverRfSolve(/* Input (in the device memory) */
                                             cusolverRfHandle_t handle,
                                             int *P,
                                             int *Q,
                                             int nrhs,     //only nrhs=1 is supported
                                             double *Temp, //of size ldt*nrhs (ldt>=n)
                                             int ldt,      
                                             /* Input/Output (in the device memory) */
                                             double *XF,
                                             /* Input */
                                             int ldxf);

/*** Batched Routines ***/
/* CUSOLVERRF-batch setup of internal structures from host */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchSetupHost(/* Input (in the host memory)*/
                                                      int batchSize,
                                                      int n,
                                                      int nnzA,
                                                      int* h_csrRowPtrA,
                                                      int* h_csrColIndA,
                                                      double* h_csrValA_array[],
                                                      int nnzL,
                                                      int* h_csrRowPtrL,
                                                      int* h_csrColIndL,
                                                      double *h_csrValL,
                                                      int nnzU,
                                                      int* h_csrRowPtrU,
                                                      int* h_csrColIndU,
                                                      double *h_csrValU,
                                                      int* h_P,
                                                      int* h_Q,
                                                      /* Output (in the device memory) */
                                                      cusolverRfHandle_t handle);

/* CUSOLVERRF-batch update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchResetValues(/* Input (in the device memory) */
                                                        int batchSize,
                                                        int n,
                                                        int nnzA,
                                                        int* csrRowPtrA,
                                                        int* csrColIndA,
                                                        double* csrValA_array[],
                                                        int* P,
                                                        int* Q,
                                                        /* Output */
                                                        cusolverRfHandle_t handle);
 
/* CUSOLVERRF-batch analysis (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchAnalyze(cusolverRfHandle_t handle);

/* CUSOLVERRF-batch re-factorization (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchRefactor(cusolverRfHandle_t handle);

/* CUSOLVERRF-batch (forward and backward triangular) solves */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchSolve(/* Input (in the device memory) */
                                                  cusolverRfHandle_t handle,
                                                  int *P,
                                                  int *Q,
                                                  int nrhs,     //only nrhs=1 is supported
                                                  double *Temp, //of size 2*batchSize*(n*nrhs)
                                                  int ldt,      //only ldt=n is supported
                                                  /* Input/Output (in the device memory) */
                                                  double *XF_array[],
                                                  /* Input */
                                                  int ldxf);

/* CUSOLVERRF-batch obtain the position of zero pivot */    
cusolverStatus_t CUSOLVERAPI cusolverRfBatchZeroPivot(/* Input */
                                                      cusolverRfHandle_t handle,
                                                      /* Output (in the host memory) */
                                                      int *position);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* CUSOLVERRF_H_ */
