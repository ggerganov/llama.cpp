/*
 * Copyright 2014 NVIDIA Corporation.  All rights reserved.
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

#if !defined(CUSOLVERSP_H_)
#define CUSOLVERSP_H_

#include "cusparse.h"
#include "cublas_v2.h"
#include "cusolver_common.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct cusolverSpContext;
typedef struct cusolverSpContext *cusolverSpHandle_t;

struct csrqrInfo;
typedef struct csrqrInfo *csrqrInfo_t;

cusolverStatus_t CUSOLVERAPI cusolverSpCreate(cusolverSpHandle_t *handle);
cusolverStatus_t CUSOLVERAPI cusolverSpDestroy(cusolverSpHandle_t handle);
cusolverStatus_t CUSOLVERAPI cusolverSpSetStream (cusolverSpHandle_t handle, cudaStream_t streamId);
cusolverStatus_t CUSOLVERAPI cusolverSpGetStream(cusolverSpHandle_t handle, cudaStream_t *streamId);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrissymHost(
    cusolverSpHandle_t handle,
    int m,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,
    int *issym);

/* -------- GPU linear solver by LU factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [lu] stands for LU factorization
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol, 
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);


/* -------- GPU linear solver by QR factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [qr] stands for QR factorization
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);



/* -------- CPU linear solver by QR factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [qr] stands for QR factorization
 */ 
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);


/* -------- CPU linear solver by Cholesky factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [chol] stands for Cholesky factorization
 *
 * Only works for symmetric positive definite matrix.
 * The upper part of A is ignored.
 */ 
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);

/* -------- GPU linear solver by Cholesky factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [chol] stands for Cholesky factorization
 *
 * Only works for symmetric positive definite matrix.
 * The upper part of A is ignored.
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    // output
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    // output
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *b,
    float tol,
    int reorder,
    // output
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    // output
    cuDoubleComplex *x,
    int *singularity);



/* ----------- CPU least square solver by QR factorization
 *       solve min|b - A*x| 
 * [lsq] stands for least square
 * [v] stands for vector
 * [qr] stands for QR factorization
 */ 
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int *rankA,
    float *x,
    int *p,
    float *min_norm);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int *rankA,
    double *x,
    int *p,
    double *min_norm);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b,
    float tol,
    int *rankA,
    cuComplex *x,
    int *p,
    float *min_norm);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,
    double tol,
    int *rankA,
    cuDoubleComplex *x,
    int *p,
    double *min_norm);

/* --------- CPU eigenvalue solver by shift inverse
 *      solve A*x = lambda * x 
 *   where lambda is the eigenvalue nearest mu0.
 * [eig] stands for eigenvalue solver
 * [si] stands for shift-inverse
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu0,
    const float *x0,
    int maxite,
    float tol,
    float *mu,
    float *x);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu0,
    const double *x0,
    int maxite,
    double tol,
    double *mu,
    double *x);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu0,
    const cuComplex *x0,
    int maxite,
    float tol,
    cuComplex *mu,
    cuComplex *x);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu0,
    const cuDoubleComplex *x0,
    int maxite,
    double tol,
    cuDoubleComplex *mu,
    cuDoubleComplex *x);


/* --------- GPU eigenvalue solver by shift inverse
 *      solve A*x = lambda * x 
 *   where lambda is the eigenvalue nearest mu0.
 * [eig] stands for eigenvalue solver
 * [si] stands for shift-inverse
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu0,
    const float *x0,
    int maxite,
    float eps,
    float *mu,
    float *x);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu0,
    const double *x0,
    int maxite,
    double eps,
    double *mu, 
    double *x);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu0,
    const cuComplex *x0,
    int maxite,
    float eps,
    cuComplex *mu, 
    cuComplex *x);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu0,
    const cuDoubleComplex *x0,
    int maxite,
    double eps,
    cuDoubleComplex *mu, 
    cuDoubleComplex *x);


// ----------- enclosed eigenvalues

cusolverStatus_t CUSOLVERAPI cusolverSpScsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex left_bottom_corner,
    cuComplex right_upper_corner,
    int *num_eigs);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex left_bottom_corner,
    cuDoubleComplex right_upper_corner,
    int *num_eigs);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex left_bottom_corner,
    cuComplex right_upper_corner,
    int *num_eigs);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex left_bottom_corner,
    cuDoubleComplex right_upper_corner,
    int *num_eigs);



/* --------- CPU symrcm
 *   Symmetric reverse Cuthill McKee permutation         
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymrcmHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU symmdq 
 *   Symmetric minimum degree algorithm by quotient graph
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymmdqHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU symmdq 
 *   Symmetric Approximate minimum degree algorithm by quotient graph
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymamdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU metis 
 *   symmetric reordering 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrmetisndHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const int64_t *options,
    int *p);


/* --------- CPU zfd
 *  Zero free diagonal reordering
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);


/* --------- CPU permuation
 *   P*A*Q^T        
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrperm_bufferSizeHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const int *p,
    const int *q,
    size_t *bufferSizeInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrpermHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    int *csrRowPtrA,
    int *csrColIndA,
    const int *p,
    const int *q,
    int *map,
    void *pBuffer);



/*
 *  Low-level API: Batched QR
 *
 */

cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrqrInfo(
    csrqrInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrqrInfo(
    csrqrInfo_t info);


cusolverStatus_t CUSOLVERAPI cusolverSpXcsrqrAnalysisBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,   
    float *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,   
    double *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b, 
    cuComplex *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,  
    cuDoubleComplex *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);




#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // define CUSOLVERSP_H_



