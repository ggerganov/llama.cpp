/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
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

#if !defined(CUSOLVERSP_LOWLEVEL_PREVIEW_H_)
#define CUSOLVERSP_LOWLEVEL_PREVIEW_H_

#include "cusolverSp.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


struct csrluInfoHost;
typedef struct csrluInfoHost *csrluInfoHost_t;


struct csrqrInfoHost;
typedef struct csrqrInfoHost *csrqrInfoHost_t;


struct csrcholInfoHost;
typedef struct csrcholInfoHost *csrcholInfoHost_t;


struct csrcholInfo;
typedef struct csrcholInfo *csrcholInfo_t;



/*
 * Low level API for CPU LU
 * 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrluInfoHost(
    csrluInfoHost_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrluInfoHost(
    csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluAnalysisHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    float pivot_threshold,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    double pivot_threshold,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    float pivot_threshold,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    double pivot_threshold,
    void *pBuffer);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    double tol,
    int *position);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuComplex *b,
    cuComplex *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluNnzHost(
    cusolverSpHandle_t handle,
    int *nnzLRef,
    int *nnzURef,
    csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    float *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    float *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    double *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    double *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    cuComplex *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    cuComplex *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    cuDoubleComplex *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    cuDoubleComplex *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);


/*
 * Low level API for CPU QR
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrqrInfoHost(
    csrqrInfoHost_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrqrInfoHost(
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrqrAnalysisHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    float *b,
    float *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    double *b,
    double *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuComplex *b,
    cuComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    double tol,
    int *position);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    float *b,
    float *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    double *b,
    double *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuComplex *b,
    cuComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);


/*
 * Low level API for GPU QR
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrqrAnalysis(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu,
    csrqrInfo_t info);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    float *b,
    float *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    double *b,
    double *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuComplex *b,
    cuComplex *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfo_t info,
    void *pBuffer);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    double tol,
    int *position);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    float *b,
    float *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    double *b,
    double *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuComplex *b,
    cuComplex *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfo_t info,
    void *pBuffer);


/*
 * Low level API for CPU Cholesky
 * 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrcholInfoHost(
    csrcholInfoHost_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrcholInfoHost(
    csrcholInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholAnalysisHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuComplex *b,
    cuComplex *x,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrcholInfoHost_t info,
    void *pBuffer);

/*
 * Low level API for GPU Cholesky
 * 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrcholInfo(
    csrcholInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrcholInfo(
    csrcholInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholAnalysis(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const cuComplex *b,
    cuComplex *x,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrcholInfo_t info,
    void *pBuffer);

/*
 * "diag" is a device array of size N.
 * cusolverSp<t>csrcholDiag returns diag(L) to "diag" where A(P,P) = L*L**T
 * "diag" can estimate det(A) because det(A(P,P)) = det(A) = det(L)^2 if A = L*L**T.
 * 
 * cusolverSp<t>csrcholDiag must be called after cusolverSp<t>csrcholFactor.
 * otherwise "diag" is wrong.
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float *diag);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double *diag);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float *diag);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double *diag);





#if defined(__cplusplus)
}
#endif /* __cplusplus */



#endif // CUSOLVERSP_LOWLEVEL_PREVIEW_H_


