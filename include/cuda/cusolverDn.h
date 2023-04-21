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
 
 /*   cuSolverDN : Dense Linear Algebra Library

 */
 
#if !defined(CUSOLVERDN_H_)
#define CUSOLVERDN_H_

struct cusolverDnContext;
typedef struct cusolverDnContext *cusolverDnHandle_t;

struct syevjInfo;
typedef struct syevjInfo *syevjInfo_t;

struct gesvdjInfo;
typedef struct gesvdjInfo *gesvdjInfo_t;


//------------------------------------------------------
// opaque cusolverDnIRS structure for IRS solver
struct cusolverDnIRSParams;
typedef struct cusolverDnIRSParams* cusolverDnIRSParams_t;

struct cusolverDnIRSInfos;
typedef struct cusolverDnIRSInfos* cusolverDnIRSInfos_t;
//------------------------------------------------------

struct cusolverDnParams;
typedef struct cusolverDnParams *cusolverDnParams_t;

typedef enum {
   CUSOLVERDN_GETRF = 0,
   CUSOLVERDN_POTRF = 1
} cusolverDnFunction_t ;



#include "cuComplex.h"   /* import complex data type */
#include "cublas_v2.h"
#include "cusolver_common.h"



/*******************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t CUSOLVERAPI cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);

//============================================================
// IRS headers 
//============================================================

// =============================================================================
// IRS helper function API
// =============================================================================
cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsCreate(
            cusolverDnIRSParams_t* params_ptr );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsDestroy(
            cusolverDnIRSParams_t params );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetRefinementSolver(
            cusolverDnIRSParams_t params,
            cusolverIRSRefinement_t refinement_solver );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetSolverMainPrecision(
            cusolverDnIRSParams_t params,
            cusolverPrecType_t solver_main_precision ); 

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetSolverLowestPrecision(
            cusolverDnIRSParams_t params,
            cusolverPrecType_t solver_lowest_precision );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetSolverPrecisions(
            cusolverDnIRSParams_t params,
            cusolverPrecType_t solver_main_precision,
            cusolverPrecType_t solver_lowest_precision );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetTol(
            cusolverDnIRSParams_t params,
            double val );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetTolInner(
            cusolverDnIRSParams_t params,
            double val );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetMaxIters(
            cusolverDnIRSParams_t params,
            cusolver_int_t maxiters );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetMaxItersInner(
            cusolverDnIRSParams_t params,
            cusolver_int_t maxiters_inner );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsGetMaxIters(
            cusolverDnIRSParams_t params,
            cusolver_int_t *maxiters );

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsEnableFallback(
    cusolverDnIRSParams_t params );

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsDisableFallback(
    cusolverDnIRSParams_t params );


// =============================================================================
// cusolverDnIRSInfos prototypes
// =============================================================================
cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosDestroy(
        cusolverDnIRSInfos_t infos );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosCreate(
        cusolverDnIRSInfos_t* infos_ptr );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosGetNiters(
            cusolverDnIRSInfos_t infos,
            cusolver_int_t *niters );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSInfosGetOuterNiters(
            cusolverDnIRSInfos_t infos,
            cusolver_int_t *outer_niters );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosRequestResidual(
        cusolverDnIRSInfos_t infos );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosGetResidualHistory(
            cusolverDnIRSInfos_t infos,
            void **residual_history );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSInfosGetMaxIters(
            cusolverDnIRSInfos_t infos,
            cusolver_int_t *maxiters );

//============================================================
//  IRS functions API
//============================================================

/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gesv 
 * users API Prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

/*******************************************************************************/


/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gesv_bufferSize 
 * users API Prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);
/*******************************************************************************/


/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels 
 * users API Prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);
/*******************************************************************************/

/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels_bufferSize 
 * API prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);
/*******************************************************************************/



/*******************************************************************************//*
 * expert users API for IRS Prototypes
 * */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t gesv_irs_params,
        cusolverDnIRSInfos_t  gesv_irs_infos,
        cusolver_int_t n, cusolver_int_t nrhs,
        void *dA, cusolver_int_t ldda,
        void *dB, cusolver_int_t lddb,
        void *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *niters,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t params,
        cusolver_int_t n, cusolver_int_t nrhs,
        size_t *lwork_bytes);


cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t gels_irs_params,
        cusolverDnIRSInfos_t  gels_irs_infos,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        void *dA, cusolver_int_t ldda,
        void *dB, cusolver_int_t lddb,
        void *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *niters,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t params,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs, 
        size_t *lwork_bytes);
/*******************************************************************************/


/* Cholesky factorization and its solver */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int *Lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda,  
    float *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );



cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


cusolverStatus_t CUSOLVERAPI cusolverDnSpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const float *A,
    int lda,
    float *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const double *A,
    int lda,
    double *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    int *devInfo);

/* batched Cholesky factorization and its solver */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    float *A[],
    int lda,
    float *B[],
    int ldb,
    int *d_info,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    double *A[],
    int lda,
    double *B[],
    int ldb,
    int *d_info,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    cuComplex *A[],
    int lda,
    cuComplex *B[],
    int ldb,
    int *d_info,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    cuDoubleComplex *A[],
    int lda,
    cuDoubleComplex *B[],
    int ldb,
    int *d_info,
    int batchSize);

/* s.p.d. matrix inversion (POTRI) and auxiliary routines (TRTRI and LAUUM)  */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnXtrtri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXtrtri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *devInfo);

/* lauum, auxiliar routine for s.p.d matrix inversion */
cusolverStatus_t CUSOLVERAPI cusolverDnSlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnClauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnClauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);



/* LU Factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *Lwork );


cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

/* Row pivoting */
cusolverStatus_t CUSOLVERAPI cusolverDnSlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    float *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUSOLVERAPI cusolverDnDlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    double *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUSOLVERAPI cusolverDnClaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    cuComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUSOLVERAPI cusolverDnZlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

/* LU solve */
cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const float *A, 
    int lda, 
    const int *devIpiv, 
    float *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const double *A, 
    int lda, 
    const int *devIpiv, 
    double *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const cuComplex *A, 
    int lda, 
    const int *devIpiv, 
    cuComplex *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const cuDoubleComplex *A, 
    int lda, 
    const int *devIpiv, 
    cuDoubleComplex *B, 
    int ldb, 
    int *devInfo );


/* QR factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda, 
    float *TAU,  
    float *Workspace,  
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *TAU, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *TAU, 
    cuComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *TAU, 
    cuDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


/* generate unitary matrix Q from QR factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCungqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZungqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCungqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZungqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* compute Q**T*b in solve min||A*x = b|| */
cusolverStatus_t CUSOLVERAPI cusolverDnSormqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDormqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    const cuComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    const cuDoubleComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);


/* L*D*L**T,U*D*U**T factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    float *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    double *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    cuComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *ipiv,
    float *work,
    int lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *ipiv,
    double *work,
    int lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *ipiv,
    cuComplex *work,
    int lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *ipiv,
    cuDoubleComplex *work,
    int lwork,
    int *info );

/* Symmetric indefinite solve (SYTRS) */
cusolverStatus_t CUSOLVERAPI cusolverDnXsytrs_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int64_t n,
        int64_t nrhs,
        cudaDataType dataTypeA,
        const void *A,
        int64_t lda,
        const int64_t *ipiv,
        cudaDataType dataTypeB,
        void *B,
        int64_t ldb,
        size_t *workspaceInBytesOnDevice,
        size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXsytrs(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int64_t n,
        int64_t nrhs,
        cudaDataType dataTypeA,
        const void *A,
        int64_t lda,
        const int64_t *ipiv,
        cudaDataType dataTypeB,
        void *B,
        int64_t ldb,
        void *bufferOnDevice,
        size_t workspaceInBytesOnDevice,
        void *bufferOnHost,
        size_t workspaceInBytesOnHost,
        int *info);

/* Symmetric indefinite inversion (sytri) */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        float *work,
        int lwork,
        int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        double *work,
        int lwork,
        int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex *A,
        int lda,
        const int *ipiv,
        cuComplex *work,
        int lwork,
        int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        cuDoubleComplex *work,
        int lwork,
        int *info);


/* bidiagonal factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda,
    float *D, 
    float *E, 
    float *TAUQ,  
    float *TAUP, 
    float *Work,
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *D, 
    double *E, 
    double *TAUQ, 
    double *TAUP, 
    double *Work,
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    float *D, 
    float *E, 
    cuComplex *TAUQ, 
    cuComplex *TAUP,
    cuComplex *Work, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A,
    int lda, 
    double *D, 
    double *E, 
    cuDoubleComplex *TAUQ,
    cuDoubleComplex *TAUP, 
    cuDoubleComplex *Work, 
    int Lwork, 
    int *devInfo );

/* generates one of the unitary matrices Q or P**T determined by GEBRD*/
cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCungbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZungbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCungbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZungbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);


/* tridiagonal factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *d,
    const float *e,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *d,
    const double *e,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChetrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *d,
    const float *e,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *d,
    const double *e,
    const cuDoubleComplex *tau,
    int *lwork);


cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *d,
    float *e,
    float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *d,
    double *e,
    double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnChetrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *d,
    float *e,
    cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *d,
    double *e,
    cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* generate unitary Q comes from sytrd */
cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCungtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZungtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCungtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZungtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* compute op(Q)*C or C*op(Q) where Q comes from sytrd */
cusolverStatus_t CUSOLVERAPI cusolverDnSormtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDormtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    const cuComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    const cuDoubleComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSormtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    float *A,
    int lda,
    float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDormtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    double *A,
    int lda,
    double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* singular value decomposition, A = U * Sigma * V^H */
cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U, 
    int ldu, 
    float *VT, 
    int ldvt, 
    float *work, 
    int lwork, 
    float *rwork, 
    int  *info );

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *S, 
    double *U, 
    int ldu, 
    double *VT, 
    int ldvt, 
    double *work,
    int lwork, 
    double *rwork, 
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    cuComplex *A,
    int lda, 
    float *S, 
    cuComplex *U, 
    int ldu, 
    cuComplex *VT, 
    int ldvt,
    cuComplex *work, 
    int lwork, 
    float *rwork, 
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    double *S, 
    cuDoubleComplex *U, 
    int ldu, 
    cuDoubleComplex *VT, 
    int ldvt, 
    cuDoubleComplex *work, 
    int lwork, 
    double *rwork, 
    int *info );


/* standard symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);

/* standard selective symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);

/* selective generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    const double *A, 
    int lda,
    const double *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    const cuComplex *A, 
    int lda,
    const cuComplex *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,  
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    double *A, 
    int lda,
    double *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    cuDoubleComplex *A, 
    int lda,
    cuDoubleComplex *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);


/* generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    const double *A, 
    int lda,
    const double *B, 
    int ldb,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    const cuComplex *A, 
    int lda,
    const cuComplex *B, 
    int ldb,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,  
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B, 
    int ldb,
    const double *W,
    int *lwork);


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,  
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    double *A, 
    int lda,
    double *B, 
    int ldb,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B, 
    int ldb,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    cuDoubleComplex *A, 
    int lda,
    cuDoubleComplex *B, 
    int ldb,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);


cusolverStatus_t CUSOLVERAPI cusolverDnCreateSyevjInfo(
    syevjInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDestroySyevjInfo(
    syevjInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetTolerance(
    syevjInfo_t info,
    double tolerance);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetMaxSweeps(
    syevjInfo_t info,
    int max_sweeps);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetSortEig(
    syevjInfo_t info,
    int sort_eig);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetResidual(
    cusolverDnHandle_t handle,
    syevjInfo_t info,
    double *residual);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetSweeps(
    cusolverDnHandle_t handle,
    syevjInfo_t info,
    int *executed_sweeps);


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const double *A, 
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A, 
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A, 
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,   
    float *A,
    int lda,
    float *W, 
    float *work,
    int lwork,
    int *info, 
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnCheevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnZheevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W, 
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnCheevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnZheevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    const double *A, 
    int lda,
    const double *B,
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A, 
    int lda,
    const cuComplex *B, 
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A, 
    int lda,
    const cuDoubleComplex *B, 
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    double *A, 
    int lda,
    double *B,
    int ldb,
    double *W, 
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    cuComplex *A, 
    int lda,
    cuComplex *B, 
    int ldb,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,  
    int n,
    cuDoubleComplex *A, 
    int lda,
    cuDoubleComplex *B, 
    int ldb,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnCreateGesvdjInfo(
    gesvdjInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDestroyGesvdjInfo(
    gesvdjInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetTolerance(
    gesvdjInfo_t info,
    double tolerance);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetMaxSweeps(
    gesvdjInfo_t info,
    int max_sweeps);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetSortEig(
    gesvdjInfo_t info,
    int sort_svd);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetResidual(
    cusolverDnHandle_t handle,
    gesvdjInfo_t info,
    double *residual);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetSweeps(
    cusolverDnHandle_t handle,
    gesvdjInfo_t info,
    int *executed_sweeps);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,                
    int n,                
    const float *A,    
    int lda,           
    const float *S, 
    const float *U,   
    int ldu, 
    const float *V,
    int ldv,  
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const double *A, 
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const float *S,
    const cuComplex *U,
    int ldu,
    const cuComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int m, 
    int n, 
    const cuDoubleComplex *A,
    int lda,
    const double *S,
    const cuDoubleComplex *U,
    int ldu, 
    const cuDoubleComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U,
    int ldu,
    float *V,
    int ldv, 
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    double *A,
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv, 
    double *work,
    int lwork,
    int *info, 
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m, 
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *V,
    int ldv,
    cuComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda, 
    double *S, 
    cuDoubleComplex *U,
    int ldu,
    cuDoubleComplex *V,
    int ldv,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int econ,
    int m,
    int n, 
    const float *A,
    int lda,
    const float *S,
    const float *U,
    int ldu, 
    const float *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int econ,
    int m,
    int n,
    const double *A, 
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int econ,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const float *S,
    const cuComplex *U,
    int ldu,
    const cuComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *S,
    const cuDoubleComplex *U,
    int ldu,
    const cuDoubleComplex *V,
    int ldv, 
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    float *A, 
    int lda,
    float *S,
    float *U,
    int ldu,
    float *V,
    int ldv,
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv,
    double *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *V,
    int ldv,
    cuComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *S,
    cuDoubleComplex *U, 
    int ldu, 
    cuDoubleComplex *V,
    int ldv,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);


/* batched approximate SVD */

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const float *d_A, 
    int lda,
    long long int strideA, 
    const float *d_S, 
    long long int strideS, 
    const float *d_U, 
    int ldu,
    long long int strideU, 
    const float *d_V, 
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const double *d_A, 
    int lda,
    long long int strideA, 
    const double *d_S,   
    long long int strideS, 
    const double *d_U,  
    int ldu,
    long long int strideU, 
    const double *d_V,
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuComplex *d_A, 
    int lda,
    long long int strideA, 
    const float *d_S, 
    long long int strideS, 
    const cuComplex *d_U,
    int ldu,
    long long int strideU, 
    const cuComplex *d_V, 
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuDoubleComplex *d_A,
    int lda,
    long long int strideA,
    const double *d_S, 
    long long int strideS, 
    const cuDoubleComplex *d_U, 
    int ldu,
    long long int strideU,
    const cuDoubleComplex *d_V,
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank, 
    int m,   
    int n,  
    const float *d_A, 
    int lda, 
    long long int strideA,
    float *d_S, 
    long long int strideS, 
    float *d_U, 
    int ldu, 
    long long int strideU,
    float *d_V, 
    int ldv,    
    long long int strideV, 
    float *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize);


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank,
    int m, 
    int n, 
    const double *d_A,
    int lda,  
    long long int strideA, 
    double *d_S, 
    long long int strideS,
    double *d_U, 
    int ldu, 
    long long int strideU, 
    double *d_V, 
    int ldv, 
    long long int strideV,
    double *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF, 
    int batchSize);


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank,  
    int m, 
    int n, 
    const cuComplex *d_A, 
    int lda,
    long long int strideA,
    float *d_S,
    long long int strideS,
    cuComplex *d_U, 
    int ldu,   
    long long int strideU,  
    cuComplex *d_V, 
    int ldv, 
    long long int strideV,
    cuComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF, 
    int batchSize);


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank, 
    int m,   
    int n,  
    const cuDoubleComplex *d_A, 
    int lda,    
    long long int strideA,
    double *d_S,
    long long int strideS,
    cuDoubleComplex *d_U, 
    int ldu,   
    long long int strideU, 
    cuDoubleComplex *d_V,
    int ldv, 
    long long int strideV, 
    cuDoubleComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCreateParams(
    cusolverDnParams_t *params);

cusolverStatus_t CUSOLVERAPI cusolverDnDestroyParams(
    cusolverDnParams_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnSetAdvOptions (
    cusolverDnParams_t params,
    cusolverDnFunction_t function,
    cusolverAlgMode_t algo   );

/* 64-bit API for POTRF */
CUSOLVER_DEPRECATED(cusolverDnXpotrf_bufferSize)
cusolverStatus_t CUSOLVERAPI cusolverDnPotrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytes );

CUSOLVER_DEPRECATED(cusolverDnXpotrf)
cusolverStatus_t CUSOLVERAPI cusolverDnPotrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for POTRS */
CUSOLVER_DEPRECATED(cusolverDnXpotrs)
cusolverStatus_t CUSOLVERAPI cusolverDnPotrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info);


/* 64-bit API for GEQRF */
CUSOLVER_DEPRECATED(cusolverDnXgeqrf_bufferSize)
cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    const void *tau,
    cudaDataType computeType,
    size_t *workspaceInBytes );

CUSOLVER_DEPRECATED(cusolverDnXgeqrf)
cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    void *tau,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for GETRF */
CUSOLVER_DEPRECATED(cusolverDnXgetrf_bufferSize)
cusolverStatus_t CUSOLVERAPI cusolverDnGetrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytes );

CUSOLVER_DEPRECATED(cusolverDnXgetrf)
cusolverStatus_t CUSOLVERAPI cusolverDnGetrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    int64_t *ipiv,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for GETRS */
CUSOLVER_DEPRECATED(cusolverDnXgetrs)
cusolverStatus_t CUSOLVERAPI cusolverDnGetrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasOperation_t trans,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    const int64_t *ipiv,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info );

/* 64-bit API for SYEVD */
CUSOLVER_DEPRECATED(cusolverDnXsyevd_bufferSize)
cusolverStatus_t CUSOLVERAPI cusolverDnSyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytes);

CUSOLVER_DEPRECATED(cusolverDnXsyevd)
cusolverStatus_t CUSOLVERAPI cusolverDnSyevd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/* 64-bit API for SYEVDX */
CUSOLVER_DEPRECATED(cusolverDnXsyevdx_bufferSize)
cusolverStatus_t CUSOLVERAPI cusolverDnSyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    void *vl,
    void *vu,
    int64_t il,
    int64_t iu,
    int64_t *h_meig,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytes);


CUSOLVER_DEPRECATED(cusolverDnXsyevdx)
cusolverStatus_t CUSOLVERAPI cusolverDnSyevdx(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    void * vl,
    void * vu,
    int64_t il,
    int64_t iu,
    int64_t *meig64,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/* 64-bit API for GESVD */
CUSOLVER_DEPRECATED(cusolverDnXgesvd_bufferSize)
cusolverStatus_t CUSOLVERAPI cusolverDnGesvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    const void *S,
    cudaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    const void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    size_t *workspaceInBytes);

CUSOLVER_DEPRECATED(cusolverDnXgesvd)
cusolverStatus_t CUSOLVERAPI cusolverDnGesvd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    void *S,
    cudaDataType dataTypeU,
    void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/*
 * new 64-bit API
 */
/* 64-bit API for POTRF */
cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for POTRS */
cusolverStatus_t CUSOLVERAPI cusolverDnXpotrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info);

/* 64-bit API for GEQRF */
cusolverStatus_t CUSOLVERAPI cusolverDnXgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    const void *tau,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgeqrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    void *tau,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for GETRF */
cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    int64_t *ipiv,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for GETRS */
cusolverStatus_t CUSOLVERAPI cusolverDnXgetrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasOperation_t trans,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    const int64_t *ipiv,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info );

/* 64-bit API for SYEVD */
cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for SYEVDX */
cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    void *vl,
    void *vu,
    int64_t il,
    int64_t iu,
    int64_t *h_meig,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    void * vl,
    void * vu,
    int64_t il,
    int64_t iu,
    int64_t *meig64,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for GESVD */
cusolverStatus_t CUSOLVERAPI cusolverDnXgesvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    const void *S,
    cudaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    const void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    void *S,
    cudaDataType dataTypeU,
    void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for GESVDP */
cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdp_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    int econ,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    const void *S,
    cudaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    cudaDataType dataTypeV,
    const void *V,
    int64_t ldv,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdp(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz, 
    int econ,   
    int64_t m,   
    int64_t n,   
    cudaDataType dataTypeA,
    void *A,            
    int64_t lda,     
    cudaDataType dataTypeS,
    void *S,  
    cudaDataType dataTypeU,
    void *U,    
    int64_t ldu,   
    cudaDataType dataTypeV,
    void *V,  
    int64_t ldv, 
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *d_info,
    double *h_err_sigma);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdr_bufferSize (
		cusolverDnHandle_t handle,
		cusolverDnParams_t params,
		signed char        jobu,
		signed char        jobv,
		int64_t            m,
		int64_t            n,
		int64_t            k,
		int64_t            p,
		int64_t            niters,
		cudaDataType       dataTypeA,
		const void         *A,
		int64_t            lda,
		cudaDataType       dataTypeSrand,
		const void         *Srand,
		cudaDataType       dataTypeUrand,
		const void         *Urand,
		int64_t            ldUrand,
		cudaDataType       dataTypeVrand,
		const void         *Vrand,
		int64_t            ldVrand,
		cudaDataType       computeType,
		size_t             *workspaceInBytesOnDevice,
		size_t             *workspaceInBytesOnHost
		);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdr(
		cusolverDnHandle_t handle,
		cusolverDnParams_t params,
		signed char        jobu,
		signed char        jobv,
		int64_t            m,
		int64_t            n,
		int64_t            k,
		int64_t            p,
		int64_t            niters,
		cudaDataType       dataTypeA,
		void               *A,
		int64_t            lda,
		cudaDataType       dataTypeSrand,
		void               *Srand,
		cudaDataType       dataTypeUrand,
		void               *Urand,
		int64_t            ldUrand,
		cudaDataType       dataTypeVrand,
		void               *Vrand,
		int64_t            ldVrand,
		cudaDataType       computeType,
		void               *bufferOnDevice,
		size_t             workspaceInBytesOnDevice,
		void               *bufferOnHost,
		size_t             workspaceInBytesOnHost,
		int                *d_info
		);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* !defined(CUDENSE_H_) */
