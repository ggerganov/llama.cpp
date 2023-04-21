/*
 * Copyright 2019 NVIDIA Corporation.  All rights reserved.
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

#if !defined(CUSOLVERMG_H_)
#define CUSOLVERMG_H_

#include <stdint.h>
#include "cusolverDn.h"


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

struct cusolverMgContext;
typedef struct cusolverMgContext *cusolverMgHandle_t;


/**
 * \beief This enum decides how 1D device Ids (or process ranks) get mapped to a 2D grid.
 */
typedef enum {

  CUDALIBMG_GRID_MAPPING_ROW_MAJOR = 1,
  CUDALIBMG_GRID_MAPPING_COL_MAJOR = 0

} cusolverMgGridMapping_t;

/** \brief Opaque structure of the distributed grid */
typedef void * cudaLibMgGrid_t;
/** \brief Opaque structure of the distributed matrix descriptor */
typedef void * cudaLibMgMatrixDesc_t;


cusolverStatus_t CUSOLVERAPI cusolverMgCreate(
    cusolverMgHandle_t *handle);

cusolverStatus_t CUSOLVERAPI cusolverMgDestroy(
    cusolverMgHandle_t handle);

cusolverStatus_t CUSOLVERAPI cusolverMgDeviceSelect(
    cusolverMgHandle_t handle,
    int nbDevices,
    int deviceId[]);


/**
 * \brief Allocates resources related to the shared memory device grid.
 * \param[out] grid the opaque data strcuture that holds the grid
 * \param[in] numRowDevices number of devices in the row
 * \param[in] numColDevices number of devices in the column
 * \param[in] deviceId This array of size height * width stores the
 *            device-ids of the 2D grid; each entry must correspond to a valid gpu or to -1 (denoting CPU).
 * \param[in] mapping whether the 2D grid is in row/column major
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgCreateDeviceGrid(
    cudaLibMgGrid_t* grid, 
    int32_t numRowDevices, 
    int32_t numColDevices,
    const int32_t deviceId[], 
    cusolverMgGridMapping_t mapping);

/**
 * \brief Releases the allocated resources related to the distributed grid.
 * \param[in] grid the opaque data strcuture that holds the distributed grid
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgDestroyGrid(
    cudaLibMgGrid_t grid);

/**
 * \brief Allocates resources related to the distributed matrix descriptor.
 * \param[out] desc the opaque data strcuture that holds the descriptor
 * \param[in] numRows number of total rows
 * \param[in] numCols number of total columns
 * \param[in] rowBlockSize row block size
 * \param[in] colBlockSize column block size
 * \param[in] dataType the data type of each element in cudaDataType
 * \param[in] grid the opaque data structure of the distributed grid
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgCreateMatrixDesc(
    cudaLibMgMatrixDesc_t * desc,
    int64_t numRows, 
    int64_t numCols, 
    int64_t rowBlockSize, 
    int64_t colBlockSize,
    cudaDataType dataType, 
    const cudaLibMgGrid_t grid);

/**
 * \brief Releases the allocated resources related to the distributed matrix descriptor.
 * \param[in] desc the opaque data strcuture that holds the descriptor
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgDestroyMatrixDesc(
    cudaLibMgMatrixDesc_t desc);



cusolverStatus_t CUSOLVERAPI cusolverMgSyevd_bufferSize(
    cusolverMgHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int N,
    void *array_d_A[], 
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    void *W,
    cudaDataType dataTypeW,
    cudaDataType computeType,
    int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgSyevd(
    cusolverMgHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    cudaLibMgMatrixDesc_t descrA,
    void *W,
    cudaDataType dataTypeW,
    cudaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverMgGetrf_bufferSize(
    cusolverMgHandle_t handle,
    int M,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],
    cudaDataType computeType,
    int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgGetrf(
    cusolverMgHandle_t handle,
    int M,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],
    cudaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverMgGetrs_bufferSize(
    cusolverMgHandle_t handle,
    cublasOperation_t TRANS,
    int N,
    int NRHS,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],  
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType,
    int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgGetrs(
    cusolverMgHandle_t handle,
    cublasOperation_t TRANS,
    int N,
    int NRHS,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[], 
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverMgPotrf_bufferSize( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA,
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
	int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgPotrf( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
    void *array_d_work[],
    int64_t lwork,
    int *h_info);

cusolverStatus_t CUSOLVERAPI cusolverMgPotrs_bufferSize( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int n, 
	int nrhs,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType, 
	int64_t *lwork );

cusolverStatus_t CUSOLVERAPI cusolverMgPotrs( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int n, 
	int nrhs,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType, 
    void *array_d_work[],
	int64_t lwork,
	int *h_info);

cusolverStatus_t CUSOLVERAPI cusolverMgPotri_bufferSize( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
	int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgPotri( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
    void *array_d_work[],
	int64_t lwork,
    int *h_info);



#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // CUSOLVERMG_H_
 

