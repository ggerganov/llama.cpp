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

/*
 * This is the public header file for the CUBLAS library, defining the API
 *
 * CUBLAS is an implementation of BLAS (Basic Linear Algebra Subroutines)
 * on top of the CUDA runtime.
 */

#if !defined(CUBLAS_H_)
#define CUBLAS_H_

#include <cuda_runtime.h>

#ifndef CUBLASWINAPI
#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI
#endif
#endif

#undef CUBLASAPI
#ifdef __CUDACC__
#define CUBLASAPI __host__
#else
#define CUBLASAPI
#endif

#include "cublas_api.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* CUBLAS data types */
#define cublasStatus cublasStatus_t

cublasStatus CUBLASWINAPI cublasInit(void);
cublasStatus CUBLASWINAPI cublasShutdown(void);
cublasStatus CUBLASWINAPI cublasGetError(void);

cublasStatus CUBLASWINAPI cublasGetVersion(int* version);
cublasStatus CUBLASWINAPI cublasAlloc(int n, int elemSize, void** devicePtr);

cublasStatus CUBLASWINAPI cublasFree(void* devicePtr);

cublasStatus CUBLASWINAPI cublasSetKernelStream(cudaStream_t stream);

/* ---------------- CUBLAS BLAS1 functions ---------------- */
/* NRM2 */
float CUBLASWINAPI cublasSnrm2(int n, const float* x, int incx);
double CUBLASWINAPI cublasDnrm2(int n, const double* x, int incx);
float CUBLASWINAPI cublasScnrm2(int n, const cuComplex* x, int incx);
double CUBLASWINAPI cublasDznrm2(int n, const cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* DOT */
float CUBLASWINAPI cublasSdot(int n, const float* x, int incx, const float* y, int incy);
double CUBLASWINAPI cublasDdot(int n, const double* x, int incx, const double* y, int incy);
cuComplex CUBLASWINAPI cublasCdotu(int n, const cuComplex* x, int incx, const cuComplex* y, int incy);
cuComplex CUBLASWINAPI cublasCdotc(int n, const cuComplex* x, int incx, const cuComplex* y, int incy);
cuDoubleComplex CUBLASWINAPI cublasZdotu(int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy);
cuDoubleComplex CUBLASWINAPI cublasZdotc(int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy);
/*------------------------------------------------------------------------*/
/* SCAL */
void CUBLASWINAPI cublasSscal(int n, float alpha, float* x, int incx);
void CUBLASWINAPI cublasDscal(int n, double alpha, double* x, int incx);
void CUBLASWINAPI cublasCscal(int n, cuComplex alpha, cuComplex* x, int incx);
void CUBLASWINAPI cublasZscal(int n, cuDoubleComplex alpha, cuDoubleComplex* x, int incx);

void CUBLASWINAPI cublasCsscal(int n, float alpha, cuComplex* x, int incx);
void CUBLASWINAPI cublasZdscal(int n, double alpha, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* AXPY */
void CUBLASWINAPI cublasSaxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
void CUBLASWINAPI cublasDaxpy(int n, double alpha, const double* x, int incx, double* y, int incy);
void CUBLASWINAPI cublasCaxpy(int n, cuComplex alpha, const cuComplex* x, int incx, cuComplex* y, int incy);
void CUBLASWINAPI
cublasZaxpy(int n, cuDoubleComplex alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
/*------------------------------------------------------------------------*/
/* COPY */
void CUBLASWINAPI cublasScopy(int n, const float* x, int incx, float* y, int incy);
void CUBLASWINAPI cublasDcopy(int n, const double* x, int incx, double* y, int incy);
void CUBLASWINAPI cublasCcopy(int n, const cuComplex* x, int incx, cuComplex* y, int incy);
void CUBLASWINAPI cublasZcopy(int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
/*------------------------------------------------------------------------*/
/* SWAP */
void CUBLASWINAPI cublasSswap(int n, float* x, int incx, float* y, int incy);
void CUBLASWINAPI cublasDswap(int n, double* x, int incx, double* y, int incy);
void CUBLASWINAPI cublasCswap(int n, cuComplex* x, int incx, cuComplex* y, int incy);
void CUBLASWINAPI cublasZswap(int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
/*------------------------------------------------------------------------*/
/* AMAX */
int CUBLASWINAPI cublasIsamax(int n, const float* x, int incx);
int CUBLASWINAPI cublasIdamax(int n, const double* x, int incx);
int CUBLASWINAPI cublasIcamax(int n, const cuComplex* x, int incx);
int CUBLASWINAPI cublasIzamax(int n, const cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* AMIN */
int CUBLASWINAPI cublasIsamin(int n, const float* x, int incx);
int CUBLASWINAPI cublasIdamin(int n, const double* x, int incx);

int CUBLASWINAPI cublasIcamin(int n, const cuComplex* x, int incx);
int CUBLASWINAPI cublasIzamin(int n, const cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* ASUM */
float CUBLASWINAPI cublasSasum(int n, const float* x, int incx);
double CUBLASWINAPI cublasDasum(int n, const double* x, int incx);
float CUBLASWINAPI cublasScasum(int n, const cuComplex* x, int incx);
double CUBLASWINAPI cublasDzasum(int n, const cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* ROT */
void CUBLASWINAPI cublasSrot(int n, float* x, int incx, float* y, int incy, float sc, float ss);
void CUBLASWINAPI cublasDrot(int n, double* x, int incx, double* y, int incy, double sc, double ss);
void CUBLASWINAPI cublasCrot(int n, cuComplex* x, int incx, cuComplex* y, int incy, float c, cuComplex s);
void CUBLASWINAPI
cublasZrot(int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, double sc, cuDoubleComplex cs);
void CUBLASWINAPI cublasCsrot(int n, cuComplex* x, int incx, cuComplex* y, int incy, float c, float s);
void CUBLASWINAPI cublasZdrot(int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, double c, double s);
/*------------------------------------------------------------------------*/
/* ROTG */
void CUBLASWINAPI cublasSrotg(float* sa, float* sb, float* sc, float* ss);
void CUBLASWINAPI cublasDrotg(double* sa, double* sb, double* sc, double* ss);
void CUBLASWINAPI cublasCrotg(cuComplex* ca, cuComplex cb, float* sc, cuComplex* cs);
void CUBLASWINAPI cublasZrotg(cuDoubleComplex* ca, cuDoubleComplex cb, double* sc, cuDoubleComplex* cs);
/*------------------------------------------------------------------------*/
/* ROTM */
void CUBLASWINAPI cublasSrotm(int n, float* x, int incx, float* y, int incy, const float* sparam);
void CUBLASWINAPI cublasDrotm(int n, double* x, int incx, double* y, int incy, const double* sparam);
/*------------------------------------------------------------------------*/
/* ROTMG */
void CUBLASWINAPI cublasSrotmg(float* sd1, float* sd2, float* sx1, const float* sy1, float* sparam);
void CUBLASWINAPI cublasDrotmg(double* sd1, double* sd2, double* sx1, const double* sy1, double* sparam);

/* --------------- CUBLAS BLAS2 functions  ---------------- */
/* GEMV */
void CUBLASWINAPI cublasSgemv(char trans,
                              int m,
                              int n,
                              float alpha,
                              const float* A,
                              int lda,
                              const float* x,
                              int incx,
                              float beta,
                              float* y,
                              int incy);
void CUBLASWINAPI cublasDgemv(char trans,
                              int m,
                              int n,
                              double alpha,
                              const double* A,
                              int lda,
                              const double* x,
                              int incx,
                              double beta,
                              double* y,
                              int incy);
void CUBLASWINAPI cublasCgemv(char trans,
                              int m,
                              int n,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* x,
                              int incx,
                              cuComplex beta,
                              cuComplex* y,
                              int incy);
void CUBLASWINAPI cublasZgemv(char trans,
                              int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* x,
                              int incx,
                              cuDoubleComplex beta,
                              cuDoubleComplex* y,
                              int incy);
/*------------------------------------------------------------------------*/
/* GBMV */
void CUBLASWINAPI cublasSgbmv(char trans,
                              int m,
                              int n,
                              int kl,
                              int ku,
                              float alpha,
                              const float* A,
                              int lda,
                              const float* x,
                              int incx,
                              float beta,
                              float* y,
                              int incy);
void CUBLASWINAPI cublasDgbmv(char trans,
                              int m,
                              int n,
                              int kl,
                              int ku,
                              double alpha,
                              const double* A,
                              int lda,
                              const double* x,
                              int incx,
                              double beta,
                              double* y,
                              int incy);
void CUBLASWINAPI cublasCgbmv(char trans,
                              int m,
                              int n,
                              int kl,
                              int ku,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* x,
                              int incx,
                              cuComplex beta,
                              cuComplex* y,
                              int incy);
void CUBLASWINAPI cublasZgbmv(char trans,
                              int m,
                              int n,
                              int kl,
                              int ku,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* x,
                              int incx,
                              cuDoubleComplex beta,
                              cuDoubleComplex* y,
                              int incy);
/*------------------------------------------------------------------------*/
/* TRMV */
void CUBLASWINAPI cublasStrmv(char uplo, char trans, char diag, int n, const float* A, int lda, float* x, int incx);
void CUBLASWINAPI cublasDtrmv(char uplo, char trans, char diag, int n, const double* A, int lda, double* x, int incx);
void CUBLASWINAPI
cublasCtrmv(char uplo, char trans, char diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
void CUBLASWINAPI
cublasZtrmv(char uplo, char trans, char diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* TBMV */
void CUBLASWINAPI
cublasStbmv(char uplo, char trans, char diag, int n, int k, const float* A, int lda, float* x, int incx);
void CUBLASWINAPI
cublasDtbmv(char uplo, char trans, char diag, int n, int k, const double* A, int lda, double* x, int incx);
void CUBLASWINAPI
cublasCtbmv(char uplo, char trans, char diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
void CUBLASWINAPI cublasZtbmv(
    char uplo, char trans, char diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* TPMV */
void CUBLASWINAPI cublasStpmv(char uplo, char trans, char diag, int n, const float* AP, float* x, int incx);

void CUBLASWINAPI cublasDtpmv(char uplo, char trans, char diag, int n, const double* AP, double* x, int incx);

void CUBLASWINAPI cublasCtpmv(char uplo, char trans, char diag, int n, const cuComplex* AP, cuComplex* x, int incx);

void CUBLASWINAPI
cublasZtpmv(char uplo, char trans, char diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* TRSV */
void CUBLASWINAPI cublasStrsv(char uplo, char trans, char diag, int n, const float* A, int lda, float* x, int incx);

void CUBLASWINAPI cublasDtrsv(char uplo, char trans, char diag, int n, const double* A, int lda, double* x, int incx);

void CUBLASWINAPI
cublasCtrsv(char uplo, char trans, char diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);

void CUBLASWINAPI
cublasZtrsv(char uplo, char trans, char diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* TPSV */
void CUBLASWINAPI cublasStpsv(char uplo, char trans, char diag, int n, const float* AP, float* x, int incx);

void CUBLASWINAPI cublasDtpsv(char uplo, char trans, char diag, int n, const double* AP, double* x, int incx);

void CUBLASWINAPI cublasCtpsv(char uplo, char trans, char diag, int n, const cuComplex* AP, cuComplex* x, int incx);

void CUBLASWINAPI
cublasZtpsv(char uplo, char trans, char diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* TBSV */
void CUBLASWINAPI
cublasStbsv(char uplo, char trans, char diag, int n, int k, const float* A, int lda, float* x, int incx);

void CUBLASWINAPI
cublasDtbsv(char uplo, char trans, char diag, int n, int k, const double* A, int lda, double* x, int incx);
void CUBLASWINAPI
cublasCtbsv(char uplo, char trans, char diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);

void CUBLASWINAPI cublasZtbsv(
    char uplo, char trans, char diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
/*------------------------------------------------------------------------*/
/* SYMV/HEMV */
void CUBLASWINAPI cublasSsymv(
    char uplo, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy);
void CUBLASWINAPI cublasDsymv(char uplo,
                              int n,
                              double alpha,
                              const double* A,
                              int lda,
                              const double* x,
                              int incx,
                              double beta,
                              double* y,
                              int incy);
void CUBLASWINAPI cublasChemv(char uplo,
                              int n,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* x,
                              int incx,
                              cuComplex beta,
                              cuComplex* y,
                              int incy);
void CUBLASWINAPI cublasZhemv(char uplo,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* x,
                              int incx,
                              cuDoubleComplex beta,
                              cuDoubleComplex* y,
                              int incy);
/*------------------------------------------------------------------------*/
/* SBMV/HBMV */
void CUBLASWINAPI cublasSsbmv(char uplo,
                              int n,
                              int k,
                              float alpha,
                              const float* A,
                              int lda,
                              const float* x,
                              int incx,
                              float beta,
                              float* y,
                              int incy);
void CUBLASWINAPI cublasDsbmv(char uplo,
                              int n,
                              int k,
                              double alpha,
                              const double* A,
                              int lda,
                              const double* x,
                              int incx,
                              double beta,
                              double* y,
                              int incy);
void CUBLASWINAPI cublasChbmv(char uplo,
                              int n,
                              int k,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* x,
                              int incx,
                              cuComplex beta,
                              cuComplex* y,
                              int incy);
void CUBLASWINAPI cublasZhbmv(char uplo,
                              int n,
                              int k,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* x,
                              int incx,
                              cuDoubleComplex beta,
                              cuDoubleComplex* y,
                              int incy);
/*------------------------------------------------------------------------*/
/* SPMV/HPMV */
void CUBLASWINAPI
cublasSspmv(char uplo, int n, float alpha, const float* AP, const float* x, int incx, float beta, float* y, int incy);
void CUBLASWINAPI cublasDspmv(
    char uplo, int n, double alpha, const double* AP, const double* x, int incx, double beta, double* y, int incy);
void CUBLASWINAPI cublasChpmv(char uplo,
                              int n,
                              cuComplex alpha,
                              const cuComplex* AP,
                              const cuComplex* x,
                              int incx,
                              cuComplex beta,
                              cuComplex* y,
                              int incy);
void CUBLASWINAPI cublasZhpmv(char uplo,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* AP,
                              const cuDoubleComplex* x,
                              int incx,
                              cuDoubleComplex beta,
                              cuDoubleComplex* y,
                              int incy);

/*------------------------------------------------------------------------*/
/* GER */
void CUBLASWINAPI
cublasSger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
void CUBLASWINAPI
cublasDger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);

void CUBLASWINAPI cublasCgeru(
    int m, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
void CUBLASWINAPI cublasCgerc(
    int m, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
void CUBLASWINAPI cublasZgeru(int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* x,
                              int incx,
                              const cuDoubleComplex* y,
                              int incy,
                              cuDoubleComplex* A,
                              int lda);
void CUBLASWINAPI cublasZgerc(int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* x,
                              int incx,
                              const cuDoubleComplex* y,
                              int incy,
                              cuDoubleComplex* A,
                              int lda);
/*------------------------------------------------------------------------*/
/* SYR/HER */
void CUBLASWINAPI cublasSsyr(char uplo, int n, float alpha, const float* x, int incx, float* A, int lda);
void CUBLASWINAPI cublasDsyr(char uplo, int n, double alpha, const double* x, int incx, double* A, int lda);

void CUBLASWINAPI cublasCher(char uplo, int n, float alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
void CUBLASWINAPI
cublasZher(char uplo, int n, double alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);

/*------------------------------------------------------------------------*/
/* SPR/HPR */
void CUBLASWINAPI cublasSspr(char uplo, int n, float alpha, const float* x, int incx, float* AP);
void CUBLASWINAPI cublasDspr(char uplo, int n, double alpha, const double* x, int incx, double* AP);
void CUBLASWINAPI cublasChpr(char uplo, int n, float alpha, const cuComplex* x, int incx, cuComplex* AP);
void CUBLASWINAPI cublasZhpr(char uplo, int n, double alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP);
/*------------------------------------------------------------------------*/
/* SYR2/HER2 */
void CUBLASWINAPI
cublasSsyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
void CUBLASWINAPI
cublasDsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
void CUBLASWINAPI cublasCher2(char uplo,
                              int n,
                              cuComplex alpha,
                              const cuComplex* x,
                              int incx,
                              const cuComplex* y,
                              int incy,
                              cuComplex* A,
                              int lda);
void CUBLASWINAPI cublasZher2(char uplo,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* x,
                              int incx,
                              const cuDoubleComplex* y,
                              int incy,
                              cuDoubleComplex* A,
                              int lda);

/*------------------------------------------------------------------------*/
/* SPR2/HPR2 */
void CUBLASWINAPI
cublasSspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* AP);
void CUBLASWINAPI
cublasDspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* AP);
void CUBLASWINAPI cublasChpr2(
    char uplo, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP);
void CUBLASWINAPI cublasZhpr2(char uplo,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* x,
                              int incx,
                              const cuDoubleComplex* y,
                              int incy,
                              cuDoubleComplex* AP);
/* ------------------------BLAS3 Functions ------------------------------- */
/* GEMM */
void CUBLASWINAPI cublasSgemm(char transa,
                              char transb,
                              int m,
                              int n,
                              int k,
                              float alpha,
                              const float* A,
                              int lda,
                              const float* B,
                              int ldb,
                              float beta,
                              float* C,
                              int ldc);
void CUBLASWINAPI cublasDgemm(char transa,
                              char transb,
                              int m,
                              int n,
                              int k,
                              double alpha,
                              const double* A,
                              int lda,
                              const double* B,
                              int ldb,
                              double beta,
                              double* C,
                              int ldc);
void CUBLASWINAPI cublasCgemm(char transa,
                              char transb,
                              int m,
                              int n,
                              int k,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* B,
                              int ldb,
                              cuComplex beta,
                              cuComplex* C,
                              int ldc);
void CUBLASWINAPI cublasZgemm(char transa,
                              char transb,
                              int m,
                              int n,
                              int k,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* B,
                              int ldb,
                              cuDoubleComplex beta,
                              cuDoubleComplex* C,
                              int ldc);
/* -------------------------------------------------------*/
/* SYRK */
void CUBLASWINAPI
cublasSsyrk(char uplo, char trans, int n, int k, float alpha, const float* A, int lda, float beta, float* C, int ldc);
void CUBLASWINAPI cublasDsyrk(
    char uplo, char trans, int n, int k, double alpha, const double* A, int lda, double beta, double* C, int ldc);

void CUBLASWINAPI cublasCsyrk(char uplo,
                              char trans,
                              int n,
                              int k,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              cuComplex beta,
                              cuComplex* C,
                              int ldc);
void CUBLASWINAPI cublasZsyrk(char uplo,
                              char trans,
                              int n,
                              int k,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              cuDoubleComplex beta,
                              cuDoubleComplex* C,
                              int ldc);
/* ------------------------------------------------------- */
/* HERK */
void CUBLASWINAPI cublasCherk(
    char uplo, char trans, int n, int k, float alpha, const cuComplex* A, int lda, float beta, cuComplex* C, int ldc);
void CUBLASWINAPI cublasZherk(char uplo,
                              char trans,
                              int n,
                              int k,
                              double alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              double beta,
                              cuDoubleComplex* C,
                              int ldc);
/* ------------------------------------------------------- */
/* SYR2K */
void CUBLASWINAPI cublasSsyr2k(char uplo,
                               char trans,
                               int n,
                               int k,
                               float alpha,
                               const float* A,
                               int lda,
                               const float* B,
                               int ldb,
                               float beta,
                               float* C,
                               int ldc);

void CUBLASWINAPI cublasDsyr2k(char uplo,
                               char trans,
                               int n,
                               int k,
                               double alpha,
                               const double* A,
                               int lda,
                               const double* B,
                               int ldb,
                               double beta,
                               double* C,
                               int ldc);
void CUBLASWINAPI cublasCsyr2k(char uplo,
                               char trans,
                               int n,
                               int k,
                               cuComplex alpha,
                               const cuComplex* A,
                               int lda,
                               const cuComplex* B,
                               int ldb,
                               cuComplex beta,
                               cuComplex* C,
                               int ldc);

void CUBLASWINAPI cublasZsyr2k(char uplo,
                               char trans,
                               int n,
                               int k,
                               cuDoubleComplex alpha,
                               const cuDoubleComplex* A,
                               int lda,
                               const cuDoubleComplex* B,
                               int ldb,
                               cuDoubleComplex beta,
                               cuDoubleComplex* C,
                               int ldc);
/* ------------------------------------------------------- */
/* HER2K */
void CUBLASWINAPI cublasCher2k(char uplo,
                               char trans,
                               int n,
                               int k,
                               cuComplex alpha,
                               const cuComplex* A,
                               int lda,
                               const cuComplex* B,
                               int ldb,
                               float beta,
                               cuComplex* C,
                               int ldc);

void CUBLASWINAPI cublasZher2k(char uplo,
                               char trans,
                               int n,
                               int k,
                               cuDoubleComplex alpha,
                               const cuDoubleComplex* A,
                               int lda,
                               const cuDoubleComplex* B,
                               int ldb,
                               double beta,
                               cuDoubleComplex* C,
                               int ldc);

/*------------------------------------------------------------------------*/
/* SYMM*/
void CUBLASWINAPI cublasSsymm(char side,
                              char uplo,
                              int m,
                              int n,
                              float alpha,
                              const float* A,
                              int lda,
                              const float* B,
                              int ldb,
                              float beta,
                              float* C,
                              int ldc);
void CUBLASWINAPI cublasDsymm(char side,
                              char uplo,
                              int m,
                              int n,
                              double alpha,
                              const double* A,
                              int lda,
                              const double* B,
                              int ldb,
                              double beta,
                              double* C,
                              int ldc);

void CUBLASWINAPI cublasCsymm(char side,
                              char uplo,
                              int m,
                              int n,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* B,
                              int ldb,
                              cuComplex beta,
                              cuComplex* C,
                              int ldc);

void CUBLASWINAPI cublasZsymm(char side,
                              char uplo,
                              int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* B,
                              int ldb,
                              cuDoubleComplex beta,
                              cuDoubleComplex* C,
                              int ldc);
/*------------------------------------------------------------------------*/
/* HEMM*/
void CUBLASWINAPI cublasChemm(char side,
                              char uplo,
                              int m,
                              int n,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              const cuComplex* B,
                              int ldb,
                              cuComplex beta,
                              cuComplex* C,
                              int ldc);
void CUBLASWINAPI cublasZhemm(char side,
                              char uplo,
                              int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              const cuDoubleComplex* B,
                              int ldb,
                              cuDoubleComplex beta,
                              cuDoubleComplex* C,
                              int ldc);

/*------------------------------------------------------------------------*/
/* TRSM*/
void CUBLASWINAPI cublasStrsm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              float alpha,
                              const float* A,
                              int lda,
                              float* B,
                              int ldb);

void CUBLASWINAPI cublasDtrsm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              double alpha,
                              const double* A,
                              int lda,
                              double* B,
                              int ldb);

void CUBLASWINAPI cublasCtrsm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              cuComplex* B,
                              int ldb);

void CUBLASWINAPI cublasZtrsm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              cuDoubleComplex* B,
                              int ldb);
/*------------------------------------------------------------------------*/
/* TRMM*/
void CUBLASWINAPI cublasStrmm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              float alpha,
                              const float* A,
                              int lda,
                              float* B,
                              int ldb);
void CUBLASWINAPI cublasDtrmm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              double alpha,
                              const double* A,
                              int lda,
                              double* B,
                              int ldb);
void CUBLASWINAPI cublasCtrmm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              cuComplex alpha,
                              const cuComplex* A,
                              int lda,
                              cuComplex* B,
                              int ldb);
void CUBLASWINAPI cublasZtrmm(char side,
                              char uplo,
                              char transa,
                              char diag,
                              int m,
                              int n,
                              cuDoubleComplex alpha,
                              const cuDoubleComplex* A,
                              int lda,
                              cuDoubleComplex* B,
                              int ldb);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(CUBLAS_H_) */
