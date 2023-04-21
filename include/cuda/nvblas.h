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

#if !defined(NVBLAS_H_)
#define NVBLAS_H_

#include "driver_types.h"
#include "cuComplex.h" /* import complex data type */

#if defined(__cplusplus)
extern "C" {
#endif

/* GEMM */
void sgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const float* alpha,
            const float* a,
            const int* lda,
            const float* b,
            const int* ldb,
            const float* beta,
            float* c,
            const int* ldc);

void dgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* b,
            const int* ldb,
            const double* beta,
            double* c,
            const int* ldc);

void cgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            const cuComplex* b,
            const int* ldb,
            const cuComplex* beta,
            cuComplex* c,
            const int* ldc);

void zgemm_(const char* transa,
            const char* transb,
            const int* m,
            const int* n,
            const int* k,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const cuDoubleComplex* b,
            const int* ldb,
            const cuDoubleComplex* beta,
            cuDoubleComplex* c,
            const int* ldc);

void sgemm(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const float* alpha,
           const float* a,
           const int* lda,
           const float* b,
           const int* ldb,
           const float* beta,
           float* c,
           const int* ldc);

void dgemm(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const double* alpha,
           const double* a,
           const int* lda,
           const double* b,
           const int* ldb,
           const double* beta,
           double* c,
           const int* ldc);

void cgemm(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const cuComplex* alpha,
           const cuComplex* a,
           const int* lda,
           const cuComplex* b,
           const int* ldb,
           const cuComplex* beta,
           cuComplex* c,
           const int* ldc);

void zgemm(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const cuDoubleComplex* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           const cuDoubleComplex* b,
           const int* ldb,
           const cuDoubleComplex* beta,
           cuDoubleComplex* c,
           const int* ldc);

/* SYRK */
void ssyrk_(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const float* alpha,
            const float* a,
            const int* lda,
            const float* beta,
            float* c,
            const int* ldc);

void dsyrk_(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* beta,
            double* c,
            const int* ldc);

void csyrk_(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            const cuComplex* beta,
            cuComplex* c,
            const int* ldc);

void zsyrk_(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const cuDoubleComplex* beta,
            cuDoubleComplex* c,
            const int* ldc);

void ssyrk(const char* uplo,
           const char* trans,
           const int* n,
           const int* k,
           const float* alpha,
           const float* a,
           const int* lda,
           const float* beta,
           float* c,
           const int* ldc);

void dsyrk(const char* uplo,
           const char* trans,
           const int* n,
           const int* k,
           const double* alpha,
           const double* a,
           const int* lda,
           const double* beta,
           double* c,
           const int* ldc);

void csyrk(const char* uplo,
           const char* trans,
           const int* n,
           const int* k,
           const cuComplex* alpha,
           const cuComplex* a,
           const int* lda,
           const cuComplex* beta,
           cuComplex* c,
           const int* ldc);

void zsyrk(const char* uplo,
           const char* trans,
           const int* n,
           const int* k,
           const cuDoubleComplex* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           const cuDoubleComplex* beta,
           cuDoubleComplex* c,
           const int* ldc);

/* HERK */
void cherk_(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const float* alpha,
            const cuComplex* a,
            const int* lda,
            const float* beta,
            cuComplex* c,
            const int* ldc);

void zherk_(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const double* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const double* beta,
            cuDoubleComplex* c,
            const int* ldc);

void cherk(const char* uplo,
           const char* trans,
           const int* n,
           const int* k,
           const float* alpha,
           const cuComplex* a,
           const int* lda,
           const float* beta,
           cuComplex* c,
           const int* ldc);

void zherk(const char* uplo,
           const char* trans,
           const int* n,
           const int* k,
           const double* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           const double* beta,
           cuDoubleComplex* c,
           const int* ldc);

/* TRSM */
void strsm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const float* alpha,
            const float* a,
            const int* lda,
            float* b,
            const int* ldb);

void dtrsm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const double* alpha,
            const double* a,
            const int* lda,
            double* b,
            const int* ldb);

void ctrsm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            cuComplex* b,
            const int* ldb);

void ztrsm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            cuDoubleComplex* b,
            const int* ldb);

void strsm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const float* alpha,
           const float* a,
           const int* lda,
           float* b,
           const int* ldb);

void dtrsm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const double* alpha,
           const double* a,
           const int* lda,
           double* b,
           const int* ldb);

void ctrsm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const cuComplex* alpha,
           const cuComplex* a,
           const int* lda,
           cuComplex* b,
           const int* ldb);

void ztrsm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const cuDoubleComplex* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           cuDoubleComplex* b,
           const int* ldb);

/* SYMM */
void ssymm_(const char* side,
            const char* uplo,
            const int* m,
            const int* n,
            const float* alpha,
            const float* a,
            const int* lda,
            const float* b,
            const int* ldb,
            const float* beta,
            float* c,
            const int* ldc);

void dsymm_(const char* side,
            const char* uplo,
            const int* m,
            const int* n,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* b,
            const int* ldb,
            const double* beta,
            double* c,
            const int* ldc);

void csymm_(const char* side,
            const char* uplo,
            const int* m,
            const int* n,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            const cuComplex* b,
            const int* ldb,
            const cuComplex* beta,
            cuComplex* c,
            const int* ldc);

void zsymm_(const char* side,
            const char* uplo,
            const int* m,
            const int* n,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const cuDoubleComplex* b,
            const int* ldb,
            const cuDoubleComplex* beta,
            cuDoubleComplex* c,
            const int* ldc);

void ssymm(const char* side,
           const char* uplo,
           const int* m,
           const int* n,
           const float* alpha,
           const float* a,
           const int* lda,
           const float* b,
           const int* ldb,
           const float* beta,
           float* c,
           const int* ldc);

void dsymm(const char* side,
           const char* uplo,
           const int* m,
           const int* n,
           const double* alpha,
           const double* a,
           const int* lda,
           const double* b,
           const int* ldb,
           const double* beta,
           double* c,
           const int* ldc);

void csymm(const char* side,
           const char* uplo,
           const int* m,
           const int* n,
           const cuComplex* alpha,
           const cuComplex* a,
           const int* lda,
           const cuComplex* b,
           const int* ldb,
           const cuComplex* beta,
           cuComplex* c,
           const int* ldc);

void zsymm(const char* side,
           const char* uplo,
           const int* m,
           const int* n,
           const cuDoubleComplex* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           const cuDoubleComplex* b,
           const int* ldb,
           const cuDoubleComplex* beta,
           cuDoubleComplex* c,
           const int* ldc);

/* HEMM */
void chemm_(const char* side,
            const char* uplo,
            const int* m,
            const int* n,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            const cuComplex* b,
            const int* ldb,
            const cuComplex* beta,
            cuComplex* c,
            const int* ldc);

void zhemm_(const char* side,
            const char* uplo,
            const int* m,
            const int* n,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const cuDoubleComplex* b,
            const int* ldb,
            const cuDoubleComplex* beta,
            cuDoubleComplex* c,
            const int* ldc);

/* HEMM with no underscore*/
void chemm(const char* side,
           const char* uplo,
           const int* m,
           const int* n,
           const cuComplex* alpha,
           const cuComplex* a,
           const int* lda,
           const cuComplex* b,
           const int* ldb,
           const cuComplex* beta,
           cuComplex* c,
           const int* ldc);

void zhemm(const char* side,
           const char* uplo,
           const int* m,
           const int* n,
           const cuDoubleComplex* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           const cuDoubleComplex* b,
           const int* ldb,
           const cuDoubleComplex* beta,
           cuDoubleComplex* c,
           const int* ldc);

/* SYR2K */
void ssyr2k_(const char* uplo,
             const char* trans,
             const int* n,
             const int* k,
             const float* alpha,
             const float* a,
             const int* lda,
             const float* b,
             const int* ldb,
             const float* beta,
             float* c,
             const int* ldc);

void dsyr2k_(const char* uplo,
             const char* trans,
             const int* n,
             const int* k,
             const double* alpha,
             const double* a,
             const int* lda,
             const double* b,
             const int* ldb,
             const double* beta,
             double* c,
             const int* ldc);

void csyr2k_(const char* uplo,
             const char* trans,
             const int* n,
             const int* k,
             const cuComplex* alpha,
             const cuComplex* a,
             const int* lda,
             const cuComplex* b,
             const int* ldb,
             const cuComplex* beta,
             cuComplex* c,
             const int* ldc);

void zsyr2k_(const char* uplo,
             const char* trans,
             const int* n,
             const int* k,
             const cuDoubleComplex* alpha,
             const cuDoubleComplex* a,
             const int* lda,
             const cuDoubleComplex* b,
             const int* ldb,
             const cuDoubleComplex* beta,
             cuDoubleComplex* c,
             const int* ldc);

/* SYR2K no_underscore*/
void ssyr2k(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const float* alpha,
            const float* a,
            const int* lda,
            const float* b,
            const int* ldb,
            const float* beta,
            float* c,
            const int* ldc);

void dsyr2k(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const double* alpha,
            const double* a,
            const int* lda,
            const double* b,
            const int* ldb,
            const double* beta,
            double* c,
            const int* ldc);

void csyr2k(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            const cuComplex* b,
            const int* ldb,
            const cuComplex* beta,
            cuComplex* c,
            const int* ldc);

void zsyr2k(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const cuDoubleComplex* b,
            const int* ldb,
            const cuDoubleComplex* beta,
            cuDoubleComplex* c,
            const int* ldc);

/* HERK */
void cher2k_(const char* uplo,
             const char* trans,
             const int* n,
             const int* k,
             const cuComplex* alpha,
             const cuComplex* a,
             const int* lda,
             const cuComplex* b,
             const int* ldb,
             const float* beta,
             cuComplex* c,
             const int* ldc);

void zher2k_(const char* uplo,
             const char* trans,
             const int* n,
             const int* k,
             const cuDoubleComplex* alpha,
             const cuDoubleComplex* a,
             const int* lda,
             const cuDoubleComplex* b,
             const int* ldb,
             const double* beta,
             cuDoubleComplex* c,
             const int* ldc);

/* HER2K with no underscore */
void cher2k(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            const cuComplex* b,
            const int* ldb,
            const float* beta,
            cuComplex* c,
            const int* ldc);

void zher2k(const char* uplo,
            const char* trans,
            const int* n,
            const int* k,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            const cuDoubleComplex* b,
            const int* ldb,
            const double* beta,
            cuDoubleComplex* c,
            const int* ldc);

/* TRMM */
void strmm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const float* alpha,
            const float* a,
            const int* lda,
            float* b,
            const int* ldb);

void dtrmm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const double* alpha,
            const double* a,
            const int* lda,
            double* b,
            const int* ldb);

void ctrmm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const cuComplex* alpha,
            const cuComplex* a,
            const int* lda,
            cuComplex* b,
            const int* ldb);

void ztrmm_(const char* side,
            const char* uplo,
            const char* transa,
            const char* diag,
            const int* m,
            const int* n,
            const cuDoubleComplex* alpha,
            const cuDoubleComplex* a,
            const int* lda,
            cuDoubleComplex* b,
            const int* ldb);

void strmm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const float* alpha,
           const float* a,
           const int* lda,
           float* b,
           const int* ldb);

void dtrmm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const double* alpha,
           const double* a,
           const int* lda,
           double* b,
           const int* ldb);

void ctrmm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const cuComplex* alpha,
           const cuComplex* a,
           const int* lda,
           cuComplex* b,
           const int* ldb);

void ztrmm(const char* side,
           const char* uplo,
           const char* transa,
           const char* diag,
           const int* m,
           const int* n,
           const cuDoubleComplex* alpha,
           const cuDoubleComplex* a,
           const int* lda,
           cuDoubleComplex* b,
           const int* ldb);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* !defined(NVBLAS_H_) */
