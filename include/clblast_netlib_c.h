
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Netlib CBLAS interface to the CLBlast BLAS routines, performing all buffer
// copies automatically and running on the default OpenCL platform and device. For full control over
// performance, it is advised to use the regular clblast.h or clblast_c.h headers instead.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_NETLIB_C_H_
#define CLBLAST_CLBLAST_NETLIB_C_H_

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#if defined(_WIN32) && defined(CLBLAST_DLL)
  #if defined(COMPILING_DLL)
    #define PUBLIC_API __declspec(dllexport)
  #else
    #define PUBLIC_API __declspec(dllimport)
  #endif
#else
  #define PUBLIC_API
#endif

// The C interface
#ifdef __cplusplus
extern "C" {
#endif

// =================================================================================================

// Matrix layout and transpose types
typedef enum CLBlastLayout_ { CLBlastLayoutRowMajor = 101,
                              CLBlastLayoutColMajor = 102 } CLBlastLayout;
typedef enum CLBlastTranspose_ { CLBlastTransposeNo = 111, CLBlastTransposeYes = 112,
                                 CLBlastTransposeConjugate = 113 } CLBlastTranspose;
typedef enum CLBlastTriangle_ { CLBlastTriangleUpper = 121,
                                CLBlastTriangleLower = 122 } CLBlastTriangle;
typedef enum CLBlastDiagonal_ { CLBlastDiagonalNonUnit = 131,
                                CLBlastDiagonalUnit = 132 } CLBlastDiagonal;
typedef enum CLBlastSide_ { CLBlastSideLeft = 141, CLBlastSideRight = 142 } CLBlastSide;
typedef enum CLBlastKernelMode_ { CLBlastKernelModeCrossCorrelation = 141, CLBlastKernelModeConvolution = 152 } CLBlastKernelMode;

// For full compatibility with CBLAS
typedef CLBlastLayout CBLAS_ORDER;
typedef CLBlastTranspose CBLAS_TRANSPOSE;
typedef CLBlastTriangle CBLAS_UPLO;
typedef CLBlastDiagonal CBLAS_DIAG;
typedef CLBlastSide CBLAS_SIDE;
#define CblasRowMajor CLBlastLayoutRowMajor
#define CblasColMajor CLBlastLayoutColMajor
#define CblasNoTrans CLBlastTransposeNo
#define CblasTrans CLBlastTransposeYes
#define CblasConjTrans CLBlastTransposeConjugate
#define CblasUpper CLBlastTriangleUpper
#define CblasLower CLBlastTriangleLower
#define CblasNonUnit CLBlastDiagonalNonUnit
#define CblasUnit CLBlastDiagonalUnit
#define CblasLeft CLBlastSideLeft
#define CblasRight CLBlastSideRight

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
void PUBLIC_API cblas_srotg(float* sa,
                            float* sb,
                            float* sc,
                            float* ss);
void PUBLIC_API cblas_drotg(double* sa,
                            double* sb,
                            double* sc,
                            double* ss);

// Generate modified givens plane rotation: SROTMG/DROTMG
void PUBLIC_API cblas_srotmg(float* sd1,
                             float* sd2,
                             float* sx1,
                             const float sy1,
                             float* sparam);
void PUBLIC_API cblas_drotmg(double* sd1,
                             double* sd2,
                             double* sx1,
                             const double sy1,
                             double* sparam);

// Apply givens plane rotation: SROT/DROT
void PUBLIC_API cblas_srot(const int n,
                           float* x, const int x_inc,
                           float* y, const int y_inc,
                           const float cos,
                           const float sin);
void PUBLIC_API cblas_drot(const int n,
                           double* x, const int x_inc,
                           double* y, const int y_inc,
                           const double cos,
                           const double sin);

// Apply modified givens plane rotation: SROTM/DROTM
void PUBLIC_API cblas_srotm(const int n,
                            float* x, const int x_inc,
                            float* y, const int y_inc,
                            float* sparam);
void PUBLIC_API cblas_drotm(const int n,
                            double* x, const int x_inc,
                            double* y, const int y_inc,
                            double* sparam);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
void PUBLIC_API cblas_sswap(const int n,
                            float* x, const int x_inc,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dswap(const int n,
                            double* x, const int x_inc,
                            double* y, const int y_inc);
void PUBLIC_API cblas_cswap(const int n,
                            void* x, const int x_inc,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zswap(const int n,
                            void* x, const int x_inc,
                            void* y, const int y_inc);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
void PUBLIC_API cblas_sscal(const int n,
                            const float alpha,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dscal(const int n,
                            const double alpha,
                            double* x, const int x_inc);
void PUBLIC_API cblas_cscal(const int n,
                            const void* alpha,
                            void* x, const int x_inc);
void PUBLIC_API cblas_zscal(const int n,
                            const void* alpha,
                            void* x, const int x_inc);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
void PUBLIC_API cblas_scopy(const int n,
                            const float* x, const int x_inc,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dcopy(const int n,
                            const double* x, const int x_inc,
                            double* y, const int y_inc);
void PUBLIC_API cblas_ccopy(const int n,
                            const void* x, const int x_inc,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zcopy(const int n,
                            const void* x, const int x_inc,
                            void* y, const int y_inc);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
void PUBLIC_API cblas_saxpy(const int n,
                            const float alpha,
                            const float* x, const int x_inc,
                            float* y, const int y_inc);
void PUBLIC_API cblas_daxpy(const int n,
                            const double alpha,
                            const double* x, const int x_inc,
                            double* y, const int y_inc);
void PUBLIC_API cblas_caxpy(const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zaxpy(const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            void* y, const int y_inc);

// Dot product of two vectors: SDOT/DDOT/HDOT
float PUBLIC_API cblas_sdot(const int n,
                            const float* x, const int x_inc,
                            const float* y, const int y_inc);
double PUBLIC_API cblas_ddot(const int n,
                             const double* x, const int x_inc,
                             const double* y, const int y_inc);

// Dot product of two complex vectors: CDOTU/ZDOTU
void PUBLIC_API cblas_cdotu_sub(const int n,
                                const void* x, const int x_inc,
                                const void* y, const int y_inc,
                                void* dot);
void PUBLIC_API cblas_zdotu_sub(const int n,
                                const void* x, const int x_inc,
                                const void* y, const int y_inc,
                                void* dot);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
void PUBLIC_API cblas_cdotc_sub(const int n,
                                const void* x, const int x_inc,
                                const void* y, const int y_inc,
                                void* dot);
void PUBLIC_API cblas_zdotc_sub(const int n,
                                const void* x, const int x_inc,
                                const void* y, const int y_inc,
                                void* dot);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
float PUBLIC_API cblas_snrm2(const int n,
                             const float* x, const int x_inc);
double PUBLIC_API cblas_dnrm2(const int n,
                              const double* x, const int x_inc);
float PUBLIC_API cblas_scnrm2(const int n,
                             const void* x, const int x_inc);
double PUBLIC_API cblas_dznrm2(const int n,
                              const void* x, const int x_inc);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
float PUBLIC_API cblas_sasum(const int n,
                             const float* x, const int x_inc);
double PUBLIC_API cblas_dasum(const int n,
                              const double* x, const int x_inc);
float PUBLIC_API cblas_scasum(const int n,
                             const void* x, const int x_inc);
double PUBLIC_API cblas_dzasum(const int n,
                              const void* x, const int x_inc);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
float PUBLIC_API cblas_ssum(const int n,
                            const float* x, const int x_inc);
double PUBLIC_API cblas_dsum(const int n,
                             const double* x, const int x_inc);
float PUBLIC_API cblas_scsum(const int n,
                            const void* x, const int x_inc);
double PUBLIC_API cblas_dzsum(const int n,
                             const void* x, const int x_inc);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
int PUBLIC_API cblas_isamax(const int n,
                           const float* x, const int x_inc);
int PUBLIC_API cblas_idamax(const int n,
                           const double* x, const int x_inc);
int PUBLIC_API cblas_icamax(const int n,
                           const void* x, const int x_inc);
int PUBLIC_API cblas_izamax(const int n,
                           const void* x, const int x_inc);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
int PUBLIC_API cblas_isamin(const int n,
                           const float* x, const int x_inc);
int PUBLIC_API cblas_idamin(const int n,
                           const double* x, const int x_inc);
int PUBLIC_API cblas_icamin(const int n,
                           const void* x, const int x_inc);
int PUBLIC_API cblas_izamin(const int n,
                           const void* x, const int x_inc);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
int PUBLIC_API cblas_ismax(const int n,
                          const float* x, const int x_inc);
int PUBLIC_API cblas_idmax(const int n,
                          const double* x, const int x_inc);
int PUBLIC_API cblas_icmax(const int n,
                          const void* x, const int x_inc);
int PUBLIC_API cblas_izmax(const int n,
                          const void* x, const int x_inc);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
int PUBLIC_API cblas_ismin(const int n,
                          const float* x, const int x_inc);
int PUBLIC_API cblas_idmin(const int n,
                          const double* x, const int x_inc);
int PUBLIC_API cblas_icmin(const int n,
                          const void* x, const int x_inc);
int PUBLIC_API cblas_izmin(const int n,
                          const void* x, const int x_inc);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
void PUBLIC_API cblas_sgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_cgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
void PUBLIC_API cblas_sgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);
void PUBLIC_API cblas_cgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                            const int m, const int n, const int kl, const int ku,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
void PUBLIC_API cblas_chemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zhemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
void PUBLIC_API cblas_chbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n, const int k,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zhbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n, const int k,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
void PUBLIC_API cblas_chpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* ap,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);
void PUBLIC_API cblas_zhpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* ap,
                            const void* x, const int x_inc,
                            const void* beta,
                            void* y, const int y_inc);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
void PUBLIC_API cblas_ssymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
void PUBLIC_API cblas_ssbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n, const int k,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n, const int k,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
void PUBLIC_API cblas_sspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const float alpha,
                            const float* ap,
                            const float* x, const int x_inc,
                            const float beta,
                            float* y, const int y_inc);
void PUBLIC_API cblas_dspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const double alpha,
                            const double* ap,
                            const double* x, const int x_inc,
                            const double beta,
                            double* y, const int y_inc);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
void PUBLIC_API cblas_strmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);
void PUBLIC_API cblas_ztrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
void PUBLIC_API cblas_stbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);
void PUBLIC_API cblas_ztbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
void PUBLIC_API cblas_stpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const float* ap,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const double* ap,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* ap,
                            void* x, const int x_inc);
void PUBLIC_API cblas_ztpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* ap,
                            void* x, const int x_inc);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
void PUBLIC_API cblas_strsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);
void PUBLIC_API cblas_ztrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
void PUBLIC_API cblas_stbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const float* a, const int a_ld,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const double* a, const int a_ld,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);
void PUBLIC_API cblas_ztbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n, const int k,
                            const void* a, const int a_ld,
                            void* x, const int x_inc);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
void PUBLIC_API cblas_stpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const float* ap,
                            float* x, const int x_inc);
void PUBLIC_API cblas_dtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const double* ap,
                            double* x, const int x_inc);
void PUBLIC_API cblas_ctpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* ap,
                            void* x, const int x_inc);
void PUBLIC_API cblas_ztpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int n,
                            const void* ap,
                            void* x, const int x_inc);

// General rank-1 matrix update: SGER/DGER/HGER
void PUBLIC_API cblas_sger(const CLBlastLayout layout,
                           const int m, const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           const float* y, const int y_inc,
                           float* a, const int a_ld);
void PUBLIC_API cblas_dger(const CLBlastLayout layout,
                           const int m, const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           const double* y, const int y_inc,
                           double* a, const int a_ld);

// General rank-1 complex matrix update: CGERU/ZGERU
void PUBLIC_API cblas_cgeru(const CLBlastLayout layout,
                            const int m, const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* a, const int a_ld);
void PUBLIC_API cblas_zgeru(const CLBlastLayout layout,
                            const int m, const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* a, const int a_ld);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
void PUBLIC_API cblas_cgerc(const CLBlastLayout layout,
                            const int m, const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* a, const int a_ld);
void PUBLIC_API cblas_zgerc(const CLBlastLayout layout,
                            const int m, const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* a, const int a_ld);

// Hermitian rank-1 matrix update: CHER/ZHER
void PUBLIC_API cblas_cher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const float alpha,
                           const void* x, const int x_inc,
                           void* a, const int a_ld);
void PUBLIC_API cblas_zher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const double alpha,
                           const void* x, const int x_inc,
                           void* a, const int a_ld);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
void PUBLIC_API cblas_chpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const float alpha,
                           const void* x, const int x_inc,
                           void* ap);
void PUBLIC_API cblas_zhpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const double alpha,
                           const void* x, const int x_inc,
                           void* ap);

// Hermitian rank-2 matrix update: CHER2/ZHER2
void PUBLIC_API cblas_cher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* a, const int a_ld);
void PUBLIC_API cblas_zher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* a, const int a_ld);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
void PUBLIC_API cblas_chpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* ap);
void PUBLIC_API cblas_zhpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const void* alpha,
                            const void* x, const int x_inc,
                            const void* y, const int y_inc,
                            void* ap);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
void PUBLIC_API cblas_ssyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           float* a, const int a_ld);
void PUBLIC_API cblas_dsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           double* a, const int a_ld);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
void PUBLIC_API cblas_sspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           float* ap);
void PUBLIC_API cblas_dspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                           const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           double* ap);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
void PUBLIC_API cblas_ssyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const float alpha,
                            const float* x, const int x_inc,
                            const float* y, const int y_inc,
                            float* a, const int a_ld);
void PUBLIC_API cblas_dsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const double alpha,
                            const double* x, const int x_inc,
                            const double* y, const int y_inc,
                            double* a, const int a_ld);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
void PUBLIC_API cblas_sspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const float alpha,
                            const float* x, const int x_inc,
                            const float* y, const int y_inc,
                            float* ap);
void PUBLIC_API cblas_dspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                            const int n,
                            const double alpha,
                            const double* x, const int x_inc,
                            const double* y, const int y_inc,
                            double* ap);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
void PUBLIC_API cblas_sgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                            const int m, const int n, const int k,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* b, const int b_ld,
                            const float beta,
                            float* c, const int c_ld);
void PUBLIC_API cblas_dgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                            const int m, const int n, const int k,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* b, const int b_ld,
                            const double beta,
                            double* c, const int c_ld);
void PUBLIC_API cblas_cgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                            const int m, const int n, const int k,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* b, const int b_ld,
                            const void* beta,
                            void* c, const int c_ld);
void PUBLIC_API cblas_zgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                            const int m, const int n, const int k,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* b, const int b_ld,
                            const void* beta,
                            void* c, const int c_ld);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
void PUBLIC_API cblas_ssymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float* b, const int b_ld,
                            const float beta,
                            float* c, const int c_ld);
void PUBLIC_API cblas_dsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double* b, const int b_ld,
                            const double beta,
                            double* c, const int c_ld);
void PUBLIC_API cblas_csymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* b, const int b_ld,
                            const void* beta,
                            void* c, const int c_ld);
void PUBLIC_API cblas_zsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* b, const int b_ld,
                            const void* beta,
                            void* c, const int c_ld);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
void PUBLIC_API cblas_chemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* b, const int b_ld,
                            const void* beta,
                            void* c, const int c_ld);
void PUBLIC_API cblas_zhemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* b, const int b_ld,
                            const void* beta,
                            void* c, const int c_ld);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
void PUBLIC_API cblas_ssyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                            const int n, const int k,
                            const float alpha,
                            const float* a, const int a_ld,
                            const float beta,
                            float* c, const int c_ld);
void PUBLIC_API cblas_dsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                            const int n, const int k,
                            const double alpha,
                            const double* a, const int a_ld,
                            const double beta,
                            double* c, const int c_ld);
void PUBLIC_API cblas_csyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                            const int n, const int k,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* beta,
                            void* c, const int c_ld);
void PUBLIC_API cblas_zsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                            const int n, const int k,
                            const void* alpha,
                            const void* a, const int a_ld,
                            const void* beta,
                            void* c, const int c_ld);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
void PUBLIC_API cblas_cherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                            const int n, const int k,
                            const float alpha,
                            const void* a, const int a_ld,
                            const float beta,
                            void* c, const int c_ld);
void PUBLIC_API cblas_zherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                            const int n, const int k,
                            const double alpha,
                            const void* a, const int a_ld,
                            const double beta,
                            void* c, const int c_ld);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
void PUBLIC_API cblas_ssyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                             const int n, const int k,
                             const float alpha,
                             const float* a, const int a_ld,
                             const float* b, const int b_ld,
                             const float beta,
                             float* c, const int c_ld);
void PUBLIC_API cblas_dsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                             const int n, const int k,
                             const double alpha,
                             const double* a, const int a_ld,
                             const double* b, const int b_ld,
                             const double beta,
                             double* c, const int c_ld);
void PUBLIC_API cblas_csyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const void* a, const int a_ld,
                             const void* b, const int b_ld,
                             const void* beta,
                             void* c, const int c_ld);
void PUBLIC_API cblas_zsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const void* a, const int a_ld,
                             const void* b, const int b_ld,
                             const void* beta,
                             void* c, const int c_ld);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
void PUBLIC_API cblas_cher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const void* a, const int a_ld,
                             const void* b, const int b_ld,
                             const float beta,
                             void* c, const int c_ld);
void PUBLIC_API cblas_zher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                             const int n, const int k,
                             const void* alpha,
                             const void* a, const int a_ld,
                             const void* b, const int b_ld,
                             const double beta,
                             void* c, const int c_ld);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
void PUBLIC_API cblas_strmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            float* b, const int b_ld);
void PUBLIC_API cblas_dtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            double* b, const int b_ld);
void PUBLIC_API cblas_ctrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            void* b, const int b_ld);
void PUBLIC_API cblas_ztrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            void* b, const int b_ld);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
void PUBLIC_API cblas_strsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const float alpha,
                            const float* a, const int a_ld,
                            float* b, const int b_ld);
void PUBLIC_API cblas_dtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const double alpha,
                            const double* a, const int a_ld,
                            double* b, const int b_ld);
void PUBLIC_API cblas_ctrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            void* b, const int b_ld);
void PUBLIC_API cblas_ztrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                            const int m, const int n,
                            const void* alpha,
                            const void* a, const int a_ld,
                            void* b, const int b_ld);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
void PUBLIC_API cblas_shad(const int n,
                           const float alpha,
                           const float* x, const int x_inc,
                           const float* y, const int y_inc,
                           const float beta,
                           float* z, const int z_inc);
void PUBLIC_API cblas_dhad(const int n,
                           const double alpha,
                           const double* x, const int x_inc,
                           const double* y, const int y_inc,
                           const double beta,
                           double* z, const int z_inc);
void PUBLIC_API cblas_chad(const int n,
                           const void* alpha,
                           const void* x, const int x_inc,
                           const void* y, const int y_inc,
                           const void* beta,
                           void* z, const int z_inc);
void PUBLIC_API cblas_zhad(const int n,
                           const void* alpha,
                           const void* x, const int x_inc,
                           const void* y, const int y_inc,
                           const void* beta,
                           void* z, const int z_inc);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
void PUBLIC_API cblas_somatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                const int m, const int n,
                                const float alpha,
                                const float* a, const int a_ld,
                                float* b, const int b_ld);
void PUBLIC_API cblas_domatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                const int m, const int n,
                                const double alpha,
                                const double* a, const int a_ld,
                                double* b, const int b_ld);
void PUBLIC_API cblas_comatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                const int m, const int n,
                                const void* alpha,
                                const void* a, const int a_ld,
                                void* b, const int b_ld);
void PUBLIC_API cblas_zomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                const int m, const int n,
                                const void* alpha,
                                const void* a, const int a_ld,
                                void* b, const int b_ld);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
void PUBLIC_API cblas_sim2col(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const float* im,
                              float* col);
void PUBLIC_API cblas_dim2col(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const double* im,
                              double* col);
void PUBLIC_API cblas_cim2col(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const void* im,
                              void* col);
void PUBLIC_API cblas_zim2col(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const void* im,
                              void* col);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
void PUBLIC_API cblas_scol2im(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const float* col,
                              float* im);
void PUBLIC_API cblas_dcol2im(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const double* col,
                              double* im);
void PUBLIC_API cblas_ccol2im(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const void* col,
                              void* im);
void PUBLIC_API cblas_zcol2im(const CLBlastKernelMode kernel_mode,
                              const int channels, const int height, const int width, const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                              const void* col,
                              void* im);

// =================================================================================================

#ifdef __cplusplus
} // extern "C"
#endif

// CLBLAST_CLBLAST_NETLIB_C_H_
#endif
