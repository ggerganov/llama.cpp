
 /* Copyright 2005-2014 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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

/*!
* \file cufftw.h
* \brief Public header file for the NVIDIA CUDA FFTW library (CUFFTW)
*/

#ifndef _CUFFTW_H_
#define _CUFFTW_H_


#include <stdio.h>
#include "cufft.h"

#ifdef __cplusplus
extern "C" {
#endif

// transform direction
#define FFTW_FORWARD -1
#define FFTW_INVERSE  1
#define FFTW_BACKWARD 1

// Planner flags

#define FFTW_ESTIMATE           0x01
#define FFTW_MEASURE            0x02
#define FFTW_PATIENT            0x03
#define FFTW_EXHAUSTIVE         0x04
#define FFTW_WISDOM_ONLY        0x05

//Algorithm restriction flags

#define FFTW_DESTROY_INPUT      0x08
#define FFTW_PRESERVE_INPUT     0x0C
#define FFTW_UNALIGNED          0x10
    
// CUFFTW defines and supports the following data types

// note if complex.h has been included we use the C99 complex types
#if !defined(FFTW_NO_Complex) && defined(_Complex_I) && defined (complex)
  typedef double _Complex fftw_complex;
  typedef float _Complex fftwf_complex;
#else
  typedef double fftw_complex[2];
  typedef float fftwf_complex[2];
#endif

typedef void *fftw_plan;

typedef void *fftwf_plan;

typedef struct {
    int n;
    int is;
    int os;
} fftw_iodim;

typedef fftw_iodim fftwf_iodim;
    
typedef struct {
    ptrdiff_t n;
    ptrdiff_t is;
    ptrdiff_t os;
} fftw_iodim64;

typedef fftw_iodim64 fftwf_iodim64;
    

// CUFFTW defines and supports the following double precision APIs


fftw_plan CUFFTAPI fftw_plan_dft_1d(int n, 
                                    fftw_complex *in,
                                    fftw_complex *out, 
                                    int sign, 
                                    unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_2d(int n0,
                                    int n1, 
                                    fftw_complex *in,
                                    fftw_complex *out, 
                                    int sign, 
                                    unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_3d(int n0,
                                    int n1,
                                    int n2, 
                                    fftw_complex *in,
                                    fftw_complex *out, 
                                    int sign, 
                                    unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft(int rank,
                                 const int *n,
                                 fftw_complex *in,
                                 fftw_complex *out, 
                                 int sign, 
                                 unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_r2c_1d(int n, 
                                        double *in,
                                        fftw_complex *out, 
                                        unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_r2c_2d(int n0,
                                        int n1, 
                                        double *in,
                                        fftw_complex *out, 
                                        unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_r2c_3d(int n0,
                                        int n1,
                                        int n2, 
                                        double *in,
                                        fftw_complex *out, 
                                        unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_r2c(int rank,
                                     const int *n,
                                     double *in,
                                     fftw_complex *out, 
                                     unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_c2r_1d(int n, 
                                        fftw_complex *in,
                                        double *out, 
                                        unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_c2r_2d(int n0,
                                        int n1, 
                                        fftw_complex *in,
                                        double *out, 
                                        unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_c2r_3d(int n0,
                                        int n1,
                                        int n2, 
                                        fftw_complex *in,
                                        double *out, 
                                        unsigned flags);

fftw_plan CUFFTAPI fftw_plan_dft_c2r(int rank,
                                     const int *n,
                                     fftw_complex *in,
                                     double *out, 
                                     unsigned flags);


fftw_plan CUFFTAPI fftw_plan_many_dft(int rank,
                                      const int *n,
                                      int batch,
                                      fftw_complex *in,
                                      const int *inembed, int istride, int idist,
                                      fftw_complex *out,
                                      const int *onembed, int ostride, int odist,
                                      int sign, unsigned flags);

fftw_plan CUFFTAPI fftw_plan_many_dft_r2c(int rank,
                                          const int *n,
                                          int batch,
                                          double *in,
                                          const int *inembed, int istride, int idist,
                                          fftw_complex *out,
                                          const int *onembed, int ostride, int odist,
                                          unsigned flags);

fftw_plan CUFFTAPI fftw_plan_many_dft_c2r(int rank,
                                          const int *n,
                                          int batch,
                                          fftw_complex *in,
                                          const int *inembed, int istride, int idist,
                                          double *out,
                                          const int *onembed, int ostride, int odist,
                                          unsigned flags);

fftw_plan CUFFTAPI fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                      int batch_rank, const fftw_iodim *batch_dims,
                                      fftw_complex *in, fftw_complex *out,
                                      int sign, unsigned flags);

fftw_plan CUFFTAPI fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                          int batch_rank, const fftw_iodim *batch_dims,
                                          double *in, fftw_complex *out, 
                                          unsigned flags);

fftw_plan CUFFTAPI fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims,
                                          int batch_rank, const fftw_iodim *batch_dims,
                                          fftw_complex *in, double *out, 
                                          unsigned flags);

void CUFFTAPI fftw_execute(const fftw_plan plan);

void CUFFTAPI fftw_execute_dft(const fftw_plan plan, 
                               fftw_complex *idata,
                               fftw_complex *odata);

void CUFFTAPI fftw_execute_dft_r2c(const fftw_plan plan, 
                                   double *idata,
                                   fftw_complex *odata);

void CUFFTAPI fftw_execute_dft_c2r(const fftw_plan plan, 
                                   fftw_complex *idata,
                                   double *odata);
                                   
                                   
// CUFFTW defines and supports the following single precision APIs

fftwf_plan CUFFTAPI fftwf_plan_dft_1d(int n, 
                                      fftwf_complex *in,
                                      fftwf_complex *out, 
                                      int sign, 
                                      unsigned flags);
                                   
fftwf_plan CUFFTAPI fftwf_plan_dft_2d(int n0,
                                      int n1, 
                                      fftwf_complex *in,
                                      fftwf_complex *out, 
                                      int sign, 
                                      unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_3d(int n0,
                                      int n1,
                                      int n2, 
                                      fftwf_complex *in,
                                      fftwf_complex *out, 
                                      int sign, 
                                      unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft(int rank,
                                   const int *n,
                                   fftwf_complex *in,
                                   fftwf_complex *out, 
                                   int sign, 
                                   unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_1d(int n, 
                                          float *in,
                                          fftwf_complex *out, 
                                          unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_2d(int n0,
                                          int n1, 
                                          float *in,
                                          fftwf_complex *out, 
                                          unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_3d(int n0,
                                          int n1,
                                          int n2, 
                                          float *in,
                                          fftwf_complex *out, 
                                          unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_r2c(int rank,
                                       const int *n,
                                       float *in,
                                       fftwf_complex *out, 
                                       unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_c2r_1d(int n, 
                                          fftwf_complex *in,
                                          float *out, 
                                          unsigned flags);
                                      
fftwf_plan CUFFTAPI fftwf_plan_dft_c2r_2d(int n0,
                                          int n1, 
                                          fftwf_complex *in,
                                          float *out, 
                                          unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_c2r_3d(int n0,
                                        int n1,
                                        int n2, 
                                        fftwf_complex *in,
                                        float *out, 
                                        unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_dft_c2r(int rank,
                                       const int *n,
                                       fftwf_complex *in,
                                       float *out, 
                                       unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_many_dft(int rank,
                                        const int *n,
                                        int batch,
                                        fftwf_complex *in,
                                        const int *inembed, int istride, int idist,
                                        fftwf_complex *out,
                                        const int *onembed, int ostride, int odist,
                                        int sign, unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_many_dft_r2c(int rank,
                                            const int *n,
                                            int batch,
                                            float *in,
                                            const int *inembed, int istride, int idist,
                                            fftwf_complex *out,
                                            const int *onembed, int ostride, int odist,
                                            unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_many_dft_c2r(int rank,
                                            const int *n,
                                            int batch,
                                            fftwf_complex *in,
                                            const int *inembed, int istride, int idist,
                                            float *out,
                                            const int *onembed, int ostride, int odist,
                                            unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_guru_dft(int rank, const fftwf_iodim *dims,
                                        int batch_rank, const fftwf_iodim *batch_dims,
                                        fftwf_complex *in, fftwf_complex *out,
                                        int sign, unsigned flags);
                                        
fftwf_plan CUFFTAPI fftwf_plan_guru_dft_r2c(int rank, const fftwf_iodim *dims,
                                            int batch_rank, const fftwf_iodim *batch_dims,
                                            float *in, fftwf_complex *out, 
                                            unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_guru_dft_c2r(int rank, const fftwf_iodim *dims,
                                            int batch_rank, const fftwf_iodim *batch_dims,
                                            fftwf_complex *in, float *out, 
                                            unsigned flags);

void CUFFTAPI fftwf_execute(const fftw_plan plan);

void CUFFTAPI fftwf_execute_dft(const fftwf_plan plan, 
                                fftwf_complex *idata,
                                fftwf_complex *odata);

void CUFFTAPI fftwf_execute_dft_r2c(const fftwf_plan plan, 
                                    float *idata,
                                    fftwf_complex *odata);

void CUFFTAPI fftwf_execute_dft_c2r(const fftwf_plan plan, 
                                    fftwf_complex *idata,
                                    float *odata);

/// CUFFTW 64-bit Guru Interface
/// dp
fftw_plan CUFFTAPI fftw_plan_guru64_dft(int rank, const fftw_iodim64* dims, int batch_rank, const fftw_iodim64* batch_dims, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);

fftw_plan CUFFTAPI fftw_plan_guru64_dft_r2c(int rank, const fftw_iodim64* dims, int batch_rank, const fftw_iodim64* batch_dims, double* in, fftw_complex* out, unsigned flags);

fftw_plan CUFFTAPI fftw_plan_guru64_dft_c2r(int rank, const fftw_iodim64* dims, int batch_rank, const fftw_iodim64* batch_dims, fftw_complex* in, double* out, unsigned flags);

/// sp
fftwf_plan CUFFTAPI fftwf_plan_guru64_dft(int rank, const fftwf_iodim64* dims, int batch_rank, const fftwf_iodim64* batch_dims, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_guru64_dft_r2c(int rank, const fftwf_iodim64* dims, int batch_rank, const fftwf_iodim64* batch_dims, float* in, fftwf_complex* out, unsigned flags);

fftwf_plan CUFFTAPI fftwf_plan_guru64_dft_c2r(int rank, const fftwf_iodim64* dims, int batch_rank, const fftwf_iodim64* batch_dims, fftwf_complex* in, float* out, unsigned flags);

#ifdef _WIN32
#define _CUFFTAPI(T) T CUFFTAPI
#else
#define _CUFFTAPI(T) CUFFTAPI T
#endif

// CUFFTW defines and supports the following support APIs
_CUFFTAPI(void *) fftw_malloc(size_t n);

_CUFFTAPI(void *) fftwf_malloc(size_t n);

void CUFFTAPI fftw_free(void *pointer);

void CUFFTAPI fftwf_free(void *pointer);

void CUFFTAPI fftw_export_wisdom_to_file(FILE * output_file); 

void CUFFTAPI fftwf_export_wisdom_to_file(FILE * output_file); 

void CUFFTAPI fftw_import_wisdom_from_file(FILE * input_file); 

void CUFFTAPI fftwf_import_wisdom_from_file(FILE * input_file); 

void CUFFTAPI fftw_print_plan(const fftw_plan plan);                                 

void CUFFTAPI fftwf_print_plan(const fftwf_plan plan);

void CUFFTAPI fftw_set_timelimit(double seconds);

void CUFFTAPI fftwf_set_timelimit(double seconds);

double CUFFTAPI fftw_cost(const fftw_plan plan);
                               
double CUFFTAPI fftwf_cost(const fftw_plan plan);

void CUFFTAPI fftw_flops(const fftw_plan plan, double *add, double *mul, double *fma);

void CUFFTAPI fftwf_flops(const fftw_plan plan, double *add, double *mul, double *fma);

void CUFFTAPI fftw_destroy_plan(fftw_plan plan);

void CUFFTAPI fftwf_destroy_plan(fftwf_plan plan);

void CUFFTAPI fftw_cleanup(void);

void CUFFTAPI fftwf_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* _CUFFTW_H_ */
