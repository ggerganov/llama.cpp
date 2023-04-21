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
 * This is the public header file for the new CUBLAS library API, it mapped the generic
 * Cublas name functions to the actual _v2 implementations.
 */

#if !defined(CUBLAS_V2_H_)
#define CUBLAS_V2_H_

#undef CUBLASAPI
#ifdef __CUDACC__
#define CUBLASAPI __host__ __device__
#else
#define CUBLASAPI
#endif

#include "cublas_api.h"

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasGetVersion cublasGetVersion_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasSetStream cublasSetStream_v2
#define cublasGetStream cublasGetStream_v2
#define cublasGetPointerMode cublasGetPointerMode_v2
#define cublasSetPointerMode cublasSetPointerMode_v2

/* Blas3 Routines   */

#define cublasSnrm2 cublasSnrm2_v2
#define cublasDnrm2 cublasDnrm2_v2
#define cublasScnrm2 cublasScnrm2_v2
#define cublasDznrm2 cublasDznrm2_v2

#define cublasSdot cublasSdot_v2
#define cublasDdot cublasDdot_v2
#define cublasCdotu cublasCdotu_v2
#define cublasCdotc cublasCdotc_v2
#define cublasZdotu cublasZdotu_v2
#define cublasZdotc cublasZdotc_v2

#define cublasSscal cublasSscal_v2
#define cublasDscal cublasDscal_v2
#define cublasCscal cublasCscal_v2
#define cublasCsscal cublasCsscal_v2
#define cublasZscal cublasZscal_v2
#define cublasZdscal cublasZdscal_v2

#define cublasSaxpy cublasSaxpy_v2
#define cublasDaxpy cublasDaxpy_v2
#define cublasCaxpy cublasCaxpy_v2
#define cublasZaxpy cublasZaxpy_v2

#define cublasScopy cublasScopy_v2
#define cublasDcopy cublasDcopy_v2
#define cublasCcopy cublasCcopy_v2
#define cublasZcopy cublasZcopy_v2

#define cublasSswap cublasSswap_v2
#define cublasDswap cublasDswap_v2
#define cublasCswap cublasCswap_v2
#define cublasZswap cublasZswap_v2

#define cublasIsamax cublasIsamax_v2
#define cublasIdamax cublasIdamax_v2
#define cublasIcamax cublasIcamax_v2
#define cublasIzamax cublasIzamax_v2

#define cublasIsamin cublasIsamin_v2
#define cublasIdamin cublasIdamin_v2
#define cublasIcamin cublasIcamin_v2
#define cublasIzamin cublasIzamin_v2

#define cublasSasum cublasSasum_v2
#define cublasDasum cublasDasum_v2
#define cublasScasum cublasScasum_v2
#define cublasDzasum cublasDzasum_v2

#define cublasSrot cublasSrot_v2
#define cublasDrot cublasDrot_v2
#define cublasCrot cublasCrot_v2
#define cublasCsrot cublasCsrot_v2
#define cublasZrot cublasZrot_v2
#define cublasZdrot cublasZdrot_v2

#define cublasSrotg cublasSrotg_v2
#define cublasDrotg cublasDrotg_v2
#define cublasCrotg cublasCrotg_v2
#define cublasZrotg cublasZrotg_v2

#define cublasSrotm cublasSrotm_v2
#define cublasDrotm cublasDrotm_v2

#define cublasSrotmg cublasSrotmg_v2
#define cublasDrotmg cublasDrotmg_v2

/* Blas2 Routines */

#define cublasSgemv cublasSgemv_v2
#define cublasDgemv cublasDgemv_v2
#define cublasCgemv cublasCgemv_v2
#define cublasZgemv cublasZgemv_v2

#define cublasSgbmv cublasSgbmv_v2
#define cublasDgbmv cublasDgbmv_v2
#define cublasCgbmv cublasCgbmv_v2
#define cublasZgbmv cublasZgbmv_v2

#define cublasStrmv cublasStrmv_v2
#define cublasDtrmv cublasDtrmv_v2
#define cublasCtrmv cublasCtrmv_v2
#define cublasZtrmv cublasZtrmv_v2

#define cublasStbmv cublasStbmv_v2
#define cublasDtbmv cublasDtbmv_v2
#define cublasCtbmv cublasCtbmv_v2
#define cublasZtbmv cublasZtbmv_v2

#define cublasStpmv cublasStpmv_v2
#define cublasDtpmv cublasDtpmv_v2
#define cublasCtpmv cublasCtpmv_v2
#define cublasZtpmv cublasZtpmv_v2

#define cublasStrsv cublasStrsv_v2
#define cublasDtrsv cublasDtrsv_v2
#define cublasCtrsv cublasCtrsv_v2
#define cublasZtrsv cublasZtrsv_v2

#define cublasStpsv cublasStpsv_v2
#define cublasDtpsv cublasDtpsv_v2
#define cublasCtpsv cublasCtpsv_v2
#define cublasZtpsv cublasZtpsv_v2

#define cublasStbsv cublasStbsv_v2
#define cublasDtbsv cublasDtbsv_v2
#define cublasCtbsv cublasCtbsv_v2
#define cublasZtbsv cublasZtbsv_v2

#define cublasSsymv cublasSsymv_v2
#define cublasDsymv cublasDsymv_v2
#define cublasCsymv cublasCsymv_v2
#define cublasZsymv cublasZsymv_v2
#define cublasChemv cublasChemv_v2
#define cublasZhemv cublasZhemv_v2

#define cublasSsbmv cublasSsbmv_v2
#define cublasDsbmv cublasDsbmv_v2
#define cublasChbmv cublasChbmv_v2
#define cublasZhbmv cublasZhbmv_v2

#define cublasSspmv cublasSspmv_v2
#define cublasDspmv cublasDspmv_v2
#define cublasChpmv cublasChpmv_v2
#define cublasZhpmv cublasZhpmv_v2

#define cublasSger cublasSger_v2
#define cublasDger cublasDger_v2
#define cublasCgeru cublasCgeru_v2
#define cublasCgerc cublasCgerc_v2
#define cublasZgeru cublasZgeru_v2
#define cublasZgerc cublasZgerc_v2

#define cublasSsyr cublasSsyr_v2
#define cublasDsyr cublasDsyr_v2
#define cublasCsyr cublasCsyr_v2
#define cublasZsyr cublasZsyr_v2
#define cublasCher cublasCher_v2
#define cublasZher cublasZher_v2

#define cublasSspr cublasSspr_v2
#define cublasDspr cublasDspr_v2
#define cublasChpr cublasChpr_v2
#define cublasZhpr cublasZhpr_v2

#define cublasSsyr2 cublasSsyr2_v2
#define cublasDsyr2 cublasDsyr2_v2
#define cublasCsyr2 cublasCsyr2_v2
#define cublasZsyr2 cublasZsyr2_v2
#define cublasCher2 cublasCher2_v2
#define cublasZher2 cublasZher2_v2

#define cublasSspr2 cublasSspr2_v2
#define cublasDspr2 cublasDspr2_v2
#define cublasChpr2 cublasChpr2_v2
#define cublasZhpr2 cublasZhpr2_v2

/* Blas3 Routines   */

#define cublasSgemm cublasSgemm_v2
#define cublasDgemm cublasDgemm_v2
#define cublasCgemm cublasCgemm_v2
#define cublasZgemm cublasZgemm_v2

#define cublasSsyrk cublasSsyrk_v2
#define cublasDsyrk cublasDsyrk_v2
#define cublasCsyrk cublasCsyrk_v2
#define cublasZsyrk cublasZsyrk_v2
#define cublasCherk cublasCherk_v2
#define cublasZherk cublasZherk_v2

#define cublasSsyr2k cublasSsyr2k_v2
#define cublasDsyr2k cublasDsyr2k_v2
#define cublasCsyr2k cublasCsyr2k_v2
#define cublasZsyr2k cublasZsyr2k_v2
#define cublasCher2k cublasCher2k_v2
#define cublasZher2k cublasZher2k_v2

#define cublasSsymm cublasSsymm_v2
#define cublasDsymm cublasDsymm_v2
#define cublasCsymm cublasCsymm_v2
#define cublasZsymm cublasZsymm_v2
#define cublasChemm cublasChemm_v2
#define cublasZhemm cublasZhemm_v2

#define cublasStrsm cublasStrsm_v2
#define cublasDtrsm cublasDtrsm_v2
#define cublasCtrsm cublasCtrsm_v2
#define cublasZtrsm cublasZtrsm_v2

#define cublasStrmm cublasStrmm_v2
#define cublasDtrmm cublasDtrmm_v2
#define cublasCtrmm cublasCtrmm_v2
#define cublasZtrmm cublasZtrmm_v2

#endif /* !defined(CUBLAS_V2_H_) */
