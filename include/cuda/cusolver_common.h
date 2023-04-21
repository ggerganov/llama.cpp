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

#if !defined(CUSOLVER_COMMON_H_)
#define CUSOLVER_COMMON_H_

#include "library_types.h"

#ifndef CUSOLVERAPI
#ifdef _WIN32
#define CUSOLVERAPI __stdcall
#else
#define CUSOLVERAPI 
#endif
#endif


#if defined(_MSC_VER)
typedef __int64 int64_t;
#else
#include <inttypes.h>
#endif

typedef int cusolver_int_t;


#define CUSOLVER_VER_MAJOR 11
#define CUSOLVER_VER_MINOR 3
#define CUSOLVER_VER_PATCH 4
#define CUSOLVER_VER_BUILD 124
#define CUSOLVER_VERSION (CUSOLVER_VER_MAJOR * 1000 + \
                        CUSOLVER_VER_MINOR *  100 + \
                        CUSOLVER_VER_PATCH)

/*
 * disable this macro to proceed old API
 */
#define DISABLE_CUSOLVER_DEPRECATED

//------------------------------------------------------------------------------

#if !defined(_MSC_VER)
#   define CUSOLVER_CPP_VERSION __cplusplus
#elif _MSC_FULL_VER >= 190024210 // Visual Studio 2015 Update 3
#   define CUSOLVER_CPP_VERSION _MSVC_LANG
#else
#   define CUSOLVER_CPP_VERSION 0
#endif

//------------------------------------------------------------------------------

#if !defined(DISABLE_CUSOLVER_DEPRECATED)

#   if CUSOLVER_CPP_VERSION >= 201402L

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            [[deprecated("please use " #new_func " instead")]]

#   elif defined(_MSC_VER)

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            __declspec(deprecated("please use " #new_func " instead"))

#   elif defined(__INTEL_COMPILER) || defined(__clang__) ||                    \
         (defined(__GNUC__) &&                                                 \
          (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            __attribute__((deprecated("please use " #new_func " instead")))

#   elif defined(__GNUC__) || defined(__xlc__)

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            __attribute__((deprecated))

#   else

#       define CUSOLVER_DEPRECATED(new_func)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L
//------------------------------------------------------------------------------

#   if CUSOLVER_CPP_VERSION >= 201703L

#       define CUSOLVER_DEPRECATED_ENUM(new_enum)                              \
            [[deprecated("please use " #new_enum " instead")]]

#   elif defined(__clang__) ||                                                 \
         (defined(__GNUC__) && __GNUC__ >= 6 && !defined(__PGI))

#       define CUSOLVER_DEPRECATED_ENUM(new_enum)                              \
            __attribute__((deprecated("please use " #new_enum " instead")))

#   else

#       define CUSOLVER_DEPRECATED_ENUM(new_enum)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L

#else // defined(DISABLE_CUSOLVER_DEPRECATED)

#   define CUSOLVER_DEPRECATED(new_func)
#   define CUSOLVER_DEPRECATED_ENUM(new_enum)

#endif // !defined(DISABLE_CUSOLVER_DEPRECATED)

#undef CUSOLVER_CPP_VERSION






#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

typedef enum{
    CUSOLVER_STATUS_SUCCESS=0,
    CUSOLVER_STATUS_NOT_INITIALIZED=1,
    CUSOLVER_STATUS_ALLOC_FAILED=2,
    CUSOLVER_STATUS_INVALID_VALUE=3,
    CUSOLVER_STATUS_ARCH_MISMATCH=4,
    CUSOLVER_STATUS_MAPPING_ERROR=5,
    CUSOLVER_STATUS_EXECUTION_FAILED=6,
    CUSOLVER_STATUS_INTERNAL_ERROR=7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT=10,
    CUSOLVER_STATUS_INVALID_LICENSE=11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED=12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID=13,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC=14,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE=15,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER=16,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR=20,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED=21,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE=22,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES=23,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED=25,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED=26,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR=30,
    CUSOLVER_STATUS_INVALID_WORKSPACE=31
} cusolverStatus_t;

typedef enum {
    CUSOLVER_EIG_TYPE_1=1,
    CUSOLVER_EIG_TYPE_2=2,
    CUSOLVER_EIG_TYPE_3=3
} cusolverEigType_t ;

typedef enum {
    CUSOLVER_EIG_MODE_NOVECTOR=0,
    CUSOLVER_EIG_MODE_VECTOR=1
} cusolverEigMode_t ;


typedef enum {
    CUSOLVER_EIG_RANGE_ALL=1001,
    CUSOLVER_EIG_RANGE_I=1002,
    CUSOLVER_EIG_RANGE_V=1003,
} cusolverEigRange_t ;



typedef enum {
    CUSOLVER_INF_NORM=104,
    CUSOLVER_MAX_NORM=105,
    CUSOLVER_ONE_NORM=106,
    CUSOLVER_FRO_NORM=107,
} cusolverNorm_t ;

typedef enum {
    CUSOLVER_IRS_REFINE_NOT_SET          = 1100,
    CUSOLVER_IRS_REFINE_NONE             = 1101,
    CUSOLVER_IRS_REFINE_CLASSICAL        = 1102,
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES  = 1103,
    CUSOLVER_IRS_REFINE_GMRES            = 1104,
    CUSOLVER_IRS_REFINE_GMRES_GMRES      = 1105,
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND    = 1106,

    CUSOLVER_PREC_DD           = 1150,
    CUSOLVER_PREC_SS           = 1151,
    CUSOLVER_PREC_SHT          = 1152,

} cusolverIRSRefinement_t;


typedef enum {
    CUSOLVER_R_8I  = 1201,
    CUSOLVER_R_8U  = 1202,
    CUSOLVER_R_64F = 1203,
    CUSOLVER_R_32F = 1204,
    CUSOLVER_R_16F = 1205,
    CUSOLVER_R_16BF  = 1206,
    CUSOLVER_R_TF32  = 1207,
    CUSOLVER_R_AP  = 1208,
    CUSOLVER_C_8I  = 1211,
    CUSOLVER_C_8U  = 1212,
    CUSOLVER_C_64F = 1213,
    CUSOLVER_C_32F = 1214,
    CUSOLVER_C_16F = 1215,
    CUSOLVER_C_16BF  = 1216,
    CUSOLVER_C_TF32  = 1217,
    CUSOLVER_C_AP  = 1218,
} cusolverPrecType_t ;

typedef enum {
   CUSOLVER_ALG_0 = 0,  /* default algorithm */
   CUSOLVER_ALG_1 = 1,
   CUSOLVER_ALG_2 = 2
} cusolverAlgMode_t;


typedef enum {
    CUBLAS_STOREV_COLUMNWISE=0, 
    CUBLAS_STOREV_ROWWISE=1
} cusolverStorevMode_t; 

typedef enum {
    CUBLAS_DIRECT_FORWARD=0, 
    CUBLAS_DIRECT_BACKWARD=1
} cusolverDirectMode_t;

cusolverStatus_t CUSOLVERAPI cusolverGetProperty(
    libraryPropertyType type, 
    int *value);

cusolverStatus_t CUSOLVERAPI cusolverGetVersion(
    int *version);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif // CUSOLVER_COMMON_H_



