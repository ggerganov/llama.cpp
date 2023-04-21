
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
* \file cufftXt.h
* \brief Public header file for the NVIDIA CUDA FFT library (CUFFT)
*/

#ifndef _CUFFTXT_H_
#define _CUFFTXT_H_
#include "cudalibxt.h"
#include "cufft.h"


#ifndef CUFFTAPI
#ifdef _WIN32
#define CUFFTAPI __stdcall
#else
#define CUFFTAPI
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// cufftXtSubFormat identifies the data layout of
// a memory descriptor owned by cufft.
// note that multi GPU cufft does not yet support out-of-place transforms
//

typedef enum cufftXtSubFormat_t {
    CUFFT_XT_FORMAT_INPUT = 0x00,              //by default input is in linear order across GPUs
    CUFFT_XT_FORMAT_OUTPUT = 0x01,             //by default output is in scrambled order depending on transform
    CUFFT_XT_FORMAT_INPLACE = 0x02,            //by default inplace is input order, which is linear across GPUs
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,   //shuffled output order after execution of the transform
    CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,  //shuffled input order prior to execution of 1D transforms
    CUFFT_FORMAT_UNDEFINED = 0x05
} cufftXtSubFormat;

//
// cufftXtCopyType specifies the type of copy for cufftXtMemcpy
//
typedef enum cufftXtCopyType_t {
    CUFFT_COPY_HOST_TO_DEVICE = 0x00,
    CUFFT_COPY_DEVICE_TO_HOST = 0x01,
    CUFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    CUFFT_COPY_UNDEFINED = 0x03
} cufftXtCopyType;

//
// cufftXtQueryType specifies the type of query for cufftXtQueryPlan
//
typedef enum cufftXtQueryType_t {
    CUFFT_QUERY_1D_FACTORS = 0x00,
    CUFFT_QUERY_UNDEFINED = 0x01
} cufftXtQueryType;

typedef struct cufftXt1dFactors_t {
    long long int size;
    long long int stringCount;
    long long int stringLength;
    long long int substringLength;
    long long int factor1;
    long long int factor2;
    long long int stringMask;
    long long int substringMask;
    long long int factor1Mask;
    long long int factor2Mask;
    int stringShift;
    int substringShift;
    int factor1Shift;
    int factor2Shift;
} cufftXt1dFactors;

//
// cufftXtWorkAreaPolicy specifies policy for cufftXtSetWorkAreaPolicy
//
typedef enum cufftXtWorkAreaPolicy_t {
    CUFFT_WORKAREA_MINIMAL = 0, /* maximum reduction */
    CUFFT_WORKAREA_USER = 1, /* use workSize parameter as limit */
    CUFFT_WORKAREA_PERFORMANCE = 2, /* default - 1x overhead or more, maximum performance */
} cufftXtWorkAreaPolicy;

// multi-GPU routines
cufftResult CUFFTAPI cufftXtSetGPUs(cufftHandle handle, int nGPUs, int *whichGPUs);

cufftResult CUFFTAPI cufftXtMalloc(cufftHandle plan,
                                   cudaLibXtDesc ** descriptor,
                                   cufftXtSubFormat format);

cufftResult CUFFTAPI cufftXtMemcpy(cufftHandle plan,
                                   void *dstPointer,
                                   void *srcPointer,
                                   cufftXtCopyType type);

cufftResult CUFFTAPI cufftXtFree(cudaLibXtDesc *descriptor);

cufftResult CUFFTAPI cufftXtSetWorkArea(cufftHandle plan, void **workArea);

cufftResult CUFFTAPI cufftXtExecDescriptorC2C(cufftHandle plan,
                                              cudaLibXtDesc *input,
                                              cudaLibXtDesc *output,
                                              int direction);

cufftResult CUFFTAPI cufftXtExecDescriptorR2C(cufftHandle plan,
                                              cudaLibXtDesc *input,
                                              cudaLibXtDesc *output);

cufftResult CUFFTAPI cufftXtExecDescriptorC2R(cufftHandle plan,
                                              cudaLibXtDesc *input,
                                              cudaLibXtDesc *output);

cufftResult CUFFTAPI cufftXtExecDescriptorZ2Z(cufftHandle plan,
                                              cudaLibXtDesc *input,
                                              cudaLibXtDesc *output,
                                              int direction);

cufftResult CUFFTAPI cufftXtExecDescriptorD2Z(cufftHandle plan,
                                              cudaLibXtDesc *input,
                                              cudaLibXtDesc *output);

cufftResult CUFFTAPI cufftXtExecDescriptorZ2D(cufftHandle plan,
                                              cudaLibXtDesc *input,
                                              cudaLibXtDesc *output);

// Utility functions

cufftResult CUFFTAPI cufftXtQueryPlan(cufftHandle plan, void *queryStruct, cufftXtQueryType queryType);


// callbacks


typedef enum cufftXtCallbackType_t {
    CUFFT_CB_LD_COMPLEX = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE = 0x3,
    CUFFT_CB_ST_COMPLEX = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE = 0x7,
    CUFFT_CB_UNDEFINED = 0x8

} cufftXtCallbackType;

typedef cufftComplex (*cufftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleComplex (*cufftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftReal (*cufftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleReal(*cufftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);

typedef void (*cufftCallbackStoreC)(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreZ)(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreR)(void *dataOut, size_t offset, cufftReal element, void *callerInfo, void *sharedPointer);
typedef void (*cufftCallbackStoreD)(void *dataOut, size_t offset, cufftDoubleReal element, void *callerInfo, void *sharedPointer);


cufftResult CUFFTAPI cufftXtSetCallback(cufftHandle plan, void **callback_routine, cufftXtCallbackType cbType, void **caller_info);
cufftResult CUFFTAPI cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType);
cufftResult CUFFTAPI cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize);

cufftResult CUFFTAPI cufftXtMakePlanMany(cufftHandle plan,
                                         int rank,
                                         long long int *n,
                                         long long int *inembed,
                                         long long int istride,
                                         long long int idist,
                                         cudaDataType inputtype,
                                         long long int *onembed,
                                         long long int ostride,
                                         long long int odist,
                                         cudaDataType outputtype,
                                         long long int batch,
                                         size_t *workSize,
                                       	 cudaDataType executiontype);

cufftResult CUFFTAPI cufftXtGetSizeMany(cufftHandle plan,
                                        int rank,
                                        long long int *n,
                                        long long int *inembed,
                                        long long int istride,
                                        long long int idist,
                                        cudaDataType inputtype,
                                        long long int *onembed,
                                        long long int ostride,
                                        long long int odist,
                                        cudaDataType outputtype,
                                        long long int batch,
                                        size_t *workSize,
                                        cudaDataType executiontype);


cufftResult CUFFTAPI cufftXtExec(cufftHandle plan,
                                 void *input,
                                 void *output,
                                 int direction);

cufftResult CUFFTAPI cufftXtExecDescriptor(cufftHandle plan,
                                           cudaLibXtDesc *input,
                                           cudaLibXtDesc *output,
                                           int direction);

cufftResult CUFFTAPI cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t *workSize);

#ifdef __cplusplus
}
#endif

#endif
