 /* Copyright 2013,2014 NVIDIA Corporation.  All rights reserved.
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
* \file cudalibxt.h  
* \brief Public header file for the NVIDIA library multi-GPU support structures  
*/ 

#ifndef _CUDA_LIB_XT_H_
#define _CUDA_LIB_XT_H_
#include <cuda_runtime.h>

#define CUDA_XT_DESCRIPTOR_VERSION 0x01000000 // This is added to CUDART_VERSION

enum cudaXtCopyType_t {
    LIB_XT_COPY_HOST_TO_DEVICE,
    LIB_XT_COPY_DEVICE_TO_HOST,
    LIB_XT_COPY_DEVICE_TO_DEVICE
} ;
typedef enum cudaXtCopyType_t cudaLibXtCopyType;

enum libFormat_t {
    LIB_FORMAT_CUFFT        = 0x0,
    LIB_FORMAT_UNDEFINED    = 0x1
};

typedef enum libFormat_t libFormat;

#define MAX_CUDA_DESCRIPTOR_GPUS 64

struct cudaXtDesc_t{
    int version;                             //descriptor version
    int nGPUs;                               //number of GPUs 
    int GPUs[MAX_CUDA_DESCRIPTOR_GPUS];      //array of device IDs
    void *data[MAX_CUDA_DESCRIPTOR_GPUS];    //array of pointers to data, one per GPU
    size_t size[MAX_CUDA_DESCRIPTOR_GPUS];   //array of data sizes, one per GPU
    void *cudaXtState;                       //opaque CUDA utility structure
};
typedef struct cudaXtDesc_t cudaXtDesc;

struct cudaLibXtDesc_t{
    int version;                //descriptor version
    cudaXtDesc *descriptor;     //multi-GPU memory descriptor
    libFormat library;          //which library recognizes the format
    int subFormat;              //library specific enumerator of sub formats
    void *libDescriptor;        //library specific descriptor e.g. FFT transform plan object
};
typedef struct cudaLibXtDesc_t cudaLibXtDesc;


#endif

