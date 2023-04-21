/*
 * Copyright 2020-2021 NVIDIA Corporation.  All rights reserved.
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

#ifndef CUDATYPEDEFS_H
#define CUDATYPEDEFS_H

#include <cuda.h>

#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptds
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## ptds_version ## _ptsz
#else
    #define __API_TYPEDEF_PTDS(api, default_version, ptds_version) api ## _v ## default_version
    #define __API_TYPEDEF_PTSZ(api, default_version, ptds_version) api ## _v ## default_version
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * Macros for the latest version for each driver function in cuda.h
 */
#define PFN_cuGetErrorString  PFN_cuGetErrorString_v6000
#define PFN_cuGetErrorName  PFN_cuGetErrorName_v6000
#define PFN_cuInit  PFN_cuInit_v2000
#define PFN_cuDriverGetVersion  PFN_cuDriverGetVersion_v2020
#define PFN_cuDeviceGet  PFN_cuDeviceGet_v2000
#define PFN_cuDeviceGetCount  PFN_cuDeviceGetCount_v2000
#define PFN_cuDeviceGetName  PFN_cuDeviceGetName_v2000
#define PFN_cuDeviceGetUuid  PFN_cuDeviceGetUuid_v11040
#define PFN_cuDeviceGetLuid  PFN_cuDeviceGetLuid_v10000
#define PFN_cuDeviceTotalMem  PFN_cuDeviceTotalMem_v3020
#define PFN_cuDeviceGetTexture1DLinearMaxWidth  PFN_cuDeviceGetTexture1DLinearMaxWidth_v11010
#define PFN_cuDeviceGetAttribute  PFN_cuDeviceGetAttribute_v2000
#define PFN_cuDeviceGetNvSciSyncAttributes  PFN_cuDeviceGetNvSciSyncAttributes_v10020
#define PFN_cuDeviceSetMemPool  PFN_cuDeviceSetMemPool_v11020
#define PFN_cuDeviceGetMemPool  PFN_cuDeviceGetMemPool_v11020
#define PFN_cuDeviceGetDefaultMemPool  PFN_cuDeviceGetDefaultMemPool_v11020
#define PFN_cuDeviceGetProperties  PFN_cuDeviceGetProperties_v2000
#define PFN_cuDeviceComputeCapability  PFN_cuDeviceComputeCapability_v2000
#define PFN_cuDevicePrimaryCtxRetain  PFN_cuDevicePrimaryCtxRetain_v7000
#define PFN_cuDevicePrimaryCtxRelease  PFN_cuDevicePrimaryCtxRelease_v11000
#define PFN_cuDevicePrimaryCtxSetFlags  PFN_cuDevicePrimaryCtxSetFlags_v11000
#define PFN_cuDevicePrimaryCtxGetState  PFN_cuDevicePrimaryCtxGetState_v7000
#define PFN_cuDevicePrimaryCtxReset  PFN_cuDevicePrimaryCtxReset_v11000
#define PFN_cuDeviceGetExecAffinitySupport  PFN_cuDeviceGetExecAffinitySupport_v11040
#define PFN_cuCtxCreate  PFN_cuCtxCreate_v11040
#define PFN_cuCtxDestroy  PFN_cuCtxDestroy_v4000
#define PFN_cuCtxPushCurrent  PFN_cuCtxPushCurrent_v4000
#define PFN_cuCtxPopCurrent  PFN_cuCtxPopCurrent_v4000
#define PFN_cuCtxSetCurrent  PFN_cuCtxSetCurrent_v4000
#define PFN_cuCtxGetCurrent  PFN_cuCtxGetCurrent_v4000
#define PFN_cuCtxGetDevice  PFN_cuCtxGetDevice_v2000
#define PFN_cuCtxGetFlags  PFN_cuCtxGetFlags_v7000
#define PFN_cuCtxSynchronize  PFN_cuCtxSynchronize_v2000
#define PFN_cuCtxSetLimit  PFN_cuCtxSetLimit_v3010
#define PFN_cuCtxGetLimit  PFN_cuCtxGetLimit_v3010
#define PFN_cuCtxGetCacheConfig  PFN_cuCtxGetCacheConfig_v3020
#define PFN_cuCtxSetCacheConfig  PFN_cuCtxSetCacheConfig_v3020
#define PFN_cuCtxGetSharedMemConfig  PFN_cuCtxGetSharedMemConfig_v4020
#define PFN_cuCtxSetSharedMemConfig  PFN_cuCtxSetSharedMemConfig_v4020
#define PFN_cuCtxGetApiVersion  PFN_cuCtxGetApiVersion_v3020
#define PFN_cuCtxGetStreamPriorityRange  PFN_cuCtxGetStreamPriorityRange_v5050
#define PFN_cuCtxResetPersistingL2Cache  PFN_cuCtxResetPersistingL2Cache_v11000
#define PFN_cuCtxAttach  PFN_cuCtxAttach_v2000
#define PFN_cuCtxDetach  PFN_cuCtxDetach_v2000
#define PFN_cuCtxGetExecAffinity  PFN_cuCtxGetExecAffinity_v11040
#define PFN_cuModuleLoad  PFN_cuModuleLoad_v2000
#define PFN_cuModuleLoadData  PFN_cuModuleLoadData_v2000
#define PFN_cuModuleLoadDataEx  PFN_cuModuleLoadDataEx_v2010
#define PFN_cuModuleLoadFatBinary  PFN_cuModuleLoadFatBinary_v2000
#define PFN_cuModuleUnload  PFN_cuModuleUnload_v2000
#define PFN_cuModuleGetFunction  PFN_cuModuleGetFunction_v2000
#define PFN_cuModuleGetGlobal  PFN_cuModuleGetGlobal_v3020
#define PFN_cuModuleGetTexRef  PFN_cuModuleGetTexRef_v2000
#define PFN_cuModuleGetSurfRef  PFN_cuModuleGetSurfRef_v3000
#define PFN_cuLinkCreate  PFN_cuLinkCreate_v6050
#define PFN_cuLinkAddData  PFN_cuLinkAddData_v6050
#define PFN_cuLinkAddFile  PFN_cuLinkAddFile_v6050
#define PFN_cuLinkComplete  PFN_cuLinkComplete_v5050
#define PFN_cuLinkDestroy  PFN_cuLinkDestroy_v5050
#define PFN_cuMemGetInfo  PFN_cuMemGetInfo_v3020
#define PFN_cuMemAlloc  PFN_cuMemAlloc_v3020
#define PFN_cuMemAllocPitch  PFN_cuMemAllocPitch_v3020
#define PFN_cuMemFree  PFN_cuMemFree_v3020
#define PFN_cuMemGetAddressRange  PFN_cuMemGetAddressRange_v3020
#define PFN_cuMemAllocHost  PFN_cuMemAllocHost_v3020
#define PFN_cuMemFreeHost  PFN_cuMemFreeHost_v2000
#define PFN_cuMemHostAlloc  PFN_cuMemHostAlloc_v2020
#define PFN_cuMemHostGetDevicePointer  PFN_cuMemHostGetDevicePointer_v3020
#define PFN_cuMemHostGetFlags  PFN_cuMemHostGetFlags_v2030
#define PFN_cuMemAllocManaged  PFN_cuMemAllocManaged_v6000
#define PFN_cuDeviceGetByPCIBusId  PFN_cuDeviceGetByPCIBusId_v4010
#define PFN_cuDeviceGetPCIBusId  PFN_cuDeviceGetPCIBusId_v4010
#define PFN_cuIpcGetEventHandle  PFN_cuIpcGetEventHandle_v4010
#define PFN_cuIpcOpenEventHandle  PFN_cuIpcOpenEventHandle_v4010
#define PFN_cuIpcGetMemHandle  PFN_cuIpcGetMemHandle_v4010
#define PFN_cuIpcOpenMemHandle  PFN_cuIpcOpenMemHandle_v11000
#define PFN_cuIpcCloseMemHandle  PFN_cuIpcCloseMemHandle_v4010
#define PFN_cuMemHostRegister  PFN_cuMemHostRegister_v6050
#define PFN_cuMemHostUnregister  PFN_cuMemHostUnregister_v4000
#define PFN_cuMemcpy  __API_TYPEDEF_PTDS(PFN_cuMemcpy, 4000, 7000)
#define PFN_cuMemcpyPeer  __API_TYPEDEF_PTDS(PFN_cuMemcpyPeer, 4000, 7000)
#define PFN_cuMemcpyHtoD  __API_TYPEDEF_PTDS(PFN_cuMemcpyHtoD, 3020, 7000)
#define PFN_cuMemcpyDtoH  __API_TYPEDEF_PTDS(PFN_cuMemcpyDtoH, 3020, 7000)
#define PFN_cuMemcpyDtoD  __API_TYPEDEF_PTDS(PFN_cuMemcpyDtoD, 3020, 7000)
#define PFN_cuMemcpyDtoA  __API_TYPEDEF_PTDS(PFN_cuMemcpyDtoA, 3020, 7000)
#define PFN_cuMemcpyAtoD  __API_TYPEDEF_PTDS(PFN_cuMemcpyAtoD, 3020, 7000)
#define PFN_cuMemcpyHtoA  __API_TYPEDEF_PTDS(PFN_cuMemcpyHtoA, 3020, 7000)
#define PFN_cuMemcpyAtoH  __API_TYPEDEF_PTDS(PFN_cuMemcpyAtoH, 3020, 7000)
#define PFN_cuMemcpyAtoA  __API_TYPEDEF_PTDS(PFN_cuMemcpyAtoA, 3020, 7000)
#define PFN_cuMemcpy2D  __API_TYPEDEF_PTDS(PFN_cuMemcpy2D, 3020, 7000)
#define PFN_cuMemcpy2DUnaligned  __API_TYPEDEF_PTDS(PFN_cuMemcpy2DUnaligned, 3020, 7000)
#define PFN_cuMemcpy3D  __API_TYPEDEF_PTDS(PFN_cuMemcpy3D, 3020, 7000)
#define PFN_cuMemcpy3DPeer  __API_TYPEDEF_PTDS(PFN_cuMemcpy3DPeer, 4000, 7000)
#define PFN_cuMemcpyAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyAsync, 4000, 7000)
#define PFN_cuMemcpyPeerAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyPeerAsync, 4000, 7000)
#define PFN_cuMemcpyHtoDAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyHtoDAsync, 3020, 7000)
#define PFN_cuMemcpyDtoHAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyDtoHAsync, 3020, 7000)
#define PFN_cuMemcpyDtoDAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyDtoDAsync, 3020, 7000)
#define PFN_cuMemcpyHtoAAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyHtoAAsync, 3020, 7000)
#define PFN_cuMemcpyAtoHAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpyAtoHAsync, 3020, 7000)
#define PFN_cuMemcpy2DAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpy2DAsync, 3020, 7000)
#define PFN_cuMemcpy3DAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpy3DAsync, 3020, 7000)
#define PFN_cuMemcpy3DPeerAsync  __API_TYPEDEF_PTSZ(PFN_cuMemcpy3DPeerAsync, 4000, 7000)
#define PFN_cuMemsetD8  __API_TYPEDEF_PTDS(PFN_cuMemsetD8, 3020, 7000)
#define PFN_cuMemsetD16  __API_TYPEDEF_PTDS(PFN_cuMemsetD16, 3020, 7000)
#define PFN_cuMemsetD32  __API_TYPEDEF_PTDS(PFN_cuMemsetD32, 3020, 7000)
#define PFN_cuMemsetD2D8  __API_TYPEDEF_PTDS(PFN_cuMemsetD2D8, 3020, 7000)
#define PFN_cuMemsetD2D16  __API_TYPEDEF_PTDS(PFN_cuMemsetD2D16, 3020, 7000)
#define PFN_cuMemsetD2D32  __API_TYPEDEF_PTDS(PFN_cuMemsetD2D32, 3020, 7000)
#define PFN_cuMemsetD8Async  __API_TYPEDEF_PTSZ(PFN_cuMemsetD8Async, 3020, 7000)
#define PFN_cuMemsetD16Async  __API_TYPEDEF_PTSZ(PFN_cuMemsetD16Async, 3020, 7000)
#define PFN_cuMemsetD32Async  __API_TYPEDEF_PTSZ(PFN_cuMemsetD32Async, 3020, 7000)
#define PFN_cuMemsetD2D8Async  __API_TYPEDEF_PTSZ(PFN_cuMemsetD2D8Async, 3020, 7000)
#define PFN_cuMemsetD2D16Async  __API_TYPEDEF_PTSZ(PFN_cuMemsetD2D16Async, 3020, 7000)
#define PFN_cuMemsetD2D32Async  __API_TYPEDEF_PTSZ(PFN_cuMemsetD2D32Async, 3020, 7000)
#define PFN_cuArrayCreate  PFN_cuArrayCreate_v3020
#define PFN_cuArrayGetDescriptor  PFN_cuArrayGetDescriptor_v3020
#define PFN_cuArrayGetSparseProperties  PFN_cuArrayGetSparseProperties_v11010
#define PFN_cuMipmappedArrayGetSparseProperties  PFN_cuMipmappedArrayGetSparseProperties_v11010

#define PFN_cuArrayGetMemoryRequirements  PFN_cuArrayGetMemoryRequirements_v11060
#define PFN_cuMipmappedArrayGetMemoryRequirements  PFN_cuMipmappedArrayGetMemoryRequirements_v11060

#define PFN_cuArrayGetPlane  PFN_cuArrayGetPlane_v11020
#define PFN_cuArrayDestroy  PFN_cuArrayDestroy_v2000
#define PFN_cuArray3DCreate  PFN_cuArray3DCreate_v3020
#define PFN_cuArray3DGetDescriptor  PFN_cuArray3DGetDescriptor_v3020
#define PFN_cuMipmappedArrayCreate  PFN_cuMipmappedArrayCreate_v5000
#define PFN_cuMipmappedArrayGetLevel  PFN_cuMipmappedArrayGetLevel_v5000
#define PFN_cuMipmappedArrayDestroy  PFN_cuMipmappedArrayDestroy_v5000
#define PFN_cuMemAddressReserve  PFN_cuMemAddressReserve_v10020
#define PFN_cuMemAddressFree  PFN_cuMemAddressFree_v10020
#define PFN_cuMemCreate  PFN_cuMemCreate_v10020
#define PFN_cuMemRelease  PFN_cuMemRelease_v10020
#define PFN_cuMemMap  PFN_cuMemMap_v10020
#define PFN_cuMemMapArrayAsync  __API_TYPEDEF_PTSZ(PFN_cuMemMapArrayAsync, 11010, 11010)
#define PFN_cuMemUnmap  PFN_cuMemUnmap_v10020
#define PFN_cuMemSetAccess  PFN_cuMemSetAccess_v10020
#define PFN_cuMemGetAccess  PFN_cuMemGetAccess_v10020
#define PFN_cuMemExportToShareableHandle  PFN_cuMemExportToShareableHandle_v10020
#define PFN_cuMemImportFromShareableHandle  PFN_cuMemImportFromShareableHandle_v10020
#define PFN_cuMemGetAllocationGranularity  PFN_cuMemGetAllocationGranularity_v10020
#define PFN_cuMemGetAllocationPropertiesFromHandle  PFN_cuMemGetAllocationPropertiesFromHandle_v10020
#define PFN_cuMemRetainAllocationHandle  PFN_cuMemRetainAllocationHandle_v11000
#define PFN_cuMemFreeAsync  __API_TYPEDEF_PTSZ(PFN_cuMemFreeAsync, 11020, 11020)
#define PFN_cuMemAllocAsync  __API_TYPEDEF_PTSZ(PFN_cuMemAllocAsync, 11020, 11020)
#define PFN_cuMemPoolTrimTo  PFN_cuMemPoolTrimTo_v11020
#define PFN_cuMemPoolSetAttribute  PFN_cuMemPoolSetAttribute_v11020
#define PFN_cuMemPoolGetAttribute  PFN_cuMemPoolGetAttribute_v11020
#define PFN_cuMemPoolSetAccess  PFN_cuMemPoolSetAccess_v11020
#define PFN_cuMemPoolGetAccess  PFN_cuMemPoolGetAccess_v11020
#define PFN_cuMemPoolCreate  PFN_cuMemPoolCreate_v11020
#define PFN_cuMemPoolDestroy  PFN_cuMemPoolDestroy_v11020
#define PFN_cuMemAllocFromPoolAsync  __API_TYPEDEF_PTSZ(PFN_cuMemAllocFromPoolAsync, 11020, 11020)
#define PFN_cuMemPoolExportToShareableHandle  PFN_cuMemPoolExportToShareableHandle_v11020
#define PFN_cuMemPoolImportFromShareableHandle  PFN_cuMemPoolImportFromShareableHandle_v11020
#define PFN_cuMemPoolExportPointer  PFN_cuMemPoolExportPointer_v11020
#define PFN_cuMemPoolImportPointer  PFN_cuMemPoolImportPointer_v11020
#define PFN_cuPointerGetAttribute  PFN_cuPointerGetAttribute_v4000
#define PFN_cuMemPrefetchAsync  __API_TYPEDEF_PTSZ(PFN_cuMemPrefetchAsync, 8000, 8000)
#define PFN_cuMemAdvise  PFN_cuMemAdvise_v8000
#define PFN_cuMemRangeGetAttribute  PFN_cuMemRangeGetAttribute_v8000
#define PFN_cuMemRangeGetAttributes  PFN_cuMemRangeGetAttributes_v8000
#define PFN_cuPointerSetAttribute  PFN_cuPointerSetAttribute_v6000
#define PFN_cuPointerGetAttributes  PFN_cuPointerGetAttributes_v7000
#define PFN_cuStreamCreate  PFN_cuStreamCreate_v2000
#define PFN_cuStreamCreateWithPriority  PFN_cuStreamCreateWithPriority_v5050
#define PFN_cuStreamGetPriority  __API_TYPEDEF_PTSZ(PFN_cuStreamGetPriority, 5050, 7000)
#define PFN_cuStreamGetFlags  __API_TYPEDEF_PTSZ(PFN_cuStreamGetFlags, 5050, 7000)
#define PFN_cuStreamGetCtx  __API_TYPEDEF_PTSZ(PFN_cuStreamGetCtx, 9020, 9020)
#define PFN_cuStreamWaitEvent  __API_TYPEDEF_PTSZ(PFN_cuStreamWaitEvent, 3020, 7000)
#define PFN_cuStreamAddCallback  __API_TYPEDEF_PTSZ(PFN_cuStreamAddCallback, 5000, 7000)
#define PFN_cuStreamBeginCapture  __API_TYPEDEF_PTSZ(PFN_cuStreamBeginCapture, 10010, 10010)
#define PFN_cuThreadExchangeStreamCaptureMode  PFN_cuThreadExchangeStreamCaptureMode_v10010
#define PFN_cuStreamEndCapture  __API_TYPEDEF_PTSZ(PFN_cuStreamEndCapture, 10000, 10000)
#define PFN_cuStreamIsCapturing  __API_TYPEDEF_PTSZ(PFN_cuStreamIsCapturing, 10000, 10000)
#define PFN_cuStreamGetCaptureInfo  __API_TYPEDEF_PTSZ(PFN_cuStreamGetCaptureInfo, 10010, 10010)
#define PFN_cuStreamGetCaptureInfo_v2  __API_TYPEDEF_PTSZ(PFN_cuStreamGetCaptureInfo, 11030, 11030)
#define PFN_cuStreamUpdateCaptureDependencies  __API_TYPEDEF_PTSZ(PFN_cuStreamUpdateCaptureDependencies, 11030, 11030)
#define PFN_cuStreamAttachMemAsync  __API_TYPEDEF_PTSZ(PFN_cuStreamAttachMemAsync, 6000, 7000)
#define PFN_cuStreamQuery  __API_TYPEDEF_PTSZ(PFN_cuStreamQuery, 2000, 7000)
#define PFN_cuStreamSynchronize  __API_TYPEDEF_PTSZ(PFN_cuStreamSynchronize, 2000, 7000)
#define PFN_cuStreamDestroy  PFN_cuStreamDestroy_v4000
#define PFN_cuStreamCopyAttributes  __API_TYPEDEF_PTSZ(PFN_cuStreamCopyAttributes, 11000, 11000)
#define PFN_cuStreamGetAttribute  __API_TYPEDEF_PTSZ(PFN_cuStreamGetAttribute, 11000, 11000)
#define PFN_cuStreamSetAttribute  __API_TYPEDEF_PTSZ(PFN_cuStreamSetAttribute, 11000, 11000)
#define PFN_cuEventCreate  PFN_cuEventCreate_v2000
#define PFN_cuEventRecord  __API_TYPEDEF_PTSZ(PFN_cuEventRecord, 2000, 7000)
#define PFN_cuEventRecordWithFlags  __API_TYPEDEF_PTSZ(PFN_cuEventRecordWithFlags, 11010, 11010)
#define PFN_cuEventQuery  PFN_cuEventQuery_v2000
#define PFN_cuEventSynchronize  PFN_cuEventSynchronize_v2000
#define PFN_cuEventDestroy  PFN_cuEventDestroy_v4000
#define PFN_cuEventElapsedTime  PFN_cuEventElapsedTime_v2000
#define PFN_cuImportExternalMemory  PFN_cuImportExternalMemory_v10000
#define PFN_cuExternalMemoryGetMappedBuffer  PFN_cuExternalMemoryGetMappedBuffer_v10000
#define PFN_cuExternalMemoryGetMappedMipmappedArray  PFN_cuExternalMemoryGetMappedMipmappedArray_v10000
#define PFN_cuDestroyExternalMemory  PFN_cuDestroyExternalMemory_v10000
#define PFN_cuImportExternalSemaphore  PFN_cuImportExternalSemaphore_v10000
#define PFN_cuSignalExternalSemaphoresAsync  __API_TYPEDEF_PTSZ(PFN_cuSignalExternalSemaphoresAsync, 10000, 10000)
#define PFN_cuWaitExternalSemaphoresAsync  __API_TYPEDEF_PTSZ(PFN_cuWaitExternalSemaphoresAsync, 10000, 10000)
#define PFN_cuDestroyExternalSemaphore  PFN_cuDestroyExternalSemaphore_v10000
#define PFN_cuStreamWaitValue32  __API_TYPEDEF_PTSZ(PFN_cuStreamWaitValue32, 8000, 8000)
#define PFN_cuStreamWaitValue64  __API_TYPEDEF_PTSZ(PFN_cuStreamWaitValue64, 9000, 9000)
#define PFN_cuStreamWriteValue32  __API_TYPEDEF_PTSZ(PFN_cuStreamWriteValue32, 8000, 8000)
#define PFN_cuStreamWriteValue64  __API_TYPEDEF_PTSZ(PFN_cuStreamWriteValue64, 9000, 9000)
#define PFN_cuStreamBatchMemOp  __API_TYPEDEF_PTSZ(PFN_cuStreamBatchMemOp, 8000, 8000)
#define PFN_cuFuncGetAttribute  PFN_cuFuncGetAttribute_v2020
#define PFN_cuFuncSetAttribute  PFN_cuFuncSetAttribute_v9000
#define PFN_cuFuncSetCacheConfig  PFN_cuFuncSetCacheConfig_v3000
#define PFN_cuFuncSetSharedMemConfig  PFN_cuFuncSetSharedMemConfig_v4020
#define PFN_cuLaunchKernel  __API_TYPEDEF_PTSZ(PFN_cuLaunchKernel, 4000, 7000)



#define PFN_cuLaunchCooperativeKernel  __API_TYPEDEF_PTSZ(PFN_cuLaunchCooperativeKernel, 9000, 9000)
#define PFN_cuLaunchCooperativeKernelMultiDevice  PFN_cuLaunchCooperativeKernelMultiDevice_v9000
#define PFN_cuLaunchHostFunc  __API_TYPEDEF_PTSZ(PFN_cuLaunchHostFunc, 10000, 10000)
#define PFN_cuFuncSetBlockShape  PFN_cuFuncSetBlockShape_v2000
#define PFN_cuFuncSetSharedSize  PFN_cuFuncSetSharedSize_v2000
#define PFN_cuParamSetSize  PFN_cuParamSetSize_v2000
#define PFN_cuParamSeti  PFN_cuParamSeti_v2000
#define PFN_cuParamSetf  PFN_cuParamSetf_v2000
#define PFN_cuParamSetv  PFN_cuParamSetv_v2000
#define PFN_cuLaunch  PFN_cuLaunch_v2000
#define PFN_cuLaunchGrid  PFN_cuLaunchGrid_v2000
#define PFN_cuLaunchGridAsync  PFN_cuLaunchGridAsync_v2000
#define PFN_cuParamSetTexRef  PFN_cuParamSetTexRef_v2000
#define PFN_cuGraphCreate  PFN_cuGraphCreate_v10000
#define PFN_cuGraphAddKernelNode  PFN_cuGraphAddKernelNode_v10000
#define PFN_cuGraphKernelNodeGetParams  PFN_cuGraphKernelNodeGetParams_v10000
#define PFN_cuGraphKernelNodeSetParams  PFN_cuGraphKernelNodeSetParams_v10000
#define PFN_cuGraphAddMemcpyNode  PFN_cuGraphAddMemcpyNode_v10000
#define PFN_cuGraphMemcpyNodeGetParams  PFN_cuGraphMemcpyNodeGetParams_v10000
#define PFN_cuGraphMemcpyNodeSetParams  PFN_cuGraphMemcpyNodeSetParams_v10000
#define PFN_cuGraphAddMemsetNode  PFN_cuGraphAddMemsetNode_v10000
#define PFN_cuGraphMemsetNodeGetParams  PFN_cuGraphMemsetNodeGetParams_v10000
#define PFN_cuGraphMemsetNodeSetParams  PFN_cuGraphMemsetNodeSetParams_v10000
#define PFN_cuGraphAddHostNode  PFN_cuGraphAddHostNode_v10000
#define PFN_cuGraphHostNodeGetParams  PFN_cuGraphHostNodeGetParams_v10000
#define PFN_cuGraphHostNodeSetParams  PFN_cuGraphHostNodeSetParams_v10000
#define PFN_cuGraphAddChildGraphNode  PFN_cuGraphAddChildGraphNode_v10000
#define PFN_cuGraphChildGraphNodeGetGraph  PFN_cuGraphChildGraphNodeGetGraph_v10000
#define PFN_cuGraphAddEmptyNode  PFN_cuGraphAddEmptyNode_v10000
#define PFN_cuGraphAddEventRecordNode  PFN_cuGraphAddEventRecordNode_v11010
#define PFN_cuGraphEventRecordNodeGetEvent  PFN_cuGraphEventRecordNodeGetEvent_v11010
#define PFN_cuGraphEventRecordNodeSetEvent  PFN_cuGraphEventRecordNodeSetEvent_v11010
#define PFN_cuGraphAddEventWaitNode  PFN_cuGraphAddEventWaitNode_v11010
#define PFN_cuGraphEventWaitNodeGetEvent  PFN_cuGraphEventWaitNodeGetEvent_v11010
#define PFN_cuGraphEventWaitNodeSetEvent  PFN_cuGraphEventWaitNodeSetEvent_v11010
#define PFN_cuGraphAddExternalSemaphoresSignalNode  PFN_cuGraphAddExternalSemaphoresSignalNode_v11020
#define PFN_cuGraphExternalSemaphoresSignalNodeGetParams  PFN_cuGraphExternalSemaphoresSignalNodeGetParams_v11020
#define PFN_cuGraphExternalSemaphoresSignalNodeSetParams  PFN_cuGraphExternalSemaphoresSignalNodeSetParams_v11020
#define PFN_cuGraphAddExternalSemaphoresWaitNode  PFN_cuGraphAddExternalSemaphoresWaitNode_v11020
#define PFN_cuGraphExternalSemaphoresWaitNodeGetParams  PFN_cuGraphExternalSemaphoresWaitNodeGetParams_v11020
#define PFN_cuGraphExternalSemaphoresWaitNodeSetParams  PFN_cuGraphExternalSemaphoresWaitNodeSetParams_v11020
#define PFN_cuGraphClone  PFN_cuGraphClone_v10000
#define PFN_cuGraphNodeFindInClone  PFN_cuGraphNodeFindInClone_v10000
#define PFN_cuGraphNodeGetType  PFN_cuGraphNodeGetType_v10000
#define PFN_cuGraphGetNodes  PFN_cuGraphGetNodes_v10000
#define PFN_cuGraphGetRootNodes  PFN_cuGraphGetRootNodes_v10000
#define PFN_cuGraphGetEdges  PFN_cuGraphGetEdges_v10000
#define PFN_cuGraphNodeGetDependencies  PFN_cuGraphNodeGetDependencies_v10000
#define PFN_cuGraphNodeGetDependentNodes  PFN_cuGraphNodeGetDependentNodes_v10000
#define PFN_cuGraphAddDependencies  PFN_cuGraphAddDependencies_v10000
#define PFN_cuGraphRemoveDependencies  PFN_cuGraphRemoveDependencies_v10000
#define PFN_cuGraphDestroyNode  PFN_cuGraphDestroyNode_v10000
#define PFN_cuGraphInstantiate  PFN_cuGraphInstantiate_v11000
#define PFN_cuGraphInstantiateWithFlags  PFN_cuGraphInstantiateWithFlags_v11040
#define PFN_cuGraphExecKernelNodeSetParams  PFN_cuGraphExecKernelNodeSetParams_v10010
#define PFN_cuGraphExecMemcpyNodeSetParams  PFN_cuGraphExecMemcpyNodeSetParams_v10020
#define PFN_cuGraphExecMemsetNodeSetParams  PFN_cuGraphExecMemsetNodeSetParams_v10020
#define PFN_cuGraphExecHostNodeSetParams  PFN_cuGraphExecHostNodeSetParams_v10020
#define PFN_cuGraphExecChildGraphNodeSetParams  PFN_cuGraphExecChildGraphNodeSetParams_v11010
#define PFN_cuGraphExecEventRecordNodeSetEvent  PFN_cuGraphExecEventRecordNodeSetEvent_v11010
#define PFN_cuGraphExecEventWaitNodeSetEvent  PFN_cuGraphExecEventWaitNodeSetEvent_v11010
#define PFN_cuGraphExecExternalSemaphoresSignalNodeSetParams  PFN_cuGraphExecExternalSemaphoresSignalNodeSetParams_v11020
#define PFN_cuGraphExecExternalSemaphoresWaitNodeSetParams  PFN_cuGraphExecExternalSemaphoresWaitNodeSetParams_v11020
#define PFN_cuGraphUpload  __API_TYPEDEF_PTSZ(PFN_cuGraphUpload, 11010, 11010)
#define PFN_cuGraphLaunch  __API_TYPEDEF_PTSZ(PFN_cuGraphLaunch, 10000, 10000)
#define PFN_cuGraphExecDestroy  PFN_cuGraphExecDestroy_v10000
#define PFN_cuGraphDestroy  PFN_cuGraphDestroy_v10000
#define PFN_cuGraphExecUpdate  PFN_cuGraphExecUpdate_v10020
#define PFN_cuGraphKernelNodeCopyAttributes  PFN_cuGraphKernelNodeCopyAttributes_v11000
#define PFN_cuGraphKernelNodeGetAttribute  PFN_cuGraphKernelNodeGetAttribute_v11000
#define PFN_cuGraphKernelNodeSetAttribute  PFN_cuGraphKernelNodeSetAttribute_v11000
#define PFN_cuGraphDebugDotPrint  PFN_cuGraphDebugDotPrint_v11030
#define PFN_cuGraphAddMemAllocNode  PFN_cuGraphAddMemAllocNode_v11040
#define PFN_cuGraphMemAllocNodeGetParams PFN_cuGraphMemAllocNodeGetParams_v11040
#define PFN_cuGraphAddMemFreeNode  PFN_cuGraphAddMemFreeNode_v11040
#define PFN_cuGraphMemFreeNodeGetParams PFN_cuGraphMemFreeNodeGetParams_v11040
#define PFN_cuGraphNodeSetEnabled PFN_cuGraphNodeSetEnabled_v11060
#define PFN_cuGraphNodeGetEnabled PFN_cuGraphNodeGetEnabled_v11060
#define PFN_cuDeviceGraphMemTrim  PFN_cuDeviceGraphMemTrim_v11040
#define PFN_cuDeviceGetGraphMemAttribute  PFN_cuDeviceGetGraphMemAttribute_v11040
#define PFN_cuDeviceSetGraphMemAttribute  PFN_cuDeviceSetGraphMemAttribute_v11040
#define PFN_cuOccupancyMaxActiveBlocksPerMultiprocessor  PFN_cuOccupancyMaxActiveBlocksPerMultiprocessor_v6050
#define PFN_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  PFN_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000
#define PFN_cuOccupancyMaxPotentialBlockSize  PFN_cuOccupancyMaxPotentialBlockSize_v6050
#define PFN_cuOccupancyMaxPotentialBlockSizeWithFlags  PFN_cuOccupancyMaxPotentialBlockSizeWithFlags_v7000
#define PFN_cuOccupancyAvailableDynamicSMemPerBlock  PFN_cuOccupancyAvailableDynamicSMemPerBlock_v10020
#define PFN_cuTexRefSetArray  PFN_cuTexRefSetArray_v2000
#define PFN_cuTexRefSetMipmappedArray  PFN_cuTexRefSetMipmappedArray_v5000
#define PFN_cuTexRefSetAddress  PFN_cuTexRefSetAddress_v3020
#define PFN_cuTexRefSetAddress2D  PFN_cuTexRefSetAddress2D_v4010
#define PFN_cuTexRefSetFormat  PFN_cuTexRefSetFormat_v2000
#define PFN_cuTexRefSetAddressMode  PFN_cuTexRefSetAddressMode_v2000
#define PFN_cuTexRefSetFilterMode  PFN_cuTexRefSetFilterMode_v2000
#define PFN_cuTexRefSetMipmapFilterMode  PFN_cuTexRefSetMipmapFilterMode_v5000
#define PFN_cuTexRefSetMipmapLevelBias  PFN_cuTexRefSetMipmapLevelBias_v5000
#define PFN_cuTexRefSetMipmapLevelClamp  PFN_cuTexRefSetMipmapLevelClamp_v5000
#define PFN_cuTexRefSetMaxAnisotropy  PFN_cuTexRefSetMaxAnisotropy_v5000
#define PFN_cuTexRefSetBorderColor  PFN_cuTexRefSetBorderColor_v8000
#define PFN_cuTexRefSetFlags  PFN_cuTexRefSetFlags_v2000
#define PFN_cuTexRefGetAddress  PFN_cuTexRefGetAddress_v3020
#define PFN_cuTexRefGetArray  PFN_cuTexRefGetArray_v2000
#define PFN_cuTexRefGetMipmappedArray  PFN_cuTexRefGetMipmappedArray_v5000
#define PFN_cuTexRefGetAddressMode  PFN_cuTexRefGetAddressMode_v2000
#define PFN_cuTexRefGetFilterMode  PFN_cuTexRefGetFilterMode_v2000
#define PFN_cuTexRefGetFormat  PFN_cuTexRefGetFormat_v2000
#define PFN_cuTexRefGetMipmapFilterMode  PFN_cuTexRefGetMipmapFilterMode_v5000
#define PFN_cuTexRefGetMipmapLevelBias  PFN_cuTexRefGetMipmapLevelBias_v5000
#define PFN_cuTexRefGetMipmapLevelClamp  PFN_cuTexRefGetMipmapLevelClamp_v5000
#define PFN_cuTexRefGetMaxAnisotropy  PFN_cuTexRefGetMaxAnisotropy_v5000
#define PFN_cuTexRefGetBorderColor  PFN_cuTexRefGetBorderColor_v8000
#define PFN_cuTexRefGetFlags  PFN_cuTexRefGetFlags_v2000
#define PFN_cuTexRefCreate  PFN_cuTexRefCreate_v2000
#define PFN_cuTexRefDestroy  PFN_cuTexRefDestroy_v2000
#define PFN_cuSurfRefSetArray  PFN_cuSurfRefSetArray_v3000
#define PFN_cuSurfRefGetArray  PFN_cuSurfRefGetArray_v3000
#define PFN_cuTexObjectCreate  PFN_cuTexObjectCreate_v5000
#define PFN_cuTexObjectDestroy  PFN_cuTexObjectDestroy_v5000
#define PFN_cuTexObjectGetResourceDesc  PFN_cuTexObjectGetResourceDesc_v5000
#define PFN_cuTexObjectGetTextureDesc  PFN_cuTexObjectGetTextureDesc_v5000
#define PFN_cuTexObjectGetResourceViewDesc  PFN_cuTexObjectGetResourceViewDesc_v5000
#define PFN_cuSurfObjectCreate  PFN_cuSurfObjectCreate_v5000
#define PFN_cuSurfObjectDestroy  PFN_cuSurfObjectDestroy_v5000
#define PFN_cuSurfObjectGetResourceDesc  PFN_cuSurfObjectGetResourceDesc_v5000
#define PFN_cuDeviceCanAccessPeer  PFN_cuDeviceCanAccessPeer_v4000
#define PFN_cuCtxEnablePeerAccess  PFN_cuCtxEnablePeerAccess_v4000
#define PFN_cuCtxDisablePeerAccess  PFN_cuCtxDisablePeerAccess_v4000
#define PFN_cuDeviceGetP2PAttribute  PFN_cuDeviceGetP2PAttribute_v8000
#define PFN_cuGraphicsUnregisterResource  PFN_cuGraphicsUnregisterResource_v3000
#define PFN_cuGraphicsSubResourceGetMappedArray  PFN_cuGraphicsSubResourceGetMappedArray_v3000
#define PFN_cuGraphicsResourceGetMappedMipmappedArray  PFN_cuGraphicsResourceGetMappedMipmappedArray_v5000
#define PFN_cuGraphicsResourceGetMappedPointer  PFN_cuGraphicsResourceGetMappedPointer_v3020
#define PFN_cuGraphicsResourceSetMapFlags  PFN_cuGraphicsResourceSetMapFlags_v6050
#define PFN_cuGraphicsMapResources  __API_TYPEDEF_PTSZ(PFN_cuGraphicsMapResources, 3000, 7000)
#define PFN_cuGraphicsUnmapResources  __API_TYPEDEF_PTSZ(PFN_cuGraphicsUnmapResources, 3000, 7000)
#define PFN_cuGetExportTable  PFN_cuGetExportTable_v3000
#define PFN_cuFuncGetModule  PFN_cuFuncGetModule_v11000
#define PFN_cuFlushGPUDirectRDMAWrites PFN_cuFlushGPUDirectRDMAWrites_v11030
#define PFN_cuGetProcAddress  PFN_cuGetProcAddress_v11030
#define PFN_cuUserObjectCreate  PFN_cuUserObjectCreate_v11030
#define PFN_cuUserObjectRetain  PFN_cuUserObjectRetain_v11030
#define PFN_cuUserObjectRelease  PFN_cuUserObjectRelease_v11030
#define PFN_cuGraphRetainUserObject  PFN_cuGraphRetainUserObject_v11030
#define PFN_cuGraphReleaseUserObject  PFN_cuGraphReleaseUserObject_v11030


/*
 * Type definitions for functions defined in cuda.h
 */
typedef CUresult (CUDAAPI *PFN_cuGetErrorString_v6000)(CUresult error, const char **pStr);
typedef CUresult (CUDAAPI *PFN_cuGetErrorName_v6000)(CUresult error, const char **pStr);
typedef CUresult (CUDAAPI *PFN_cuInit_v2000)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuDriverGetVersion_v2020)(int *driverVersion);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet_v2000)(CUdevice_v1 *device, int ordinal);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetCount_v2000)(int *count);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName_v2000)(char *name, int len, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetLuid_v10000)(char *luid, unsigned int *deviceNodeMask, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceTotalMem_v3020)(size_t *bytes, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetTexture1DLinearMaxWidth_v11010)(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetAttribute_v2000)(int *pi, CUdevice_attribute attrib, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetNvSciSyncAttributes_v10020)(void *nvSciSyncAttrList, CUdevice_v1 dev, int flags);
typedef CUresult (CUDAAPI *PFN_cuDeviceSetMemPool_v11020)(CUdevice_v1 dev, CUmemoryPool pool);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetMemPool_v11020)(CUmemoryPool *pool, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetDefaultMemPool_v11020)(CUmemoryPool *pool_out, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetProperties_v2000)(CUdevprop_v1 *prop, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceComputeCapability_v2000)(int *major, int *minor, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRetain_v7000)(CUcontext *pctx, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRelease_v11000)(CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxSetFlags_v11000)(CUdevice_v1 dev, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxGetState_v7000)(CUdevice_v1 dev, unsigned int *flags, int *active);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxReset_v11000)(CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetExecAffinitySupport_v11040)(int *pi, CUexecAffinityType type, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuCtxCreate_v3020)(CUcontext *pctx, unsigned int flags, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuCtxCreate_v11040)(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuCtxDestroy_v4000)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxPushCurrent_v4000)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxPopCurrent_v4000)(CUcontext *pctx);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCurrent_v4000)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxGetCurrent_v4000)(CUcontext *pctx);
typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice_v2000)(CUdevice_v1 *device);
typedef CUresult (CUDAAPI *PFN_cuCtxGetFlags_v7000)(unsigned int *flags);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize_v2000)(void);
typedef CUresult (CUDAAPI *PFN_cuCtxSetLimit_v3010)(CUlimit limit, size_t value);
typedef CUresult (CUDAAPI *PFN_cuCtxGetLimit_v3010)(size_t *pvalue, CUlimit limit);
typedef CUresult (CUDAAPI *PFN_cuCtxGetCacheConfig_v3020)(CUfunc_cache *pconfig);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCacheConfig_v3020)(CUfunc_cache config);
typedef CUresult (CUDAAPI *PFN_cuCtxGetSharedMemConfig_v4020)(CUsharedconfig *pConfig);
typedef CUresult (CUDAAPI *PFN_cuCtxSetSharedMemConfig_v4020)(CUsharedconfig config);
typedef CUresult (CUDAAPI *PFN_cuCtxGetApiVersion_v3020)(CUcontext ctx, unsigned int *version);
typedef CUresult (CUDAAPI *PFN_cuCtxGetStreamPriorityRange_v5050)(int *leastPriority, int *greatestPriority);
typedef CUresult (CUDAAPI *PFN_cuCtxResetPersistingL2Cache_v11000)(void);
typedef CUresult (CUDAAPI *PFN_cuCtxAttach_v2000)(CUcontext *pctx, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuCtxDetach_v2000)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxGetExecAffinity_v11040)(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type);
typedef CUresult (CUDAAPI *PFN_cuModuleLoad_v2000)(CUmodule *module, const char *fname);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadData_v2000)(CUmodule *module, const void *image);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadDataEx_v2010)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadFatBinary_v2000)(CUmodule *module, const void *fatCubin);
typedef CUresult (CUDAAPI *PFN_cuModuleUnload_v2000)(CUmodule hmod);
typedef CUresult (CUDAAPI *PFN_cuModuleGetFunction_v2000)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (CUDAAPI *PFN_cuModuleGetGlobal_v3020)(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule hmod, const char *name);
typedef CUresult (CUDAAPI *PFN_cuModuleGetTexRef_v2000)(CUtexref *pTexRef, CUmodule hmod, const char *name);
typedef CUresult (CUDAAPI *PFN_cuModuleGetSurfRef_v3000)(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
typedef CUresult (CUDAAPI *PFN_cuLinkCreate_v6050)(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
typedef CUresult (CUDAAPI *PFN_cuLinkAddData_v6050)(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (CUDAAPI *PFN_cuLinkAddFile_v6050)(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (CUDAAPI *PFN_cuLinkComplete_v5050)(CUlinkState state, void **cubinOut, size_t *sizeOut);
typedef CUresult (CUDAAPI *PFN_cuLinkDestroy_v5050)(CUlinkState state);
typedef CUresult (CUDAAPI *PFN_cuMemGetInfo_v3020)(size_t *free, size_t *total);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemAllocPitch_v3020)(CUdeviceptr_v2 *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
typedef CUresult (CUDAAPI *PFN_cuMemFree_v3020)(CUdeviceptr_v2 dptr);
typedef CUresult (CUDAAPI *PFN_cuMemGetAddressRange_v3020)(CUdeviceptr_v2 *pbase, size_t *psize, CUdeviceptr_v2 dptr);
typedef CUresult (CUDAAPI *PFN_cuMemAllocHost_v3020)(void **pp, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemFreeHost_v2000)(void *p);
typedef CUresult (CUDAAPI *PFN_cuMemHostAlloc_v2020)(void **pp, size_t bytesize, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuMemHostGetDevicePointer_v3020)(CUdeviceptr_v2 *pdptr, void *p, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuMemHostGetFlags_v2030)(unsigned int *pFlags, void *p);
typedef CUresult (CUDAAPI *PFN_cuMemAllocManaged_v6000)(CUdeviceptr_v2 *dptr, size_t bytesize, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetByPCIBusId_v4010)(CUdevice_v1 *dev, const char *pciBusId);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetPCIBusId_v4010)(char *pciBusId, int len, CUdevice_v1 dev);
typedef CUresult (CUDAAPI *PFN_cuIpcGetEventHandle_v4010)(CUipcEventHandle_v1 *pHandle, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuIpcOpenEventHandle_v4010)(CUevent *phEvent, CUipcEventHandle_v1 handle);
typedef CUresult (CUDAAPI *PFN_cuIpcGetMemHandle_v4010)(CUipcMemHandle_v1 *pHandle, CUdeviceptr_v2 dptr);
typedef CUresult (CUDAAPI *PFN_cuIpcOpenMemHandle_v11000)(CUdeviceptr_v2 *pdptr, CUipcMemHandle_v1 handle, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuIpcCloseMemHandle_v4010)(CUdeviceptr_v2 dptr);
typedef CUresult (CUDAAPI *PFN_cuMemHostRegister_v6050)(void *p, size_t bytesize, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuMemHostUnregister_v4000)(void *p);
typedef CUresult (CUDAAPI *PFN_cuMemcpy_v7000_ptds)(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyPeer_v7000_ptds)(CUdeviceptr_v2 dstDevice, CUcontext dstContext, CUdeviceptr_v2 srcDevice, CUcontext srcContext, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoD_v7000_ptds)(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH_v7000_ptds)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoD_v7000_ptds)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoA_v7000_ptds)(CUarray dstArray, size_t dstOffset, CUdeviceptr_v2 srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoD_v7000_ptds)(CUdeviceptr_v2 dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoA_v7000_ptds)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoH_v7000_ptds)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoA_v7000_ptds)(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2D_v7000_ptds)(const CUDA_MEMCPY2D_v2 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2DUnaligned_v7000_ptds)(const CUDA_MEMCPY2D_v2 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3D_v7000_ptds)(const CUDA_MEMCPY3D_v2 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DPeer_v7000_ptds)(const CUDA_MEMCPY3D_PEER_v1 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAsync_v7000_ptsz)(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyPeerAsync_v7000_ptsz)(CUdeviceptr_v2 dstDevice, CUcontext dstContext, CUdeviceptr_v2 srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoDAsync_v7000_ptsz)(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoHAsync_v7000_ptsz)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoDAsync_v7000_ptsz)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoAAsync_v7000_ptsz)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoHAsync_v7000_ptsz)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2DAsync_v7000_ptsz)(const CUDA_MEMCPY2D_v2 *pCopy, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DAsync_v7000_ptsz)(const CUDA_MEMCPY3D_v2 *pCopy, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DPeerAsync_v7000_ptsz)(const CUDA_MEMCPY3D_PEER_v1 *pCopy, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD8_v7000_ptds)(CUdeviceptr_v2 dstDevice, unsigned char uc, size_t N);
typedef CUresult (CUDAAPI *PFN_cuMemsetD16_v7000_ptds)(CUdeviceptr_v2 dstDevice, unsigned short us, size_t N);
typedef CUresult (CUDAAPI *PFN_cuMemsetD32_v7000_ptds)(CUdeviceptr_v2 dstDevice, unsigned int ui, size_t N);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D8_v7000_ptds)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D16_v7000_ptds)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D32_v7000_ptds)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
typedef CUresult (CUDAAPI *PFN_cuMemsetD8Async_v7000_ptsz)(CUdeviceptr_v2 dstDevice, unsigned char uc, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD16Async_v7000_ptsz)(CUdeviceptr_v2 dstDevice, unsigned short us, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD32Async_v7000_ptsz)(CUdeviceptr_v2 dstDevice, unsigned int ui, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D8Async_v7000_ptsz)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D16Async_v7000_ptsz)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D32Async_v7000_ptsz)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuArrayCreate_v3020)(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR_v2 *pAllocateArray);
typedef CUresult (CUDAAPI *PFN_cuArrayGetDescriptor_v3020)(CUDA_ARRAY_DESCRIPTOR_v2 *pArrayDescriptor, CUarray hArray);
typedef CUresult (CUDAAPI *PFN_cuArrayGetSparseProperties_v11010)(CUDA_ARRAY_SPARSE_PROPERTIES_v1 *sparseProperties, CUarray array);
typedef CUresult (CUDAAPI *PFN_cuMipmappedArrayGetSparseProperties_v11010)(CUDA_ARRAY_SPARSE_PROPERTIES_v1 *sparseProperties, CUmipmappedArray mipmap);

typedef CUresult (CUDAAPI *PFN_cuArrayGetMemoryRequirements_v11060)(CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 *memoryRequirements, CUarray array, CUdevice device);
typedef CUresult (CUDAAPI *PFN_cuMipmappedArrayGetMemoryRequirements_v11060)(CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 *memoryRequirements, CUmipmappedArray mipmap, CUdevice device);

typedef CUresult (CUDAAPI *PFN_cuArrayGetPlane_v11020)(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx);
typedef CUresult (CUDAAPI *PFN_cuArrayDestroy_v2000)(CUarray hArray);
typedef CUresult (CUDAAPI *PFN_cuArray3DCreate_v3020)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR_v2 *pAllocateArray);
typedef CUresult (CUDAAPI *PFN_cuArray3DGetDescriptor_v3020)(CUDA_ARRAY3D_DESCRIPTOR_v2 *pArrayDescriptor, CUarray hArray);
typedef CUresult (CUDAAPI *PFN_cuMipmappedArrayCreate_v5000)(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR_v2 *pMipmappedArrayDesc, unsigned int numMipmapLevels);
typedef CUresult (CUDAAPI *PFN_cuMipmappedArrayGetLevel_v5000)(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
typedef CUresult (CUDAAPI *PFN_cuMipmappedArrayDestroy_v5000)(CUmipmappedArray hMipmappedArray);
typedef CUresult (CUDAAPI *PFN_cuMemAddressReserve_v10020)(CUdeviceptr_v2 *ptr, size_t size, size_t alignment, CUdeviceptr_v2 addr, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemAddressFree_v10020)(CUdeviceptr_v2 ptr, size_t size);
typedef CUresult (CUDAAPI *PFN_cuMemCreate_v10020)(CUmemGenericAllocationHandle_v1 *handle, size_t size, const CUmemAllocationProp_v1 *prop, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemRelease_v10020)(CUmemGenericAllocationHandle_v1 handle);
typedef CUresult (CUDAAPI *PFN_cuMemMap_v10020)(CUdeviceptr_v2 ptr, size_t size, size_t offset, CUmemGenericAllocationHandle_v1 handle, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemMapArrayAsync_v11010_ptsz)(CUarrayMapInfo_v1 *mapInfoList, unsigned int count, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemUnmap_v10020)(CUdeviceptr_v2 ptr, size_t size);
typedef CUresult (CUDAAPI *PFN_cuMemSetAccess_v10020)(CUdeviceptr_v2 ptr, size_t size, const CUmemAccessDesc_v1 *desc, size_t count);
typedef CUresult (CUDAAPI *PFN_cuMemGetAccess_v10020)(unsigned long long *flags, const CUmemLocation_v1 *location, CUdeviceptr_v2 ptr);
typedef CUresult (CUDAAPI *PFN_cuMemExportToShareableHandle_v10020)(void *shareableHandle, CUmemGenericAllocationHandle_v1 handle, CUmemAllocationHandleType handleType, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemImportFromShareableHandle_v10020)(CUmemGenericAllocationHandle_v1 *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
typedef CUresult (CUDAAPI *PFN_cuMemGetAllocationGranularity_v10020)(size_t *granularity, const CUmemAllocationProp_v1 *prop, CUmemAllocationGranularity_flags option);
typedef CUresult (CUDAAPI *PFN_cuMemGetAllocationPropertiesFromHandle_v10020)(CUmemAllocationProp_v1 *prop, CUmemGenericAllocationHandle_v1 handle);
typedef CUresult (CUDAAPI *PFN_cuMemRetainAllocationHandle_v11000)(CUmemGenericAllocationHandle_v1 *handle, void *addr);
typedef CUresult (CUDAAPI *PFN_cuMemFreeAsync_v11020_ptsz)(CUdeviceptr_v2 dptr, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemAllocAsync_v11020_ptsz)(CUdeviceptr_v2 *dptr, size_t bytesize, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemPoolTrimTo_v11020)(CUmemoryPool pool, size_t minBytesToKeep);
typedef CUresult (CUDAAPI *PFN_cuMemPoolSetAttribute_v11020)(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
typedef CUresult (CUDAAPI *PFN_cuMemPoolGetAttribute_v11020)(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
typedef CUresult (CUDAAPI *PFN_cuMemPoolSetAccess_v11020)(CUmemoryPool pool, const CUmemAccessDesc_v1 *map, size_t count);
typedef CUresult (CUDAAPI *PFN_cuMemPoolGetAccess_v11020)(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation_v1 *location);
typedef CUresult (CUDAAPI *PFN_cuMemPoolCreate_v11020)(CUmemoryPool *pool, const CUmemPoolProps_v1 *poolProps);
typedef CUresult (CUDAAPI *PFN_cuMemPoolDestroy_v11020)(CUmemoryPool pool);
typedef CUresult (CUDAAPI *PFN_cuMemAllocFromPoolAsync_v11020_ptsz)(CUdeviceptr_v2 *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemPoolExportToShareableHandle_v11020)(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemPoolImportFromShareableHandle_v11020)(CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuMemPoolExportPointer_v11020)(CUmemPoolPtrExportData_v1 *shareData_out, CUdeviceptr_v2 ptr);
typedef CUresult (CUDAAPI *PFN_cuMemPoolImportPointer_v11020)(CUdeviceptr_v2 *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData_v1 *shareData);
typedef CUresult (CUDAAPI *PFN_cuPointerGetAttribute_v4000)(void *data, CUpointer_attribute attribute, CUdeviceptr_v2 ptr);
typedef CUresult (CUDAAPI *PFN_cuMemPrefetchAsync_v8000_ptsz)(CUdeviceptr_v2 devPtr, size_t count, CUdevice_v1 dstDevice, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemAdvise_v8000)(CUdeviceptr_v2 devPtr, size_t count, CUmem_advise advice, CUdevice_v1 device);
typedef CUresult (CUDAAPI *PFN_cuMemRangeGetAttribute_v8000)(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr_v2 devPtr, size_t count);
typedef CUresult (CUDAAPI *PFN_cuMemRangeGetAttributes_v8000)(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr_v2 devPtr, size_t count);
typedef CUresult (CUDAAPI *PFN_cuPointerSetAttribute_v6000)(const void *value, CUpointer_attribute attribute, CUdeviceptr_v2 ptr);
typedef CUresult (CUDAAPI *PFN_cuPointerGetAttributes_v7000)(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr_v2 ptr);
typedef CUresult (CUDAAPI *PFN_cuStreamCreate_v2000)(CUstream *phStream, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuStreamCreateWithPriority_v5050)(CUstream *phStream, unsigned int flags, int priority);
typedef CUresult (CUDAAPI *PFN_cuStreamGetPriority_v7000_ptsz)(CUstream hStream, int *priority);
typedef CUresult (CUDAAPI *PFN_cuStreamGetFlags_v7000_ptsz)(CUstream hStream, unsigned int *flags);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCtx_v9020_ptsz)(CUstream hStream, CUcontext *pctx);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitEvent_v7000_ptsz)(CUstream hStream, CUevent hEvent, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuStreamAddCallback_v7000_ptsz)(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010_ptsz)(CUstream hStream, CUstreamCaptureMode mode);
typedef CUresult (CUDAAPI *PFN_cuThreadExchangeStreamCaptureMode_v10010)(CUstreamCaptureMode *mode);
typedef CUresult (CUDAAPI *PFN_cuStreamEndCapture_v10000_ptsz)(CUstream hStream, CUgraph *phGraph);
typedef CUresult (CUDAAPI *PFN_cuStreamIsCapturing_v10000_ptsz)(CUstream hStream, CUstreamCaptureStatus *captureStatus);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCaptureInfo_v10010_ptsz)(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCaptureInfo_v11030_ptsz)(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
typedef CUresult (CUDAAPI *PFN_cuStreamUpdateCaptureDependencies_v11030_ptsz)(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamAttachMemAsync_v7000_ptsz)(CUstream hStream, CUdeviceptr_v2 dptr, size_t length, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamQuery_v7000_ptsz)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamSynchronize_v7000_ptsz)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamDestroy_v4000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamCopyAttributes_v11000_ptsz)(CUstream dst, CUstream src);
typedef CUresult (CUDAAPI *PFN_cuStreamGetAttribute_v11000_ptsz)(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue_v1 *value_out);
typedef CUresult (CUDAAPI *PFN_cuStreamSetAttribute_v11000_ptsz)(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue_v1 *value);
typedef CUresult (CUDAAPI *PFN_cuEventCreate_v2000)(CUevent *phEvent, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuEventRecord_v7000_ptsz)(CUevent hEvent, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuEventRecordWithFlags_v11010_ptsz)(CUevent hEvent, CUstream hStream, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuEventQuery_v2000)(CUevent hEvent);
typedef CUresult (CUDAAPI *PFN_cuEventSynchronize_v2000)(CUevent hEvent);
typedef CUresult (CUDAAPI *PFN_cuEventDestroy_v4000)(CUevent hEvent);
typedef CUresult (CUDAAPI *PFN_cuEventElapsedTime_v2000)(float *pMilliseconds, CUevent hStart, CUevent hEnd);
typedef CUresult (CUDAAPI *PFN_cuImportExternalMemory_v10000)(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 *memHandleDesc);
typedef CUresult (CUDAAPI *PFN_cuExternalMemoryGetMappedBuffer_v10000)(CUdeviceptr_v2 *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 *bufferDesc);
typedef CUresult (CUDAAPI *PFN_cuExternalMemoryGetMappedMipmappedArray_v10000)(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 *mipmapDesc);
typedef CUresult (CUDAAPI *PFN_cuDestroyExternalMemory_v10000)(CUexternalMemory extMem);
typedef CUresult (CUDAAPI *PFN_cuImportExternalSemaphore_v10000)(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 *semHandleDesc);
typedef CUresult (CUDAAPI *PFN_cuSignalExternalSemaphoresAsync_v10000_ptsz)(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 *paramsArray, unsigned int numExtSems, CUstream stream);
typedef CUresult (CUDAAPI *PFN_cuWaitExternalSemaphoresAsync_v10000_ptsz)(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 *paramsArray, unsigned int numExtSems, CUstream stream);
typedef CUresult (CUDAAPI *PFN_cuDestroyExternalSemaphore_v10000)(CUexternalSemaphore extSem);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitValue32_v8000_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitValue64_v9000_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamWriteValue32_v8000_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamWriteValue64_v9000_ptsz)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamBatchMemOp_v8000_ptsz)(CUstream stream, unsigned int count, CUstreamBatchMemOpParams_v1 *paramArray, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuFuncGetAttribute_v2020)(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
typedef CUresult (CUDAAPI *PFN_cuFuncSetAttribute_v9000)(CUfunction hfunc, CUfunction_attribute attrib, int value);
typedef CUresult (CUDAAPI *PFN_cuFuncSetCacheConfig_v3000)(CUfunction hfunc, CUfunc_cache config);
typedef CUresult (CUDAAPI *PFN_cuFuncSetSharedMemConfig_v4020)(CUfunction hfunc, CUsharedconfig config);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel_v7000_ptsz)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);



typedef CUresult (CUDAAPI *PFN_cuLaunchCooperativeKernel_v9000_ptsz)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
typedef CUresult (CUDAAPI *PFN_cuLaunchCooperativeKernelMultiDevice_v9000)(CUDA_LAUNCH_PARAMS_v1 *launchParamsList, unsigned int numDevices, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuLaunchHostFunc_v10000_ptsz)(CUstream hStream, CUhostFn fn, void *userData);
typedef CUresult (CUDAAPI *PFN_cuFuncSetBlockShape_v2000)(CUfunction hfunc, int x, int y, int z);
typedef CUresult (CUDAAPI *PFN_cuFuncSetSharedSize_v2000)(CUfunction hfunc, unsigned int bytes);
typedef CUresult (CUDAAPI *PFN_cuParamSetSize_v2000)(CUfunction hfunc, unsigned int numbytes);
typedef CUresult (CUDAAPI *PFN_cuParamSeti_v2000)(CUfunction hfunc, int offset, unsigned int value);
typedef CUresult (CUDAAPI *PFN_cuParamSetf_v2000)(CUfunction hfunc, int offset, float value);
typedef CUresult (CUDAAPI *PFN_cuParamSetv_v2000)(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
typedef CUresult (CUDAAPI *PFN_cuLaunch_v2000)(CUfunction f);
typedef CUresult (CUDAAPI *PFN_cuLaunchGrid_v2000)(CUfunction f, int grid_width, int grid_height);
typedef CUresult (CUDAAPI *PFN_cuLaunchGridAsync_v2000)(CUfunction f, int grid_width, int grid_height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuParamSetTexRef_v2000)(CUfunction hfunc, int texunit, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuGraphCreate_v10000)(CUgraph *phGraph, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGraphAddKernelNode_v10000)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphKernelNodeGetParams_v10000)(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphKernelNodeSetParams_v10000)(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphAddMemcpyNode_v10000)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D_v2 *copyParams, CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuGraphMemcpyNodeGetParams_v10000)(CUgraphNode hNode, CUDA_MEMCPY3D_v2 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphMemcpyNodeSetParams_v10000)(CUgraphNode hNode, const CUDA_MEMCPY3D_v2 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphAddMemsetNode_v10000)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS_v1 *memsetParams, CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuGraphMemsetNodeGetParams_v10000)(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphMemsetNodeSetParams_v10000)(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphAddHostNode_v10000)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphHostNodeGetParams_v10000)(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphHostNodeSetParams_v10000)(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphAddChildGraphNode_v10000)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph);
typedef CUresult (CUDAAPI *PFN_cuGraphChildGraphNodeGetGraph_v10000)(CUgraphNode hNode, CUgraph *phGraph);
typedef CUresult (CUDAAPI *PFN_cuGraphAddEmptyNode_v10000)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies);
typedef CUresult (CUDAAPI *PFN_cuGraphAddEventRecordNode_v11010)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuGraphEventRecordNodeGetEvent_v11010)(CUgraphNode hNode, CUevent *event_out);
typedef CUresult (CUDAAPI *PFN_cuGraphEventRecordNodeSetEvent_v11010)(CUgraphNode hNode, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuGraphAddEventWaitNode_v11010)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuGraphEventWaitNodeGetEvent_v11010)(CUgraphNode hNode, CUevent *event_out);
typedef CUresult (CUDAAPI *PFN_cuGraphEventWaitNodeSetEvent_v11010)(CUgraphNode hNode, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuGraphAddExternalSemaphoresSignalNode_v11020)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphExternalSemaphoresSignalNodeGetParams_v11020)(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *params_out);
typedef CUresult (CUDAAPI *PFN_cuGraphExternalSemaphoresSignalNodeSetParams_v11020)(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphAddExternalSemaphoresWaitNode_v11020)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphExternalSemaphoresWaitNodeGetParams_v11020)(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *params_out);
typedef CUresult (CUDAAPI *PFN_cuGraphExternalSemaphoresWaitNodeSetParams_v11020)(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphClone_v10000)(CUgraph *phGraphClone, CUgraph originalGraph);
typedef CUresult (CUDAAPI *PFN_cuGraphNodeFindInClone_v10000)(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
typedef CUresult (CUDAAPI *PFN_cuGraphNodeGetType_v10000)(CUgraphNode hNode, CUgraphNodeType *type);
typedef CUresult (CUDAAPI *PFN_cuGraphGetNodes_v10000)(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
typedef CUresult (CUDAAPI *PFN_cuGraphGetRootNodes_v10000)(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes);
typedef CUresult (CUDAAPI *PFN_cuGraphGetEdges_v10000)(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges);
typedef CUresult (CUDAAPI *PFN_cuGraphNodeGetDependencies_v10000)(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies);
typedef CUresult (CUDAAPI *PFN_cuGraphNodeGetDependentNodes_v10000)(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes);
typedef CUresult (CUDAAPI *PFN_cuGraphAddDependencies_v10000)(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
typedef CUresult (CUDAAPI *PFN_cuGraphRemoveDependencies_v10000)(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
typedef CUresult (CUDAAPI *PFN_cuGraphDestroyNode_v10000)(CUgraphNode hNode);
typedef CUresult (CUDAAPI *PFN_cuGraphInstantiate_v11000)(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
typedef CUresult (CUDAAPI *PFN_cuGraphInstantiateWithFlags_v11040)(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
typedef CUresult (CUDAAPI *PFN_cuGraphExecKernelNodeSetParams_v10010)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphExecMemcpyNodeSetParams_v10020)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D_v2 *copyParams, CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuGraphExecMemsetNodeSetParams_v10020)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS_v1 *memsetParams, CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuGraphExecHostNodeSetParams_v10020)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphExecChildGraphNodeSetParams_v11010)(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);
typedef CUresult (CUDAAPI *PFN_cuGraphExecEventRecordNodeSetEvent_v11010)(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuGraphExecEventWaitNodeSetEvent_v11010)(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
typedef CUresult (CUDAAPI *PFN_cuGraphExecExternalSemaphoresSignalNodeSetParams_v11020)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphExecExternalSemaphoresWaitNodeSetParams_v11020)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphUpload_v11010_ptsz)(CUgraphExec hGraphExec, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGraphLaunch_v10000_ptsz)(CUgraphExec hGraphExec, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGraphExecDestroy_v10000)(CUgraphExec hGraphExec);
typedef CUresult (CUDAAPI *PFN_cuGraphDestroy_v10000)(CUgraph hGraph);
typedef CUresult (CUDAAPI *PFN_cuGraphExecUpdate_v10020)(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out);
typedef CUresult (CUDAAPI *PFN_cuGraphKernelNodeCopyAttributes_v11000)(CUgraphNode dst, CUgraphNode src);
typedef CUresult (CUDAAPI *PFN_cuGraphKernelNodeGetAttribute_v11000)(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue_v1 *value_out);
typedef CUresult (CUDAAPI *PFN_cuGraphKernelNodeSetAttribute_v11000)(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue_v1 *value);
typedef CUresult (CUDAAPI *PFN_cuGraphDebugDotPrint_v11030)(CUgraph hGraph, const char *path, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGraphAddMemAllocNode_v11040)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams);
typedef CUresult (CUDAAPI *PFN_cuGraphMemAllocNodeGetParams_v11040)(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out);
typedef CUresult (CUDAAPI *PFN_cuGraphAddMemFreeNode_v11040)(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr);
typedef CUresult (CUDAAPI *PFN_cuGraphMemFreeNodeGetParams_v11040)(CUgraphNode hNode, CUdeviceptr *dptr_out);
typedef CUresult (CUDAAPI *PFN_cuGraphNodeSetEnabled_v11060)(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled);
typedef CUresult (CUDAAPI *PFN_cuGraphNodeGetEnabled_v11060)(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int *isEnabled);
typedef CUresult (CUDAAPI *PFN_cuDeviceGraphMemTrim_v11040)(CUdevice device);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetGraphMemAttribute_v11040)(CUdevice device, CUgraphMem_attribute attr, void* value);
typedef CUresult (CUDAAPI *PFN_cuDeviceSetGraphMemAttribute_v11040)(CUdevice device, CUgraphMem_attribute attr, void* value);
typedef CUresult (CUDAAPI *PFN_cuOccupancyMaxActiveBlocksPerMultiprocessor_v6050)(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
typedef CUresult (CUDAAPI *PFN_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000)(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuOccupancyMaxPotentialBlockSize_v6050)(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
typedef CUresult (CUDAAPI *PFN_cuOccupancyMaxPotentialBlockSizeWithFlags_v7000)(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuOccupancyAvailableDynamicSMemPerBlock_v10020)(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetArray_v2000)(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetMipmappedArray_v5000)(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetAddress_v3020)(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr_v2 dptr, size_t bytes);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetAddress2D_v4010)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR_v2 *desc, CUdeviceptr_v2 dptr, size_t Pitch);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetFormat_v2000)(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetAddressMode_v2000)(CUtexref hTexRef, int dim, CUaddress_mode am);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetFilterMode_v2000)(CUtexref hTexRef, CUfilter_mode fm);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetMipmapFilterMode_v5000)(CUtexref hTexRef, CUfilter_mode fm);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetMipmapLevelBias_v5000)(CUtexref hTexRef, float bias);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetMipmapLevelClamp_v5000)(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetMaxAnisotropy_v5000)(CUtexref hTexRef, unsigned int maxAniso);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetBorderColor_v8000)(CUtexref hTexRef, float *pBorderColor);
typedef CUresult (CUDAAPI *PFN_cuTexRefSetFlags_v2000)(CUtexref hTexRef, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetAddress_v3020)(CUdeviceptr_v2 *pdptr, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetArray_v2000)(CUarray *phArray, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetMipmappedArray_v5000)(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetAddressMode_v2000)(CUaddress_mode *pam, CUtexref hTexRef, int dim);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetFilterMode_v2000)(CUfilter_mode *pfm, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetFormat_v2000)(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetMipmapFilterMode_v5000)(CUfilter_mode *pfm, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetMipmapLevelBias_v5000)(float *pbias, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetMipmapLevelClamp_v5000)(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetMaxAnisotropy_v5000)(int *pmaxAniso, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetBorderColor_v8000)(float *pBorderColor, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefGetFlags_v2000)(unsigned int *pFlags, CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefCreate_v2000)(CUtexref *pTexRef);
typedef CUresult (CUDAAPI *PFN_cuTexRefDestroy_v2000)(CUtexref hTexRef);
typedef CUresult (CUDAAPI *PFN_cuSurfRefSetArray_v3000)(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuSurfRefGetArray_v3000)(CUarray *phArray, CUsurfref hSurfRef);
typedef CUresult (CUDAAPI *PFN_cuTexObjectCreate_v5000)(CUtexObject_v1 *pTexObject, const CUDA_RESOURCE_DESC_v1 *pResDesc, const CUDA_TEXTURE_DESC_v1 *pTexDesc, const CUDA_RESOURCE_VIEW_DESC_v1 *pResViewDesc);
typedef CUresult (CUDAAPI *PFN_cuTexObjectDestroy_v5000)(CUtexObject_v1 texObject);
typedef CUresult (CUDAAPI *PFN_cuTexObjectGetResourceDesc_v5000)(CUDA_RESOURCE_DESC_v1 *pResDesc, CUtexObject_v1 texObject);
typedef CUresult (CUDAAPI *PFN_cuTexObjectGetTextureDesc_v5000)(CUDA_TEXTURE_DESC_v1 *pTexDesc, CUtexObject_v1 texObject);
typedef CUresult (CUDAAPI *PFN_cuTexObjectGetResourceViewDesc_v5000)(CUDA_RESOURCE_VIEW_DESC_v1 *pResViewDesc, CUtexObject_v1 texObject);
typedef CUresult (CUDAAPI *PFN_cuSurfObjectCreate_v5000)(CUsurfObject_v1 *pSurfObject, const CUDA_RESOURCE_DESC_v1 *pResDesc);
typedef CUresult (CUDAAPI *PFN_cuSurfObjectDestroy_v5000)(CUsurfObject_v1 surfObject);
typedef CUresult (CUDAAPI *PFN_cuSurfObjectGetResourceDesc_v5000)(CUDA_RESOURCE_DESC_v1 *pResDesc, CUsurfObject_v1 surfObject);
typedef CUresult (CUDAAPI *PFN_cuDeviceCanAccessPeer_v4000)(int *canAccessPeer, CUdevice_v1 dev, CUdevice_v1 peerDev);
typedef CUresult (CUDAAPI *PFN_cuCtxEnablePeerAccess_v4000)(CUcontext peerContext, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuCtxDisablePeerAccess_v4000)(CUcontext peerContext);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetP2PAttribute_v8000)(int *value, CUdevice_P2PAttribute attrib, CUdevice_v1 srcDevice, CUdevice_v1 dstDevice);
typedef CUresult (CUDAAPI *PFN_cuGraphicsUnregisterResource_v3000)(CUgraphicsResource resource);
typedef CUresult (CUDAAPI *PFN_cuGraphicsSubResourceGetMappedArray_v3000)(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
typedef CUresult (CUDAAPI *PFN_cuGraphicsResourceGetMappedMipmappedArray_v5000)(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource);
typedef CUresult (CUDAAPI *PFN_cuGraphicsResourceGetMappedPointer_v3020)(CUdeviceptr_v2 *pDevPtr, size_t *pSize, CUgraphicsResource resource);
typedef CUresult (CUDAAPI *PFN_cuGraphicsResourceSetMapFlags_v6050)(CUgraphicsResource resource, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGraphicsMapResources_v7000_ptsz)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGraphicsUnmapResources_v7000_ptsz)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGetExportTable_v3000)(const void **ppExportTable, const CUuuid *pExportTableId);
typedef CUresult (CUDAAPI *PFN_cuFuncGetModule_v11000)(CUmodule *hmod, CUfunction hfunc);
typedef CUresult (CUDAAPI *PFN_cuGetProcAddress_v11030)(const char *symbol, void **pfn, int driverVersion, cuuint64_t flags);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoD_v3020)(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH_v3020)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoD_v3020)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoA_v3020)(CUarray dstArray, size_t dstOffset, CUdeviceptr_v2 srcDevice, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoD_v3020)(CUdeviceptr_v2 dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoA_v3020)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoH_v3020)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoA_v3020)(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoAAsync_v3020)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoHAsync_v3020)(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2D_v3020)(const CUDA_MEMCPY2D_v2 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2DUnaligned_v3020)(const CUDA_MEMCPY2D_v2 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3D_v3020)(const CUDA_MEMCPY3D_v2 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoDAsync_v3020)(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoHAsync_v3020)(void *dstHost, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoDAsync_v3020)(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2DAsync_v3020)(const CUDA_MEMCPY2D_v2 *pCopy, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DAsync_v3020)(const CUDA_MEMCPY3D_v2 *pCopy, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD8_v3020)(CUdeviceptr_v2 dstDevice, unsigned char uc, size_t N);
typedef CUresult (CUDAAPI *PFN_cuMemsetD16_v3020)(CUdeviceptr_v2 dstDevice, unsigned short us, size_t N);
typedef CUresult (CUDAAPI *PFN_cuMemsetD32_v3020)(CUdeviceptr_v2 dstDevice, unsigned int ui, size_t N);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D8_v3020)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D16_v3020)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D32_v3020)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
typedef CUresult (CUDAAPI *PFN_cuMemcpy_v4000)(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyAsync_v4000)(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpyPeer_v4000)(CUdeviceptr_v2 dstDevice, CUcontext dstContext, CUdeviceptr_v2 srcDevice, CUcontext srcContext, size_t ByteCount);
typedef CUresult (CUDAAPI *PFN_cuMemcpyPeerAsync_v4000)(CUdeviceptr_v2 dstDevice, CUcontext dstContext, CUdeviceptr_v2 srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DPeer_v4000)(const CUDA_MEMCPY3D_PEER_v1 *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DPeerAsync_v4000)(const CUDA_MEMCPY3D_PEER_v1 *pCopy, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD8Async_v3020)(CUdeviceptr_v2 dstDevice, unsigned char uc, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD16Async_v3020)(CUdeviceptr_v2 dstDevice, unsigned short us, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD32Async_v3020)(CUdeviceptr_v2 dstDevice, unsigned int ui, size_t N, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D8Async_v3020)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D16Async_v3020)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemsetD2D32Async_v3020)(CUdeviceptr_v2 dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamGetPriority_v5050)(CUstream hStream, int *priority);
typedef CUresult (CUDAAPI *PFN_cuStreamGetFlags_v5050)(CUstream hStream, unsigned int *flags);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCtx_v9020)(CUstream hStream, CUcontext *pctx);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitEvent_v3020)(CUstream hStream, CUevent hEvent, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuStreamAddCallback_v5000)(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamAttachMemAsync_v6000)(CUstream hStream, CUdeviceptr_v2 dptr, size_t length, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamQuery_v2000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamSynchronize_v2000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuEventRecord_v2000)(CUevent hEvent, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuEventRecordWithFlags_v11010)(CUevent hEvent, CUstream hStream, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel_v4000)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);



typedef CUresult (CUDAAPI *PFN_cuLaunchHostFunc_v10000)(CUstream hStream, CUhostFn fn, void *userData);
typedef CUresult (CUDAAPI *PFN_cuGraphicsMapResources_v3000)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGraphicsUnmapResources_v3000)(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamWriteValue32_v8000)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitValue32_v8000)(CUstream stream, CUdeviceptr_v2 addr, cuuint32_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamWriteValue64_v9000)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamWaitValue64_v9000)(CUstream stream, CUdeviceptr_v2 addr, cuuint64_t value, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuStreamBatchMemOp_v8000)(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuMemPrefetchAsync_v8000)(CUdeviceptr_v2 devPtr, size_t count, CUdevice_v1 dstDevice, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuLaunchCooperativeKernel_v9000)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
typedef CUresult (CUDAAPI *PFN_cuSignalExternalSemaphoresAsync_v10000)(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 *paramsArray, unsigned int numExtSems, CUstream stream);
typedef CUresult (CUDAAPI *PFN_cuWaitExternalSemaphoresAsync_v10000)(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 *paramsArray, unsigned int numExtSems, CUstream stream);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);
typedef CUresult (CUDAAPI *PFN_cuStreamEndCapture_v10000)(CUstream hStream, CUgraph *phGraph);
typedef CUresult (CUDAAPI *PFN_cuStreamIsCapturing_v10000)(CUstream hStream, CUstreamCaptureStatus *captureStatus);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCaptureInfo_v10010)(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
typedef CUresult (CUDAAPI *PFN_cuStreamGetCaptureInfo_v11030)(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
typedef CUresult (CUDAAPI *PFN_cuStreamUpdateCaptureDependencies_v11030)(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGraphUpload_v11010)(CUgraphExec hGraph, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuGraphLaunch_v10000)(CUgraphExec hGraph, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamCopyAttributes_v11000)(CUstream dstStream, CUstream srcStream);
typedef CUresult (CUDAAPI *PFN_cuStreamGetAttribute_v11000)(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue_v1 *value);
typedef CUresult (CUDAAPI *PFN_cuStreamSetAttribute_v11000)(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue_v1 *param);
typedef CUresult (CUDAAPI *PFN_cuMemMapArrayAsync_v11010)(CUarrayMapInfo_v1 *mapInfoList, unsigned int count, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemFreeAsync_v11020)(CUdeviceptr_v2 dptr, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemAllocAsync_v11020)(CUdeviceptr_v2 *dptr, size_t bytesize, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuMemAllocFromPoolAsync_v11020)(CUdeviceptr_v2 *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuFlushGPUDirectRDMAWrites_v11030)(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope);
typedef CUresult (CUDAAPI *PFN_cuUserObjectCreate_v11030)(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuUserObjectRetain_v11030)(CUuserObject object, unsigned int count);
typedef CUresult (CUDAAPI *PFN_cuUserObjectRelease_v11030)(CUuserObject object, unsigned int count);
typedef CUresult (CUDAAPI *PFN_cuGraphRetainUserObject_v11030)(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
typedef CUresult (CUDAAPI *PFN_cuGraphReleaseUserObject_v11030)(CUgraph graph, CUuserObject object, unsigned int count);

/*
 * Type definitions for older versioned functions in cuda.h
 */
#if defined(__CUDA_API_VERSION_INTERNAL)
    typedef CUresult (CUDAAPI *PFN_cuMemHostRegister_v4000)(void *p, size_t bytesize, unsigned int Flags);
    typedef CUresult (CUDAAPI *PFN_cuGraphicsResourceSetMapFlags_v3000)(CUgraphicsResource resource, unsigned int flags);
    typedef CUresult (CUDAAPI *PFN_cuLinkCreate_v5050)(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
    typedef CUresult (CUDAAPI *PFN_cuLinkAddData_v5050)(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);
    typedef CUresult (CUDAAPI *PFN_cuLinkAddFile_v5050)(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues);
    typedef CUresult (CUDAAPI *PFN_cuTexRefSetAddress2D_v3020)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR_v2 *desc, CUdeviceptr_v2 dptr, size_t Pitch);
    typedef CUresult (CUDAAPI *PFN_cuDeviceTotalMem_v2000)(unsigned int *bytes, CUdevice_v1 dev);
    typedef CUresult (CUDAAPI *PFN_cuCtxCreate_v2000)(CUcontext *pctx, unsigned int flags, CUdevice_v1 dev);
    typedef CUresult (CUDAAPI *PFN_cuModuleGetGlobal_v2000)(CUdeviceptr_v1 *dptr, unsigned int *bytes, CUmodule hmod, const char *name);
    typedef CUresult (CUDAAPI *PFN_cuMemGetInfo_v2000)(unsigned int *free, unsigned int *total);
    typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);
    typedef CUresult (CUDAAPI *PFN_cuMemAllocPitch_v2000)(CUdeviceptr_v1 *dptr, unsigned int *pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes);
    typedef CUresult (CUDAAPI *PFN_cuMemFree_v2000)(CUdeviceptr_v1 dptr);
    typedef CUresult (CUDAAPI *PFN_cuMemGetAddressRange_v2000)(CUdeviceptr_v1 *pbase, unsigned int *psize, CUdeviceptr_v1 dptr);
    typedef CUresult (CUDAAPI *PFN_cuMemAllocHost_v2000)(void **pp, unsigned int bytesize);
    typedef CUresult (CUDAAPI *PFN_cuMemHostGetDevicePointer_v2020)(CUdeviceptr_v1 *pdptr, void *p, unsigned int Flags);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoD_v2000)(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH_v2000)(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoD_v2000)(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoA_v2000)(CUarray dstArray, unsigned int dstOffset, CUdeviceptr_v1 srcDevice, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoD_v2000)(CUdeviceptr_v1 dstDevice, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoA_v2000)(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoH_v2000)(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoA_v2000)(CUarray dstArray, unsigned int dstOffset, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoAAsync_v2000)(CUarray dstArray, unsigned int dstOffset, const void *srcHost, unsigned int ByteCount, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyAtoHAsync_v2000)(void *dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemcpy2D_v2000)(const CUDA_MEMCPY2D_v1 *pCopy);
    typedef CUresult (CUDAAPI *PFN_cuMemcpy2DUnaligned_v2000)(const CUDA_MEMCPY2D_v1 *pCopy);
    typedef CUresult (CUDAAPI *PFN_cuMemcpy3D_v2000)(const CUDA_MEMCPY3D_v1 *pCopy);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoDAsync_v2000)(CUdeviceptr_v1 dstDevice, const void *srcHost, unsigned int ByteCount, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoHAsync_v2000)(void *dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoDAsync_v3000)(CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemcpy2DAsync_v2000)(const CUDA_MEMCPY2D_v1 *pCopy, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemcpy3DAsync_v2000)(const CUDA_MEMCPY3D_v1 *pCopy, CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuMemsetD8_v2000)(CUdeviceptr_v1 dstDevice, unsigned char uc, unsigned int N);
    typedef CUresult (CUDAAPI *PFN_cuMemsetD16_v2000)(CUdeviceptr_v1 dstDevice, unsigned short us, unsigned int N);
    typedef CUresult (CUDAAPI *PFN_cuMemsetD32_v2000)(CUdeviceptr_v1 dstDevice, unsigned int ui, unsigned int N);
    typedef CUresult (CUDAAPI *PFN_cuMemsetD2D8_v2000)(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned char uc, unsigned int Width, unsigned int Height);
    typedef CUresult (CUDAAPI *PFN_cuMemsetD2D16_v2000)(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned short us, unsigned int Width, unsigned int Height);
    typedef CUresult (CUDAAPI *PFN_cuMemsetD2D32_v2000)(CUdeviceptr_v1 dstDevice, unsigned int dstPitch, unsigned int ui, unsigned int Width, unsigned int Height);
    typedef CUresult (CUDAAPI *PFN_cuArrayCreate_v2000)(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR_v1 *pAllocateArray);
    typedef CUresult (CUDAAPI *PFN_cuArrayGetDescriptor_v2000)(CUDA_ARRAY_DESCRIPTOR_v1 *pArrayDescriptor, CUarray hArray);
    typedef CUresult (CUDAAPI *PFN_cuArray3DCreate_v2000)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR_v1 *pAllocateArray);
    typedef CUresult (CUDAAPI *PFN_cuArray3DGetDescriptor_v2000)(CUDA_ARRAY3D_DESCRIPTOR_v1 *pArrayDescriptor, CUarray hArray);
    typedef CUresult (CUDAAPI *PFN_cuTexRefSetAddress_v2000)(unsigned int *ByteOffset, CUtexref hTexRef, CUdeviceptr_v1 dptr, unsigned int bytes);
    typedef CUresult (CUDAAPI *PFN_cuTexRefSetAddress2D_v2020)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR_v1 *desc, CUdeviceptr_v1 dptr, unsigned int Pitch);
    typedef CUresult (CUDAAPI *PFN_cuTexRefGetAddress_v2000)(CUdeviceptr_v1 *pdptr, CUtexref hTexRef);
    typedef CUresult (CUDAAPI *PFN_cuGraphicsResourceGetMappedPointer_v3000)(CUdeviceptr_v1 *pDevPtr, unsigned int *pSize, CUgraphicsResource resource);
    typedef CUresult (CUDAAPI *PFN_cuCtxDestroy_v2000)(CUcontext ctx);
    typedef CUresult (CUDAAPI *PFN_cuCtxPopCurrent_v2000)(CUcontext *pctx);
    typedef CUresult (CUDAAPI *PFN_cuCtxPushCurrent_v2000)(CUcontext ctx);
    typedef CUresult (CUDAAPI *PFN_cuStreamDestroy_v2000)(CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuEventDestroy_v2000)(CUevent hEvent);
    typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRelease_v7000)(CUdevice_v1 dev);
    typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxReset_v7000)(CUdevice_v1 dev);
    typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxSetFlags_v7000)(CUdevice_v1 dev, unsigned int flags);
    typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000_ptsz)(CUstream hStream);
    typedef CUresult (CUDAAPI *PFN_cuIpcOpenMemHandle_v4010)(CUdeviceptr_v2 *pdptr, CUipcMemHandle_v1 handle, unsigned int Flags);
    typedef CUresult (CUDAAPI *PFN_cuGraphInstantiate_v10000)(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
#endif

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
