
#ifndef _MUSA_COMPATIBLE_CUH
#define _MUSA_COMPATIBLE_CUH


#define CUresult MUresult
#define CUdevice MUdevice
#define CUdeviceptr MUdeviceptr

#define cudaDataType_t musaDataType_t
#define cudaError_t musaError_t
#define cudaEvent_t musaEvent_t
#define cudaStream_t musaStream_t
#define cudaDeviceProp musaDeviceProp

#define cublasStatus_t mublasStatus_t
#define cublasHandle_t mublasHandle_t
#define cublasComputeType_t musaDataType_t // reserved in musa

#define cuGetErrorString muGetErrorString
#define cuDeviceGet muDeviceGet
#define cuDeviceGetAttribute muDeviceGetAttribute
// #define cuMemGetAllocationGranularity muMemGetAllocationGranularity // so far, not implemeted
// #define CUmemAllocationProp MUmemAllocationProp

#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaMemGetInfo musaMemGetInfo
#define cudaMemset musaMemset
#define cudaMalloc musaMalloc
#define cudaMallocHost musaMallocHost
#define cudaFree musaFree
#define cudaFreeHost musaFreeHost
#define cudaHostUnregister musaHostUnregister
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpy2DAsync musaMemcpy2DAsync
#define cudaMemcpyPeerAsync musaMemcpyPeerAsync
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#define cudaGetDevice musaGetDevice
#define cudaGetDeviceCount musaGetDeviceCount
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaSetDevice musaSetDevice
#define cudaEventRecord musaEventRecord
#define cudaEventDestroy musaEventDestroy
#define cudaEventCreate musaEventCreate
#define cudaEventSynchronize musaEventSynchronize
#define cudaEventDisableTiming musaEventDisableTiming
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#define cudaStreamPerThread musaStreamPerThread
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#define cudaStreamNonBlocking musaStreamNonBlocking
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamWaitEvent musaStreamWaitEvent

#define cublasCreate mublasCreate
#define cublasDestroy mublasDestroy
#define cublasSetMathMode mublasSetMathMode
#define cublasSetStream mublasSetStream
#define cublasGemmEx mublasGemmEx
#define cublasSgemm mublasSgemm
#ifdef mublasGemmStridedBatchedEx
#undef mublasGemmStridedBatchedEx
#endif // mublasGemmStridedBatchedEx
#define cublasGemmStridedBatchedEx( \
    handle, \
    transA, \
    transB, \
    m, \
    n, \
    k, \
    alpha, \
    A, \
    Atype, \
    lda, \
    strideA, \
    B, \
    Btype, \
    ldb, \
    strideB, \
    beta, \
    C, \
    Ctype, \
    ldc, \
    strideC, \
    batchCount, \
    computeType, \
    algo \
) \
mublasGemmStridedBatchedEx( \
    handle, \
    transA, \
    transB, \
    m, \
    n, \
    k, \
    alpha, \
    A, \
    Atype, \
    lda, \
    strideA, \
    B, \
    Btype, \
    ldb, \
    strideB, \
    beta, \
    C, \
    Ctype, \
    ldc, \
    strideC, \
    C /* D */, \
    Ctype, \
    ldc, \
    strideC, \
    batchCount, \
    computeType, \
    algo, \
    0 /* solution type, reserved */, \
    0 /* flags */ \
)

#define cublasGemmBatchedEx( \
    handle, \
    transA, \
    transB, \
    m, \
    n, \
    k, \
    alpha, \
    A, \
    Atype, \
    lda, \
    B, \
    Btype, \
    ldb, \
    beta, \
    C, \
    Ctype, \
    ldc, \
    batchCount, \
    computeType, \
    algo \
) \
mublasGemmBatchedEx( \
    handle, \
    transA, \
    transB, \
    m, \
    n, \
    k, \
    alpha, \
    A, \
    Atype, \
    lda, \
    B, \
    Btype, \
    ldb, \
    beta, \
    C, \
    Ctype, \
    ldc, \
    C /* D */, \
    Ctype, \
    ldc, \
    batchCount, \
    computeType, \
    algo, \
    0 /* solution type, reserved */, \
    0 /* flags */ \
)

#define CUDART_VERSION MUSART_VERSION

#define CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
// #define CU_MEM_ALLOCATION_TYPE_PINNED MU_MEM_ALLOCATION_TYPE_PINNED
// #define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED MU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED

#define CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED MUBLAS_STATUS_NOT_IMPLEMENTED
#define CUBLAS_STATUS_ALLOC_FAILED MUBLAS_STATUS_NOT_IMPLEMENTED
#define CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_MATH_MODE_TP32_TENSOR // ???
#define CUBLAS_OP_T MUBLAS_OP_T
#define CUBLAS_OP_N MUBLAS_OP_N
#define CUBLAS_COMPUTE_16F MUSA_R_16F // reserved in musa
#define CUBLAS_COMPUTE_32F MUSA_R_32F // reserved in musa
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT_TENSOR_OP

#define CUDA_SUCCESS MUSA_SUCCESS
#define CUDA_R_16F MUSA_R_16F
#define CUDA_R_32F MUSA_R_32F
#define cudaSuccess musaSuccess
#define cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled musaErrorPeerAccessNotEnabled

#endif // _MUSA_COMPATIBLE_CUH