
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the plain C interface to the CLBlast BLAS routines, the counter-part of the
// normal 'clblast.h' C++ header.
//
// =================================================================================================

#ifndef CLBLAST_CLBLAST_C_H_
#define CLBLAST_CLBLAST_C_H_

// Includes the normal OpenCL C header
#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
#else
  #include <CL/opencl.h>
#endif

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

// Version numbering (v1.5.3)
#define CLBLAST_VERSION_MAJOR 1
#define CLBLAST_VERSION_MINOR 5
#define CLBLAST_VERSION_PATCH 3

// The C interface
#ifdef __cplusplus
extern "C" {
#endif

// =================================================================================================

// Status codes. These codes can be returned by functions declared in this header file. The error
// codes match either the standard OpenCL error codes or the clBLAS error codes. 
typedef enum CLBlastStatusCode_ {

  // Status codes in common with the OpenCL standard
  CLBlastSuccess                   =   0, // CL_SUCCESS
  CLBlastOpenCLCompilerNotAvailable=  -3, // CL_COMPILER_NOT_AVAILABLE
  CLBlastTempBufferAllocFailure    =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
  CLBlastOpenCLOutOfResources      =  -5, // CL_OUT_OF_RESOURCES
  CLBlastOpenCLOutOfHostMemory     =  -6, // CL_OUT_OF_HOST_MEMORY
  CLBlastOpenCLBuildProgramFailure = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
  CLBlastInvalidValue              = -30, // CL_INVALID_VALUE
  CLBlastInvalidCommandQueue       = -36, // CL_INVALID_COMMAND_QUEUE
  CLBlastInvalidMemObject          = -38, // CL_INVALID_MEM_OBJECT
  CLBlastInvalidBinary             = -42, // CL_INVALID_BINARY
  CLBlastInvalidBuildOptions       = -43, // CL_INVALID_BUILD_OPTIONS
  CLBlastInvalidProgram            = -44, // CL_INVALID_PROGRAM
  CLBlastInvalidProgramExecutable  = -45, // CL_INVALID_PROGRAM_EXECUTABLE
  CLBlastInvalidKernelName         = -46, // CL_INVALID_KERNEL_NAME
  CLBlastInvalidKernelDefinition   = -47, // CL_INVALID_KERNEL_DEFINITION
  CLBlastInvalidKernel             = -48, // CL_INVALID_KERNEL
  CLBlastInvalidArgIndex           = -49, // CL_INVALID_ARG_INDEX
  CLBlastInvalidArgValue           = -50, // CL_INVALID_ARG_VALUE
  CLBlastInvalidArgSize            = -51, // CL_INVALID_ARG_SIZE
  CLBlastInvalidKernelArgs         = -52, // CL_INVALID_KERNEL_ARGS
  CLBlastInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
  CLBlastInvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
  CLBlastInvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
  CLBlastInvalidGlobalOffset       = -56, // CL_INVALID_GLOBAL_OFFSET
  CLBlastInvalidEventWaitList      = -57, // CL_INVALID_EVENT_WAIT_LIST
  CLBlastInvalidEvent              = -58, // CL_INVALID_EVENT
  CLBlastInvalidOperation          = -59, // CL_INVALID_OPERATION
  CLBlastInvalidBufferSize         = -61, // CL_INVALID_BUFFER_SIZE
  CLBlastInvalidGlobalWorkSize     = -63, // CL_INVALID_GLOBAL_WORK_SIZE

  // Status codes in common with the clBLAS library
  CLBlastNotImplemented            = -1024, // Routine or functionality not implemented yet
  CLBlastInvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
  CLBlastInvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
  CLBlastInvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
  CLBlastInvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
  CLBlastInvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
  CLBlastInvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
  CLBlastInvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
  CLBlastInvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
  CLBlastInvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
  CLBlastInvalidIncrementX         = -1013, // Increment of vector X cannot be zero
  CLBlastInvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
  CLBlastInsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
  CLBlastInsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
  CLBlastInsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
  CLBlastInsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
  CLBlastInsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

  // Custom additional status codes for CLBlast
  CLBlastInsufficientMemoryTemp    = -2050, // Temporary buffer provided to GEMM routine is too small
  CLBlastInvalidBatchCount         = -2049, // The batch count needs to be positive
  CLBlastInvalidOverrideKernel     = -2048, // Trying to override parameters for an invalid kernel
  CLBlastMissingOverrideParameter  = -2047, // Missing override parameter(s) for the target kernel
  CLBlastInvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
  CLBlastNoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
  CLBlastNoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
  CLBlastInvalidVectorScalar       = -2043, // The unit-sized vector is not a valid OpenCL buffer
  CLBlastInsufficientMemoryScalar  = -2042, // The unit-sized vector's OpenCL buffer is too small
  CLBlastDatabaseError             = -2041, // Entry for the device was not found in the database
  CLBlastUnknownError              = -2040, // A catch-all error code representing an unspecified error
  CLBlastUnexpectedError           = -2039, // A catch-all error code representing an unexpected exception
} CLBlastStatusCode;

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
typedef enum CLBlastKernelMode_ { CLBlastKernelModeCrossCorrelation = 151, CLBlastKernelModeConvolution = 152 } CLBlastKernelMode;

// Precision enum (values in bits)
typedef enum CLBlastPrecision_ { CLBlastPrecisionHalf = 16, CLBlastPrecisionSingle = 32,
                                 CLBlastPrecisionDouble = 64, CLBlastPrecisionComplexSingle = 3232,
                                 CLBlastPrecisionComplexDouble = 6464 } CLBlastPrecision;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

// Generate givens plane rotation: SROTG/DROTG
CLBlastStatusCode PUBLIC_API CLBlastSrotg(cl_mem sa_buffer, const size_t sa_offset,
                                          cl_mem sb_buffer, const size_t sb_offset,
                                          cl_mem sc_buffer, const size_t sc_offset,
                                          cl_mem ss_buffer, const size_t ss_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrotg(cl_mem sa_buffer, const size_t sa_offset,
                                          cl_mem sb_buffer, const size_t sb_offset,
                                          cl_mem sc_buffer, const size_t sc_offset,
                                          cl_mem ss_buffer, const size_t ss_offset,
                                          cl_command_queue* queue, cl_event* event);

// Generate modified givens plane rotation: SROTMG/DROTMG
CLBlastStatusCode PUBLIC_API CLBlastSrotmg(cl_mem sd1_buffer, const size_t sd1_offset,
                                           cl_mem sd2_buffer, const size_t sd2_offset,
                                           cl_mem sx1_buffer, const size_t sx1_offset,
                                           const cl_mem sy1_buffer, const size_t sy1_offset,
                                           cl_mem sparam_buffer, const size_t sparam_offset,
                                           cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrotmg(cl_mem sd1_buffer, const size_t sd1_offset,
                                           cl_mem sd2_buffer, const size_t sd2_offset,
                                           cl_mem sx1_buffer, const size_t sx1_offset,
                                           const cl_mem sy1_buffer, const size_t sy1_offset,
                                           cl_mem sparam_buffer, const size_t sparam_offset,
                                           cl_command_queue* queue, cl_event* event);

// Apply givens plane rotation: SROT/DROT
CLBlastStatusCode PUBLIC_API CLBlastSrot(const size_t n,
                                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const float cos,
                                         const float sin,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrot(const size_t n,
                                         cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const double cos,
                                         const double sin,
                                         cl_command_queue* queue, cl_event* event);

// Apply modified givens plane rotation: SROTM/DROTM
CLBlastStatusCode PUBLIC_API CLBlastSrotm(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem sparam_buffer, const size_t sparam_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDrotm(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem sparam_buffer, const size_t sparam_offset,
                                          cl_command_queue* queue, cl_event* event);

// Swap two vectors: SSWAP/DSWAP/CSWAP/ZSWAP/HSWAP
CLBlastStatusCode PUBLIC_API CLBlastSswap(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDswap(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCswap(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZswap(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHswap(const size_t n,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Vector scaling: SSCAL/DSCAL/CSCAL/ZSCAL/HSCAL
CLBlastStatusCode PUBLIC_API CLBlastSscal(const size_t n,
                                          const float alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDscal(const size_t n,
                                          const double alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCscal(const size_t n,
                                          const cl_float2 alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZscal(const size_t n,
                                          const cl_double2 alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHscal(const size_t n,
                                          const cl_half alpha,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Vector copy: SCOPY/DCOPY/CCOPY/ZCOPY/HCOPY
CLBlastStatusCode PUBLIC_API CLBlastScopy(const size_t n,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDcopy(const size_t n,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCcopy(const size_t n,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZcopy(const size_t n,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHcopy(const size_t n,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Vector-times-constant plus vector: SAXPY/DAXPY/CAXPY/ZAXPY/HAXPY
CLBlastStatusCode PUBLIC_API CLBlastSaxpy(const size_t n,
                                          const float alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDaxpy(const size_t n,
                                          const double alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCaxpy(const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZaxpy(const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHaxpy(const size_t n,
                                          const cl_half alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Dot product of two vectors: SDOT/DDOT/HDOT
CLBlastStatusCode PUBLIC_API CLBlastSdot(const size_t n,
                                         cl_mem dot_buffer, const size_t dot_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDdot(const size_t n,
                                         cl_mem dot_buffer, const size_t dot_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHdot(const size_t n,
                                         cl_mem dot_buffer, const size_t dot_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors: CDOTU/ZDOTU
CLBlastStatusCode PUBLIC_API CLBlastCdotu(const size_t n,
                                          cl_mem dot_buffer, const size_t dot_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZdotu(const size_t n,
                                          cl_mem dot_buffer, const size_t dot_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Dot product of two complex vectors, one conjugated: CDOTC/ZDOTC
CLBlastStatusCode PUBLIC_API CLBlastCdotc(const size_t n,
                                          cl_mem dot_buffer, const size_t dot_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZdotc(const size_t n,
                                          cl_mem dot_buffer, const size_t dot_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Euclidian norm of a vector: SNRM2/DNRM2/ScNRM2/DzNRM2/HNRM2
CLBlastStatusCode PUBLIC_API CLBlastSnrm2(const size_t n,
                                          cl_mem nrm2_buffer, const size_t nrm2_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDnrm2(const size_t n,
                                          cl_mem nrm2_buffer, const size_t nrm2_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastScnrm2(const size_t n,
                                          cl_mem nrm2_buffer, const size_t nrm2_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDznrm2(const size_t n,
                                          cl_mem nrm2_buffer, const size_t nrm2_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHnrm2(const size_t n,
                                          cl_mem nrm2_buffer, const size_t nrm2_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Absolute sum of values in a vector: SASUM/DASUM/ScASUM/DzASUM/HASUM
CLBlastStatusCode PUBLIC_API CLBlastSasum(const size_t n,
                                          cl_mem asum_buffer, const size_t asum_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDasum(const size_t n,
                                          cl_mem asum_buffer, const size_t asum_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastScasum(const size_t n,
                                          cl_mem asum_buffer, const size_t asum_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDzasum(const size_t n,
                                          cl_mem asum_buffer, const size_t asum_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHasum(const size_t n,
                                          cl_mem asum_buffer, const size_t asum_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Sum of values in a vector (non-BLAS function): SSUM/DSUM/ScSUM/DzSUM/HSUM
CLBlastStatusCode PUBLIC_API CLBlastSsum(const size_t n,
                                         cl_mem sum_buffer, const size_t sum_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsum(const size_t n,
                                         cl_mem sum_buffer, const size_t sum_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastScsum(const size_t n,
                                         cl_mem sum_buffer, const size_t sum_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDzsum(const size_t n,
                                         cl_mem sum_buffer, const size_t sum_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsum(const size_t n,
                                         cl_mem sum_buffer, const size_t sum_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);

// Index of absolute maximum value in a vector: iSAMAX/iDAMAX/iCAMAX/iZAMAX/iHAMAX
CLBlastStatusCode PUBLIC_API CLBlastiSamax(const size_t n,
                                          cl_mem imax_buffer, const size_t imax_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDamax(const size_t n,
                                          cl_mem imax_buffer, const size_t imax_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCamax(const size_t n,
                                          cl_mem imax_buffer, const size_t imax_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZamax(const size_t n,
                                          cl_mem imax_buffer, const size_t imax_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHamax(const size_t n,
                                          cl_mem imax_buffer, const size_t imax_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Index of absolute minimum value in a vector (non-BLAS function): iSAMIN/iDAMIN/iCAMIN/iZAMIN/iHAMIN
CLBlastStatusCode PUBLIC_API CLBlastiSamin(const size_t n,
                                          cl_mem imin_buffer, const size_t imin_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDamin(const size_t n,
                                          cl_mem imin_buffer, const size_t imin_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCamin(const size_t n,
                                          cl_mem imin_buffer, const size_t imin_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZamin(const size_t n,
                                          cl_mem imin_buffer, const size_t imin_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHamin(const size_t n,
                                          cl_mem imin_buffer, const size_t imin_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Index of maximum value in a vector (non-BLAS function): iSMAX/iDMAX/iCMAX/iZMAX/iHMAX
CLBlastStatusCode PUBLIC_API CLBlastiSmax(const size_t n,
                                         cl_mem imax_buffer, const size_t imax_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDmax(const size_t n,
                                         cl_mem imax_buffer, const size_t imax_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCmax(const size_t n,
                                         cl_mem imax_buffer, const size_t imax_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZmax(const size_t n,
                                         cl_mem imax_buffer, const size_t imax_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHmax(const size_t n,
                                         cl_mem imax_buffer, const size_t imax_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);

// Index of minimum value in a vector (non-BLAS function): iSMIN/iDMIN/iCMIN/iZMIN/iHMIN
CLBlastStatusCode PUBLIC_API CLBlastiSmin(const size_t n,
                                         cl_mem imin_buffer, const size_t imin_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiDmin(const size_t n,
                                         cl_mem imin_buffer, const size_t imin_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiCmin(const size_t n,
                                         cl_mem imin_buffer, const size_t imin_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiZmin(const size_t n,
                                         cl_mem imin_buffer, const size_t imin_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastiHmin(const size_t n,
                                         cl_mem imin_buffer, const size_t imin_offset,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

// General matrix-vector multiplication: SGEMV/DGEMV/CGEMV/ZGEMV/HGEMV
CLBlastStatusCode PUBLIC_API CLBlastSgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const float beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const double beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_float2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_double2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_half beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// General banded matrix-vector multiplication: SGBMV/DGBMV/CGBMV/ZGBMV/HGBMV
CLBlastStatusCode PUBLIC_API CLBlastSgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n, const size_t kl, const size_t ku,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const float beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n, const size_t kl, const size_t ku,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const double beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n, const size_t kl, const size_t ku,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_float2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n, const size_t kl, const size_t ku,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_double2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgbmv(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                          const size_t m, const size_t n, const size_t kl, const size_t ku,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_half beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian matrix-vector multiplication: CHEMV/ZHEMV
CLBlastStatusCode PUBLIC_API CLBlastChemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_float2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhemv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_double2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian banded matrix-vector multiplication: CHBMV/ZHBMV
CLBlastStatusCode PUBLIC_API CLBlastChbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n, const size_t k,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_float2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n, const size_t k,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_double2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian packed matrix-vector multiplication: CHPMV/ZHPMV
CLBlastStatusCode PUBLIC_API CLBlastChpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_float2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhpmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_double2 beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric matrix-vector multiplication: SSYMV/DSYMV/HSYMV
CLBlastStatusCode PUBLIC_API CLBlastSsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const float beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const double beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsymv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_half beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric banded matrix-vector multiplication: SSBMV/DSBMV/HSBMV
CLBlastStatusCode PUBLIC_API CLBlastSsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n, const size_t k,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const float beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n, const size_t k,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const double beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsbmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n, const size_t k,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_half beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric packed matrix-vector multiplication: SSPMV/DSPMV/HSPMV
CLBlastStatusCode PUBLIC_API CLBlastSspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const float alpha,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const float beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const double alpha,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const double beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHspmv(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_half alpha,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_half beta,
                                          cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_command_queue* queue, cl_event* event);

// Triangular matrix-vector multiplication: STRMV/DTRMV/CTRMV/ZTRMV/HTRMV
CLBlastStatusCode PUBLIC_API CLBlastStrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtrmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Triangular banded matrix-vector multiplication: STBMV/DTBMV/CTBMV/ZTBMV/HTBMV
CLBlastStatusCode PUBLIC_API CLBlastStbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtbmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Triangular packed matrix-vector multiplication: STPMV/DTPMV/CTPMV/ZTPMV/HTPMV
CLBlastStatusCode PUBLIC_API CLBlastStpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtpmv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Solves a triangular system of equations: STRSV/DTRSV/CTRSV/ZTRSV
CLBlastStatusCode PUBLIC_API CLBlastStrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Solves a banded triangular system of equations: STBSV/DTBSV/CTBSV/ZTBSV
CLBlastStatusCode PUBLIC_API CLBlastStbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtbsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n, const size_t k,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// Solves a packed triangular system of equations: STPSV/DTPSV/CTPSV/ZTPSV
CLBlastStatusCode PUBLIC_API CLBlastStpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtpsv(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t n,
                                          const cl_mem ap_buffer, const size_t ap_offset,
                                          cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          cl_command_queue* queue, cl_event* event);

// General rank-1 matrix update: SGER/DGER/HGER
CLBlastStatusCode PUBLIC_API CLBlastSger(const CLBlastLayout layout,
                                         const size_t m, const size_t n,
                                         const float alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDger(const CLBlastLayout layout,
                                         const size_t m, const size_t n,
                                         const double alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHger(const CLBlastLayout layout,
                                         const size_t m, const size_t n,
                                         const cl_half alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);

// General rank-1 complex matrix update: CGERU/ZGERU
CLBlastStatusCode PUBLIC_API CLBlastCgeru(const CLBlastLayout layout,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgeru(const CLBlastLayout layout,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);

// General rank-1 complex conjugated matrix update: CGERC/ZGERC
CLBlastStatusCode PUBLIC_API CLBlastCgerc(const CLBlastLayout layout,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgerc(const CLBlastLayout layout,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian rank-1 matrix update: CHER/ZHER
CLBlastStatusCode PUBLIC_API CLBlastCher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const float alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZher(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const double alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);

// Hermitian packed rank-1 matrix update: CHPR/ZHPR
CLBlastStatusCode PUBLIC_API CLBlastChpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const float alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem ap_buffer, const size_t ap_offset,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhpr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const double alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem ap_buffer, const size_t ap_offset,
                                         cl_command_queue* queue, cl_event* event);

// Hermitian rank-2 matrix update: CHER2/ZHER2
CLBlastStatusCode PUBLIC_API CLBlastCher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZher2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian packed rank-2 matrix update: CHPR2/ZHPR2
CLBlastStatusCode PUBLIC_API CLBlastChpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem ap_buffer, const size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhpr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem ap_buffer, const size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric rank-1 matrix update: SSYR/DSYR/HSYR
CLBlastStatusCode PUBLIC_API CLBlastSsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const float alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const double alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const cl_half alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                         cl_command_queue* queue, cl_event* event);

// Symmetric packed rank-1 matrix update: SSPR/DSPR/HSPR
CLBlastStatusCode PUBLIC_API CLBlastSspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const float alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem ap_buffer, const size_t ap_offset,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const double alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem ap_buffer, const size_t ap_offset,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHspr(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                         const size_t n,
                                         const cl_half alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         cl_mem ap_buffer, const size_t ap_offset,
                                         cl_command_queue* queue, cl_event* event);

// Symmetric rank-2 matrix update: SSYR2/DSYR2/HSYR2
CLBlastStatusCode PUBLIC_API CLBlastSsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const float alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const double alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_half alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric packed rank-2 matrix update: SSPR2/DSPR2/HSPR2
CLBlastStatusCode PUBLIC_API CLBlastSspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const float alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem ap_buffer, const size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const double alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem ap_buffer, const size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHspr2(const CLBlastLayout layout, const CLBlastTriangle triangle,
                                          const size_t n,
                                          const cl_half alpha,
                                          const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                          const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                          cl_mem ap_buffer, const size_t ap_offset,
                                          cl_command_queue* queue, cl_event* event);

// =================================================================================================
// BLAS level-3 (matrix-matrix) routines
// =================================================================================================

// General matrix-matrix multiplication: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
CLBlastStatusCode PUBLIC_API CLBlastSgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                          const size_t m, const size_t n, const size_t k,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const float beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                          const size_t m, const size_t n, const size_t k,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const double beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                          const size_t m, const size_t n, const size_t k,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_float2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                          const size_t m, const size_t n, const size_t k,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_double2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemm(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                          const size_t m, const size_t n, const size_t k,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_half beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Symmetric matrix-matrix multiplication: SSYMM/DSYMM/CSYMM/ZSYMM/HSYMM
CLBlastStatusCode PUBLIC_API CLBlastSsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const float beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const double beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_float2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_double2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsymm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_half beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Hermitian matrix-matrix multiplication: CHEMM/ZHEMM
CLBlastStatusCode PUBLIC_API CLBlastChemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_float2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhemm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          const cl_double2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Rank-K update of a symmetric matrix: SSYRK/DSYRK/CSYRK/ZSYRK/HSYRK
CLBlastStatusCode PUBLIC_API CLBlastSsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const float beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const double beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_float2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_double2 beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyrk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const cl_half beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Rank-K update of a hermitian matrix: CHERK/ZHERK
CLBlastStatusCode PUBLIC_API CLBlastCherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const float beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZherk(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
                                          const size_t n, const size_t k,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          const double beta,
                                          cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                          cl_command_queue* queue, cl_event* event);

// Rank-2K update of a symmetric matrix: SSYR2K/DSYR2K/CSYR2K/ZSYR2K/HSYR2K
CLBlastStatusCode PUBLIC_API CLBlastSsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const float alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const float beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const double alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const double beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const cl_float2 alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const cl_float2 beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const cl_double2 alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const cl_double2 beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHsyr2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const cl_half alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const cl_half beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);

// Rank-2K update of a hermitian matrix: CHER2K/ZHER2K
CLBlastStatusCode PUBLIC_API CLBlastCher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const cl_float2 alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const float beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZher2k(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
                                           const size_t n, const size_t k,
                                           const cl_double2 alpha,
                                           const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                           const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                           const double beta,
                                           cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                           cl_command_queue* queue, cl_event* event);

// Triangular matrix-matrix multiplication: STRMM/DTRMM/CTRMM/ZTRMM/HTRMM
CLBlastStatusCode PUBLIC_API CLBlastStrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHtrmm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const cl_half alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);

// Solves a triangular system of equations: STRSM/DTRSM/CTRSM/ZTRSM
CLBlastStatusCode PUBLIC_API CLBlastStrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const float alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const double alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const cl_float2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZtrsm(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
                                          const size_t m, const size_t n,
                                          const cl_double2 alpha,
                                          const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                          cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                          cl_command_queue* queue, cl_event* event);

// =================================================================================================
// Extra non-BLAS routines (level-X)
// =================================================================================================

// Element-wise vector product (Hadamard): SHAD/DHAD/CHAD/ZHAD/HHAD
CLBlastStatusCode PUBLIC_API CLBlastShad(const size_t n,
                                         const float alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const float beta,
                                         cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDhad(const size_t n,
                                         const double alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const double beta,
                                         cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastChad(const size_t n,
                                         const cl_float2 alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const cl_float2 beta,
                                         cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZhad(const size_t n,
                                         const cl_double2 alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const cl_double2 beta,
                                         cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                                         cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHhad(const size_t n,
                                         const cl_half alpha,
                                         const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
                                         const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
                                         const cl_half beta,
                                         cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
                                         cl_command_queue* queue, cl_event* event);

// Scaling and out-place transpose/copy (non-BLAS function): SOMATCOPY/DOMATCOPY/COMATCOPY/ZOMATCOPY/HOMATCOPY
CLBlastStatusCode PUBLIC_API CLBlastSomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                              const size_t m, const size_t n,
                                              const float alpha,
                                              const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                              cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                              const size_t m, const size_t n,
                                              const double alpha,
                                              const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                              cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastComatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                              const size_t m, const size_t n,
                                              const cl_float2 alpha,
                                              const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                              cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                              const size_t m, const size_t n,
                                              const cl_double2 alpha,
                                              const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                              cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHomatcopy(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
                                              const size_t m, const size_t n,
                                              const cl_half alpha,
                                              const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                              cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                              cl_command_queue* queue, cl_event* event);

// Im2col function (non-BLAS function): SIM2COL/DIM2COL/CIM2COL/ZIM2COL/HIM2COL
CLBlastStatusCode PUBLIC_API CLBlastSim2col(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem im_buffer, const size_t im_offset,
                                            cl_mem col_buffer, const size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDim2col(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem im_buffer, const size_t im_offset,
                                            cl_mem col_buffer, const size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCim2col(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem im_buffer, const size_t im_offset,
                                            cl_mem col_buffer, const size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZim2col(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem im_buffer, const size_t im_offset,
                                            cl_mem col_buffer, const size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHim2col(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem im_buffer, const size_t im_offset,
                                            cl_mem col_buffer, const size_t col_offset,
                                            cl_command_queue* queue, cl_event* event);

// Col2im function (non-BLAS function): SCOL2IM/DCOL2IM/CCOL2IM/ZCOL2IM/HCOL2IM
CLBlastStatusCode PUBLIC_API CLBlastScol2im(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem col_buffer, const size_t col_offset,
                                            cl_mem im_buffer, const size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDcol2im(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem col_buffer, const size_t col_offset,
                                            cl_mem im_buffer, const size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCcol2im(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem col_buffer, const size_t col_offset,
                                            cl_mem im_buffer, const size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZcol2im(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem col_buffer, const size_t col_offset,
                                            cl_mem im_buffer, const size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHcol2im(const CLBlastKernelMode kernel_mode,
                                            const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
                                            const cl_mem col_buffer, const size_t col_offset,
                                            cl_mem im_buffer, const size_t im_offset,
                                            cl_command_queue* queue, cl_event* event);

// Batched convolution as GEMM (non-BLAS function): SCONVGEMM/DCONVGEMM/HCONVGEMM
CLBlastStatusCode PUBLIC_API CLBlastSconvgemm(const CLBlastKernelMode kernel_mode,
                                              const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                                              const cl_mem im_buffer, const size_t im_offset,
                                              const cl_mem kernel_buffer, const size_t kernel_offset,
                                              cl_mem result_buffer, const size_t result_offset,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDconvgemm(const CLBlastKernelMode kernel_mode,
                                              const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                                              const cl_mem im_buffer, const size_t im_offset,
                                              const cl_mem kernel_buffer, const size_t kernel_offset,
                                              cl_mem result_buffer, const size_t result_offset,
                                              cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHconvgemm(const CLBlastKernelMode kernel_mode,
                                              const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
                                              const cl_mem im_buffer, const size_t im_offset,
                                              const cl_mem kernel_buffer, const size_t kernel_offset,
                                              cl_mem result_buffer, const size_t result_offset,
                                              cl_command_queue* queue, cl_event* event);

// Batched version of AXPY: SAXPYBATCHED/DAXPYBATCHED/CAXPYBATCHED/ZAXPYBATCHED/HAXPYBATCHED
CLBlastStatusCode PUBLIC_API CLBlastSaxpyBatched(const size_t n,
                                                 const float *alphas,
                                                 const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                                 cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDaxpyBatched(const size_t n,
                                                 const double *alphas,
                                                 const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                                 cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCaxpyBatched(const size_t n,
                                                 const cl_float2 *alphas,
                                                 const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                                 cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZaxpyBatched(const size_t n,
                                                 const cl_double2 *alphas,
                                                 const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                                 cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHaxpyBatched(const size_t n,
                                                 const cl_half *alphas,
                                                 const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
                                                 cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);

// Batched version of GEMM: SGEMMBATCHED/DGEMMBATCHED/CGEMMBATCHED/ZGEMMBATCHED/HGEMMBATCHED
CLBlastStatusCode PUBLIC_API CLBlastSgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                 const size_t m, const size_t n, const size_t k,
                                                 const float *alphas,
                                                 const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                                 const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                                 const float *betas,
                                                 cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                 const size_t m, const size_t n, const size_t k,
                                                 const double *alphas,
                                                 const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                                 const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                                 const double *betas,
                                                 cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                 const size_t m, const size_t n, const size_t k,
                                                 const cl_float2 *alphas,
                                                 const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                                 const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                                 const cl_float2 *betas,
                                                 cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                 const size_t m, const size_t n, const size_t k,
                                                 const cl_double2 *alphas,
                                                 const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                                 const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                                 const cl_double2 *betas,
                                                 cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemmBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                 const size_t m, const size_t n, const size_t k,
                                                 const cl_half *alphas,
                                                 const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
                                                 const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
                                                 const cl_half *betas,
                                                 cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
                                                 const size_t batch_count,
                                                 cl_command_queue* queue, cl_event* event);

// StridedBatched version of GEMM: SGEMMSTRIDEDBATCHED/DGEMMSTRIDEDBATCHED/CGEMMSTRIDEDBATCHED/ZGEMMSTRIDEDBATCHED/HGEMMSTRIDEDBATCHED
CLBlastStatusCode PUBLIC_API CLBlastSgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const float alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                                        const float beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                                        const size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastDgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const double alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                                        const double beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                                        const size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastCgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const cl_float2 alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                                        const cl_float2 beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                                        const size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastZgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const cl_double2 alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                                        const cl_double2 beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                                        const size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);
CLBlastStatusCode PUBLIC_API CLBlastHgemmStridedBatched(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const cl_half alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
                                                        const cl_half beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
                                                        const size_t batch_count,
                                                        cl_command_queue* queue, cl_event* event);

// =================================================================================================
// General matrix-matrix multiplication with temporary buffer from user (optional, for advanced users): SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM
CLBlastStatusCode PUBLIC_API CLBlastSgemmWithTempBuffer(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const float alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                                        const float beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastDgemmWithTempBuffer(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const double alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                                        const double beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastCgemmWithTempBuffer(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const cl_float2 alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                                        const cl_float2 beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastZgemmWithTempBuffer(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const cl_double2 alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                                        const cl_double2 beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);
CLBlastStatusCode PUBLIC_API CLBlastHgemmWithTempBuffer(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const cl_half alpha,
                                                        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
                                                        const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
                                                        const cl_half beta,
                                                        cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue, cl_event* event, cl_mem temp_buffer);

// =================================================================================================
// Retrieves the required size of the temporary buffer for the GEMM kernel: SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM (optional)
CLBlastStatusCode PUBLIC_API CLBlastSGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const size_t a_offset, const size_t a_ld,
                                                        const size_t b_offset, const size_t b_ld,
                                                        const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastDGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const size_t a_offset, const size_t a_ld,
                                                        const size_t b_offset, const size_t b_ld,
                                                        const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastCGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const size_t a_offset, const size_t a_ld,
                                                        const size_t b_offset, const size_t b_ld,
                                                        const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastZGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const size_t a_offset, const size_t a_ld,
                                                        const size_t b_offset, const size_t b_ld,
                                                        const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

CLBlastStatusCode PUBLIC_API CLBlastHGemmTempBufferSize(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
                                                        const size_t m, const size_t n, const size_t k,
                                                        const size_t a_offset, const size_t a_ld,
                                                        const size_t b_offset, const size_t b_ld,
                                                        const size_t c_offset, const size_t c_ld,
                                                        cl_command_queue* queue,
                                                        size_t* temp_buffer_size);

// =================================================================================================

// CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on
// for the same device. This cache can be cleared to free up system memory or in case of debugging.
CLBlastStatusCode PUBLIC_API CLBlastClearCache();

// The cache can also be pre-initialized for a specific device with all possible CLBlast kernels.
// Further CLBlast routine calls will then run at maximum speed.
CLBlastStatusCode PUBLIC_API CLBlastFillCache(const cl_device_id device);

// =================================================================================================

// Overrides tuning parameters for a specific device-precision-kernel combination. The next time
// the target routine is called it will re-compile and use the new parameters from then on.
CLBlastStatusCode PUBLIC_API CLBlastOverrideParameters(const cl_device_id device, const char* kernel_name,
                                                       const CLBlastPrecision precision, const size_t num_parameters,
                                                       const char** parameters_names, const size_t* parameters_values);

// =================================================================================================

#ifdef __cplusplus
} // extern "C"
#endif

// CLBLAST_CLBLAST_C_H_
#endif
