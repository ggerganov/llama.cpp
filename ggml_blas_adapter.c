//this is a drop-in for all CLBlast related code, to keep the main ggml.c unmodified
// we will imitate the function definition from OpenBLAS instead, replaced as necessary.

//windows binaries for clblast obtained from https://github.com/CNugteren/CLBlast (apache license)
//windows binaries for opencl obtained from https://github.com/KhronosGroup/OpenCL-SDK (apache license)

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

#if GGML_USE_CLBLAST

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_event event;
bool cl_initialized = false;

static void ggml_cl_sgemm_wrapper(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const float alpha, const float *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc) {
    cl_int err = 0;

    if (!cl_initialized) {
        char * KCPP_CLBLAST_PLATFORM = getenv("KCPP_CLBLAST_PLATFORM");
        char * KCPP_CLBLAST_DEVICES = getenv("KCPP_CLBLAST_DEVICES");       
        int plat_num = (KCPP_CLBLAST_PLATFORM == NULL ? 0 : atoi(KCPP_CLBLAST_PLATFORM));
        int dev_num = (KCPP_CLBLAST_DEVICES == NULL ? 0 : atoi(KCPP_CLBLAST_DEVICES));
        printf("\nInitializing CLBlast (First Run)...");
        printf("\nSelected: Platform=%d, Device=%d (If invalid, program will crash)\n",plat_num,dev_num);
        cl_uint num_platforms;
        clGetPlatformIDs(0, NULL, &num_platforms);
        cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
        clGetPlatformIDs(num_platforms, platforms, NULL);
        platform = platforms[plat_num];
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        device = devices[dev_num];
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL context: %d\n", err);
            fflush(stdout);
        }
        queue = clCreateCommandQueue(context, device, 0, &err);
        event = NULL;

        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Command Queue: %d\n", err);
            fflush(stdout);
        }

        free(platforms);
        free(devices);
        cl_initialized = true;
    }

    // Prepare buffers
    cl_mem cl_buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*k*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL Buffer A: %d\n", err);
        fflush(stdout);
    }
    cl_mem cl_buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*k*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL Buffer B: %d\n", err);
        fflush(stdout);
    }
    cl_mem cl_buffer_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL Buffer C: %d\n", err);
        fflush(stdout);
    }

    clEnqueueWriteBuffer(queue, cl_buffer_a, CL_TRUE, 0, m*k*sizeof(float), host_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cl_buffer_b, CL_TRUE, 0, n*k*sizeof(float), host_b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cl_buffer_c, CL_TRUE, 0, m*n*sizeof(float), host_c, 0, NULL, NULL);

    // Call the SGEMM routine.
    CLBlastStatusCode status = CLBlastSgemm(order,
                                            trans_a, trans_b,
                                            m, n, k,
                                            alpha,
                                            cl_buffer_a, 0, lda,
                                            cl_buffer_b, 0, ldb,
                                            beta,
                                            cl_buffer_c, 0, ldc,
                                            &queue, &event);

    // Wait for completion
    if (status == CLBlastSuccess) {
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    }

    clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, m*n*sizeof(float), host_c, 0, NULL, NULL);

    clReleaseMemObject(cl_buffer_a);
    clReleaseMemObject(cl_buffer_b);
    clReleaseMemObject(cl_buffer_c);
}

#endif

static void do_blas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc)
{
#if GGML_USE_CLBLAST
    ggml_cl_sgemm_wrapper(Order, TransA, TransB,
                M, N, K,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
#else
    cblas_sgemm(Order, TransA, TransB,
                M, N, K,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
#endif
}