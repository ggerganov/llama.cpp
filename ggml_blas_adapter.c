//this is a drop-in for all CLBlast related code, to keep the main ggml.c unmodified
// we will imitate the function definition from OpenBLAS instead, replaced as necessary.

//windows binaries for clblast obtained from https://github.com/CNugteren/CLBlast (apache license)
//windows binaries for opencl obtained from https://github.com/KhronosGroup/OpenCL-SDK (apache license)

#if GGML_USE_OPENBLAS
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

#if GGML_USE_CLBLAST

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>
#include <ggml_clblast_dequant.cl>

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_q4_0, kernel_q4_1;
bool cl_initialized = false;

size_t cl_size_a = 0, cl_size_b = 0, cl_size_qb = 0, cl_size_c = 0;
cl_mem cl_buffer_a, cl_buffer_b, cl_buffer_qb, cl_buffer_c;

// Function taken from https://github.com/rsnemmen/OpenCL-examples/blob/master/add_numbers/add_numbers.c
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("OpenCL kernel file not found");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("OpenCL error creating program");
      exit(1);
   }
   free(program_buffer);

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {

   cl_program program;
   char *program_log;
   size_t program_size, log_size;
   int err;

   program_size = strlen(program_buffer);
   
   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("OpenCL error creating program");
      exit(1);
   }

   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

static void ggml_cl_sgemm_wrapper(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype) {
    cl_int err = 0;

    cl_event events[4];
    events[0] = NULL;
    events[1] = NULL;
    events[2] = NULL;
    events[3] = NULL;

    if (!cl_initialized) {
        char * KCPP_CLBLAST_PLATFORM = getenv("KCPP_CLBLAST_PLATFORM");
        char * KCPP_CLBLAST_DEVICES = getenv("KCPP_CLBLAST_DEVICES");
        int plat_num = (KCPP_CLBLAST_PLATFORM == NULL ? 0 : atoi(KCPP_CLBLAST_PLATFORM));
        int dev_num = (KCPP_CLBLAST_DEVICES == NULL ? 0 : atoi(KCPP_CLBLAST_DEVICES));
        printf("\nInitializing CLBlast (First Run)...");
        printf("\nAttempting to use: Platform=%d, Device=%d (If invalid, program will crash)\n",plat_num,dev_num);
        cl_uint num_platforms;
        clGetPlatformIDs(0, NULL, &num_platforms);
        cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
        clGetPlatformIDs(num_platforms, platforms, NULL);
        platform = platforms[plat_num];
        char platform_buffer[1024];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_buffer), &platform_buffer, NULL);
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        device = devices[dev_num];
        char device_buffer[1024];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_buffer), &device_buffer, NULL);
        printf("Using Platform: %s Device: %s\n", platform_buffer, device_buffer);
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL context: %d\n", err);
            fflush(stdout);
        }
        queue = clCreateCommandQueue(context, device, 0, &err);

        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Command Queue: %d\n", err);
            fflush(stdout);
        }

        free(platforms);
        free(devices);

        program = build_program_from_source(context, device, clblast_dequant);

        // Prepare dequantize kernels
        kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err);
        if(err < 0) {
            printf("Error creating OpenCL dequantize q4_0 kernel: %d\n", err);
            fflush(stdout);
        };
        kernel_q4_1 = clCreateKernel(program, "dequantize_row_q4_1", &err);
        if(err < 0) {
            printf("Error creating OpenCL dequantize q4_1 kernel: %d\n", err);
            fflush(stdout);
        };

        size_t defaultBufSize = 8*1024*1024;
        cl_size_a = defaultBufSize * sizeof(float);
        cl_size_b = defaultBufSize * sizeof(float);
        cl_size_qb = defaultBufSize * sizeof(float);
        cl_size_c = defaultBufSize * sizeof(float);
        // Prepare buffers
        cl_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, cl_size_a, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer A: %d\n", err);
            fflush(stdout);
        }
        cl_buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_size_b, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer B: %d\n", err);
            fflush(stdout);
        }
        cl_buffer_qb = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_size_qb, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer B: %d\n", err);
            fflush(stdout);
        }
        cl_buffer_c = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_size_c, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer C: %d\n", err);
            fflush(stdout);
        }

        cl_initialized = true;
    }

    bool dequant = (btype == 2 || btype == 3);
    cl_kernel kernel = btype == 2 ? kernel_q4_0 : kernel_q4_1;

    size_t global = n * k, local = 16, qb_size;

    // Prepare buffers
    if(m*k*sizeof(float) > cl_size_a)
    {
        cl_size_a = m*k*sizeof(float);
        clReleaseMemObject(cl_buffer_a);
        cl_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, cl_size_a, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer A: %d\n", err);
            fflush(stdout);
        }
        //printf("\nRealloc A: %d",cl_size_a);
    }
    if (dequant) {        
        qb_size = global * (sizeof(float) * (btype == 2 ? 1 : 2) + 16) / 32;
        if(qb_size > cl_size_qb)
        {
            cl_size_qb = qb_size;
            clReleaseMemObject(cl_buffer_qb);
            cl_buffer_qb = clCreateBuffer(context, CL_MEM_READ_ONLY, qb_size, NULL, &err);
            if (err != CL_SUCCESS) {
                printf("Error creating OpenCL Buffer QB: %d\n", err);
                fflush(stdout);
            }
            //printf("\nRealloc qB: %d",cl_size_qb);
        }
    }
    if(n*k*sizeof(float) > cl_size_b)
    {
        cl_size_b = n*k*sizeof(float);
        clReleaseMemObject(cl_buffer_b);
        cl_buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_size_b, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer B: %d\n", err);
            fflush(stdout);
        }
        //printf("\nRealloc B: %d",cl_size_b);
    }
    if(m*n*sizeof(float) > cl_size_c)
    {
        cl_size_c = m*n*sizeof(float);
        clReleaseMemObject(cl_buffer_c);
        cl_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cl_size_c, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error creating OpenCL Buffer C: %d\n", err);
            fflush(stdout);
        }
        //printf("\nRealloc C: %d",cl_size_c);
    }

    if (dequant) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
        if(err < 0) {
            printf("Error setting OpenCL kernel args: %d\n", err);
            fflush(stdout);
        }
        clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_FALSE, 0, qb_size, host_b, 0, NULL, events + 1);
    } else {
        clEnqueueWriteBuffer(queue, cl_buffer_b, CL_FALSE, 0, n*k*sizeof(float), host_b, 0, NULL, events + 1);
    }

    clEnqueueWriteBuffer(queue, cl_buffer_a, CL_FALSE, 0, m*k*sizeof(float), host_a, 0, NULL, events);
    //clEnqueueWriteBuffer(queue, cl_buffer_c, CL_FALSE, 0, m*n*sizeof(float), host_c, 0, NULL, events + 2);
    if (dequant) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, events + 1, events + 2);
        if(err < 0) {
            printf("Error enqueueing OpenCL dequantize kernel: %d\n", err);
            fflush(stdout);
        }
    }
    clWaitForEvents(dequant ? 3 : 2, events);
    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);
    //clReleaseEvent(events[2]);
    if (dequant) {
        clReleaseEvent(events[2]);
    }

    // Call the SGEMM routine.
    CLBlastStatusCode status = CLBlastSgemm(order,
                                            trans_a, trans_b,
                                            m, n, k,
                                            alpha,
                                            cl_buffer_a, 0, lda,
                                            cl_buffer_b, 0, ldb,
                                            beta,
                                            cl_buffer_c, 0, ldc,
                                            &queue, events);

    clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, m*n*sizeof(float), host_c, 1, events, events + 1);

    // Wait for completion
    if (status == CLBlastSuccess) {
        clWaitForEvents(2, events);
        clReleaseEvent(events[0]);
        clReleaseEvent(events[1]);
    }
    
}
#endif
#endif

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
#if GGML_USE_CLBLAST
#define do_blas_sgemm(Order, TransA, TransB,M, N, K,alpha, A, lda, B, ldb, beta, C, ldc, btype) ({\
ggml_cl_sgemm_wrapper(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, btype);\
})
#else
#define do_blas_sgemm(Order, TransA, TransB,M, N, K,alpha, A, lda, B, ldb, beta, C, ldc, btype) ({\
cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);\
})
#endif
#endif
