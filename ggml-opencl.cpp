#include "ggml-opencl.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#include "ggml.h"

#include <ggml_clblast_dequant.cl>

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_q4_0, kernel_q4_1, kernel_q4_2, kernel_q4_3;
cl_mem cl_buffer_a, cl_buffer_qb, cl_buffer_b, cl_buffer_c;
size_t cl_size_a = 0, cl_size_qb = 0, cl_size_b = 0, cl_size_c = 0;

cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {
   cl_program program;
   char *program_log;
   size_t program_size, log_size;
   int err;

   program_size = strlen(program_buffer);

   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      fprintf(stderr, "OpenCL error creating program");
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

void ggml_cl_init(void) {
    cl_int err = 0;
    char * GGML_CLBLAST_PLATFORM = getenv("GGML_CLBLAST_PLATFORM");
    char * GGML_CLBLAST_DEVICE = getenv("GGML_CLBLAST_DEVICE");
    int plat_num = (GGML_CLBLAST_PLATFORM == NULL ? 0 : atoi(GGML_CLBLAST_PLATFORM));
    int dev_num = (GGML_CLBLAST_DEVICE == NULL ? 0 : atoi(GGML_CLBLAST_DEVICE));
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
    CL_CHECK(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    free(platforms);
    free(devices);

    program = build_program_from_source(context, device, clblast_dequant);

    // Prepare dequantize kernels
    kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_1 = clCreateKernel(program, "dequantize_row_q4_1", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_2 = clCreateKernel(program, "dequantize_row_q4_2", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_3 = clCreateKernel(program, "dequantize_row_q4_3", &err);
    CL_CHECK(err, "clCreateKernel");
}

void ggml_cl_malloc(size_t req_size, size_t* cur_size, cl_mem_flags flags, cl_mem* buf) {
    if (req_size <= *cur_size) {
        return;
    }

    // Reallocate buffer with enough space
    if (*cur_size > 0) {
        clReleaseMemObject(*buf);
    }
    cl_int err;
    *buf = clCreateBuffer(context, flags, req_size, NULL, &err);
    *cur_size = req_size;
    CL_CHECK(err, "clCreateBuffer");
}

void ggml_cl_sgemm_wrapper(const CLBlastLayout order, const CLBlastTranspose trans_a, const CLBlastTranspose trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype) {
    cl_int err = 0;

    cl_event events[4] = { NULL };

    cl_kernel kernel;
    size_t global = n * k, local, size_qb;
    bool dequant;

    switch (btype) {
    case GGML_TYPE_F32:
        dequant = false;
        break;
    case GGML_TYPE_Q4_0:
        dequant = true;
        kernel = kernel_q4_0;
        local = 16;
        size_qb = global * (sizeof(float) + local) / 32;
        break;
    case GGML_TYPE_Q4_1:
        dequant = true;
        kernel = kernel_q4_1;
        local = 16;
        size_qb = global * (sizeof(float) * 2 + local) / 32;
        break;
    case GGML_TYPE_Q4_2:
        dequant = true;
        kernel = kernel_q4_2;
        local = 8;
        size_qb = global * (sizeof(short) + local) / 16;
        break;
    case GGML_TYPE_Q4_3:
        dequant = true;
        kernel = kernel_q4_3;
        local = 8;
        size_qb = global * (sizeof(short) * 2 + local) / 16;
        break;
    default:
        fprintf(stderr, "Error: Unsupported OpenCL btype %d\n", btype);
        abort();
    }

    const size_t size_a =  m * k * sizeof(float);
    const size_t size_b =  n * k * sizeof(float);
    const size_t size_c =  m * n * sizeof(float);

    // Prepare buffers
    ggml_cl_malloc(size_a, &cl_size_a, CL_MEM_READ_ONLY, &cl_buffer_a);
    if (dequant) {
        ggml_cl_malloc(size_qb, &cl_size_qb, CL_MEM_READ_ONLY, &cl_buffer_qb);
    }
    ggml_cl_malloc(size_b, &cl_size_b, CL_MEM_READ_WRITE, &cl_buffer_b);
    ggml_cl_malloc(size_c, &cl_size_c, CL_MEM_WRITE_ONLY, &cl_buffer_c);

    if (dequant) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
        CL_CHECK(err, "clSetKernelArg");
        clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_FALSE, 0, size_qb, host_b, 0, NULL, events + 1);
    } else {
        clEnqueueWriteBuffer(queue, cl_buffer_b, CL_FALSE, 0, size_b, host_b, 0, NULL, events + 1);
    }

    clEnqueueWriteBuffer(queue, cl_buffer_a, CL_FALSE, 0, size_a, host_a, 0, NULL, events);
    if (dequant) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, events + 1, events + 3);
        CL_CHECK(err, "clEnqueueNDRangeKernel");
    }
    clWaitForEvents(dequant ? 4 : 3, events);
    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);
    clReleaseEvent(events[2]);
    if (dequant) {
        clReleaseEvent(events[3]);
    }

    CLBlastSgemm(order,
                 trans_a, trans_b,
                 m, n, k,
                 alpha,
                 cl_buffer_a, 0, lda,
                 cl_buffer_b, 0, ldb,
                 beta,
                 cl_buffer_c, 0, ldc,
                 &queue, events);

    clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, host_c, 1, events, events + 1);

    // Wait for completion
    clWaitForEvents(2, events);
    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);
}
