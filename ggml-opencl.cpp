#include "ggml-opencl.h"

#include <atomic>
#include <cstdio>
#include <cstring>

#include <ggml_clblast_dequant.cl>

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cl_buffer {
    cl_mem mem;
    size_t size = 0;
};

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_q4_0, kernel_q4_1, kernel_q4_2, kernel_q4_3;
size_t cl_size_a = 0, cl_size_b = 0, cl_size_c = 0;

static cl_buffer g_cl_buffer_pool[MAX_CL_BUFFERS];
static std::atomic_flag g_cl_pool_lock = ATOMIC_FLAG_INIT;

cl_mem ggml_cl_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cl_pool_lock);

    for (int i = 0; i < MAX_CL_BUFFERS; ++i) {
        cl_buffer& b = g_cl_buffer_pool[i];
        if (b.size >= size && b.size != 0) {
            cl_mem mem = b.mem;
            *actual_size = b.size;
            b.size = 0;
            return mem;
        }
    }
    cl_int err;
    cl_mem mem = clCreateBuffer(context, 0, size, NULL, &err);
    *actual_size = size;
    CL_CHECK(err, "clCreateBuffer");
    return mem;
}

void ggml_cl_pool_free(cl_mem mem, size_t size) {
    scoped_spin_lock lock(g_cl_pool_lock);

    for (int i = 0; i < MAX_CL_BUFFERS; ++i) {
        cl_buffer& b = g_cl_buffer_pool[i];
        if (b.size == 0) {
            b.mem = mem;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cl buffer pool full, increase MAX_CL_BUFFERS\n");
    clReleaseMemObject(mem);
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

void ggml_cl_init() {
    cl_int err = 0;
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

void ggml_cl_sgemm_wrapper(const CLBlastLayout order, const CLBlastTranspose trans_a, const CLBlastTranspose trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype) {
    cl_int err = 0;

    cl_event events[4];
    events[0] = NULL;
    events[1] = NULL;
    events[2] = NULL;
    events[3] = NULL;

    cl_kernel kernel;
    size_t global, local, qb_size;
    bool dequant = btype >= 2 && btype < 6;
    if (dequant) {
        global = n * k;

        switch (btype) {
        case 2:
            kernel = kernel_q4_0;
            local = 16;
            qb_size = global * (sizeof(float) + local) / 32;
            break;
        case 3:
            kernel = kernel_q4_1;
            local = 16;
            qb_size = global * (sizeof(float) * 2 + local) / 32;
            break;
        case 4:
            kernel = kernel_q4_2;
            local = 8;
            qb_size = global * (sizeof(short) + local) / 16;
            break;
        case 5:
            kernel = kernel_q4_3;
            local = 8;
            qb_size = global * (sizeof(short) * 2 + local) / 16;
            break;
        }
    }

    cl_mem cl_buffer_a, cl_buffer_qb, cl_buffer_b, cl_buffer_c;

    size_t buf_size_a, buf_size_qb, buf_size_b, buf_size_c;

    // Prepare buffers
    cl_buffer_a = ggml_cl_pool_malloc(m * k * sizeof(float), &buf_size_a);
    if (dequant) {
        cl_buffer_qb = ggml_cl_pool_malloc(qb_size, &buf_size_qb);
    }
    cl_buffer_b = ggml_cl_pool_malloc(n*k*sizeof(float), &buf_size_b);
    cl_buffer_c = ggml_cl_pool_malloc(m*n*sizeof(float), &buf_size_c);

    if (dequant) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
        CL_CHECK(err, "clSetKernelArg");
        clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_FALSE, 0, qb_size, host_b, 0, NULL, events + 1);
    } else {
        clEnqueueWriteBuffer(queue, cl_buffer_b, CL_FALSE, 0, n*k*sizeof(float), host_b, 0, NULL, events + 1);
    }

    clEnqueueWriteBuffer(queue, cl_buffer_a, CL_FALSE, 0, m*k*sizeof(float), host_a, 0, NULL, events);
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
    clWaitForEvents(2, events);
    clReleaseEvent(events[0]);
    clReleaseEvent(events[1]);

    ggml_cl_pool_free(cl_buffer_a, buf_size_a);
    if (dequant) {
        ggml_cl_pool_free(cl_buffer_qb, buf_size_qb);
    }
    ggml_cl_pool_free(cl_buffer_b, buf_size_b);
    ggml_cl_pool_free(cl_buffer_c, buf_size_c);
}
