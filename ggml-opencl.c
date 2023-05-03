#include "ggml-opencl.h"

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"

#define MULTILINE_QUOTE(...) #__VA_ARGS__
const char * clblast_dequant = MULTILINE_QUOTE(

struct block_q4_0
{
    float d;
    uchar qs[16];
};

__kernel void dequantize_row_q4_0(__global struct block_q4_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uchar vi = blocks[i].qs[l];

    const uint index = i*32 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}

struct block_q4_1
{
    float d;
    float m;
    uchar qs[16];
};

__kernel void dequantize_row_q4_1(__global struct block_q4_1* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;
    const float m = blocks[i].m;

    const uchar vi = blocks[i].qs[l];

    const uint index = i*32 + l*2;
    result[index + 0] = (vi & 0xf) * d + m;
    result[index + 1] = (vi >> 4) * d + m;
}

struct block_q4_2
{
    ushort d;
    uchar qs[8];
};

__kernel void dequantize_row_q4_2(__global struct block_q4_2* blocks, __global float* result) {
    const uint i = get_global_id(0) / 16;
    const uint l = get_local_id(0);

    const float d = vload_half(0, (__global half*) &blocks[i].d);

    const uchar vi = blocks[i].qs[l];

    const uint index = i*16 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}


struct block_q5_0
{
    float d;
    uint qh;
    uchar qs[16];
};

__kernel void dequantize_row_q5_0(__global struct block_q5_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uchar vi = blocks[i].qs[l];

    const uint l2 = l * 2;

    const uchar vh0 = ((blocks[i].qh & (1 << (l2 + 0))) >> (l2 + 0)) << 4;
    const uchar vh1 = ((blocks[i].qh & (1 << (l2 + 1))) >> (l2 + 1)) << 4;

    const uint index = i*32 + l2;
    result[index + 0] = (((vi & 0xf) | vh0) - 16)*d;
    result[index + 1] = (((vi >>  4) | vh1) - 16)*d;
}

struct block_q5_1
{
    ushort d;
    ushort m;
    uint qh;
    uchar qs[16];
};

__kernel void dequantize_row_q5_1(__global struct block_q5_1* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = vload_half(0, (__global half*) &blocks[i].d);
    const float m = vload_half(0, (__global half*) &blocks[i].m);

    const uchar vi = blocks[i].qs[l];

    const uint l2 = l * 2;

    const uchar vh0 = ((blocks[i].qh & (1 << (l2 + 0))) >> (l2 + 0)) << 4;
    const uchar vh1 = ((blocks[i].qh & (1 << (l2 + 1))) >> (l2 + 1)) << 4;

    const uint index = i*32 + l2;
    result[index + 0] = ((vi & 0xf) | vh0)*d + m;
    result[index + 1] = ((vi >>  4) | vh1)*d + m;
}

struct block_q8_0
{
    float d;
    char qs[32];
};

__kernel void dequantize_row_q8_0(__global struct block_q8_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    result[i*32 + l] = blocks[i].qs[l] * blocks[i].d;
}

);

#define CL_CHECK(err, name)                                                                     \
    do {                                                                                        \
        cl_int err_ = (err);                                                                    \
        if (err_ != CL_SUCCESS) {                                                               \
            fprintf(stderr, "OpenCL %s error %d at %s:%d\n", name, err_, __FILE__, __LINE__);   \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

#define QK5_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;


typedef struct {
    float d;                // delta
    uint32_t qh;          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} cl_block_q5_0;

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel_q4_0, kernel_q4_1, kernel_q4_2, kernel_q5_0, kernel_q5_1, kernel_q8_0;
static cl_mem cl_buffer_a, cl_buffer_qb, cl_buffer_b, cl_buffer_c;
static size_t cl_size_a = 0, cl_size_qb = 0, cl_size_b = 0, cl_size_c = 0;

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {
    cl_program p;
    char *program_log;
    size_t program_size, log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        fprintf(stderr, "OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
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
    queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
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
    kernel_q5_0 = clCreateKernel(program, "dequantize_row_q5_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q5_1 = clCreateKernel(program, "dequantize_row_q5_1", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q8_0 = clCreateKernel(program, "dequantize_row_q8_0", &err);
    CL_CHECK(err, "clCreateKernel");
}

static void ggml_cl_malloc(size_t req_size, size_t* cur_size, cl_mem_flags flags, cl_mem* buf) {
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

void ggml_cl_sgemm_wrapper(
        const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b,
        const int m, const int n, const int k,
        const float alpha, const void *host_a, const int lda,
        const float *host_b, const int ldb, const float beta,
        float *host_c, const int ldc, const int btype) {
    cl_int err = 0;

    cl_kernel kernel;
    size_t global = n * k, local, size_qb;
    bool dequant;
    cl_block_q5_0* cl_host_b;

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
        size_qb = global * (sizeof(ggml_fp16_t) + local) / 16;
        break;
    case GGML_TYPE_Q5_0:
        dequant = true;
        kernel = kernel_q5_0;
        local = 16;
        // For some reason OpenCL seems to be incapable of working with structs of size 22.
        // 20 and 24 bytes are fine. Workaround to do the fp16 to fp32 step on CPU...
        // TODO Find the reason, fix and remove workaround.
        const block_q5_0* b = (const block_q5_0*) host_b;
        cl_host_b = (cl_block_q5_0*) malloc(sizeof(cl_block_q5_0) * global / 32);
        for (size_t i = 0; i < global / 32; i++) {
            cl_host_b[i].d = ggml_fp16_to_fp32(b[i].d);
            memcpy(&cl_host_b[i].qh, b[i].qh, sizeof(uint32_t));
            memcpy(&cl_host_b[i].qs, b[i].qs, QK5_0 / 2);
        }
        host_b = (const float*) cl_host_b;
        size_qb = global * (sizeof(float) + sizeof(uint32_t) + local) / 32;
        break;
    case GGML_TYPE_Q5_1:
        dequant = true;
        kernel = kernel_q5_1;
        local = 16;
        size_qb = global * (sizeof(ggml_fp16_t) * 2 + sizeof(uint32_t) + local) / 32;
        break;
    case GGML_TYPE_Q8_0:
        dequant = true;
        kernel = kernel_q8_0;
        local = 32;
        size_qb = global * (sizeof(float) + local) / 32;
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

    cl_event ev_a, ev_qb, ev_b;

    if (dequant) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
        CL_CHECK(err, "clSetKernelArg");
        err = clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_FALSE, 0, size_qb, host_b, 0, NULL, &ev_qb);
        CL_CHECK(err, "clEnqueueWriteBuffer qb");
    } else {
        err = clEnqueueWriteBuffer(queue, cl_buffer_b, CL_FALSE, 0, size_b, host_b, 0, NULL, &ev_b);
        CL_CHECK(err, "clEnqueueWriteBuffer b");
    }

    err = clEnqueueWriteBuffer(queue, cl_buffer_a, CL_FALSE, 0, size_a, host_a, 0, NULL, &ev_a);
    CL_CHECK(err, "clEnqueueWriteBuffer a");
    if (dequant) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, &ev_qb, &ev_b);
        CL_CHECK(err, "clEnqueueNDRangeKernel");
        clReleaseEvent(ev_qb);
    }
    clWaitForEvents(1, &ev_a);
    clWaitForEvents(1, &ev_b);
    clReleaseEvent(ev_a);
    clReleaseEvent(ev_b);

    cl_event ev_sgemm;
    CLBlastStatusCode status = CLBlastSgemm((CLBlastLayout)order,
                                            (CLBlastTranspose)trans_a, (CLBlastTranspose)trans_b,
                                            m, n, k,
                                            alpha,
                                            cl_buffer_a, 0, lda,
                                            cl_buffer_b, 0, ldb,
                                            beta,
                                            cl_buffer_c, 0, ldc,
                                            &queue, &ev_sgemm);

    if (status != CLBlastSuccess) {
        fprintf(stderr, "Error: CLBlast SGEMM %d\n", status);
        abort();
    }

    cl_event ev_c;
    clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, host_c, 1, &ev_sgemm, &ev_c);

    // Wait for completion
    clWaitForEvents(1, &ev_c);
    clReleaseEvent(ev_sgemm);
    clReleaseEvent(ev_c);
    if (btype == GGML_TYPE_Q5_0) {
        free((void*) cl_host_b);
    }
}
