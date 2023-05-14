#include "ggml-opencl.h"

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"

#define MULTILINE_QUOTE(...) #__VA_ARGS__
static const char * program_source = MULTILINE_QUOTE(

typedef uchar uint8_t;
typedef int int32_t;
typedef uint uint32_t;

struct block_q4_0
{
    float d;
    uint8_t qs[16]; /* QK4_0 / 2 */
};

struct block_q4_1
{
    float d;
    float m;
    uint8_t qs[16]; /* QK4_1 / 2 */
};

struct __attribute__ ((packed)) block_q5_0
{
    half d;
    uint32_t qh;
    uint8_t qs[16]; /* QK5_0 / 2 */
};

struct block_q5_1
{
    half d;
    half m;
    uint32_t qh;
    uint8_t qs[16]; /* QK5_1 / 2 */
};

struct block_q8_0
{
    float d;
    uint8_t qs[16]; /* QK8_0 / 2 */
};


__kernel void dequantize_row_q4_0(__global struct block_q4_0* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK4_0 */
    const uint j = get_local_id(0);

    const float d = x[i].d;

    const int x0 = (x[i].qs[j] & 0xf) - 8;
    const int x1 = (x[i].qs[j] >>  4) - 8;

    y[i*qk + j + 0   ] = x0*d;
    y[i*qk + j + qk/2] = x1*d;
}

__kernel void dequantize_row_q4_1(__global struct block_q4_1* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK4_1 */
    const uint j = get_local_id(0);

    const float d = x[i].d;
    const float m = x[i].m;

    const int x0 = (x[i].qs[j] & 0xf);
    const int x1 = (x[i].qs[j] >>  4);

    y[i*qk + j + 0   ] = x0*d + m;
    y[i*qk + j + qk/2] = x1*d + m;
}

__kernel void dequantize_row_q5_0(__global struct block_q5_0* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK5_0 */
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);

    uint32_t qh = x[i].qh;

    const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

    const int32_t x0 = ((x[i].qs[j] & 0xf) | xh_0) - 16;
    const int32_t x1 = ((x[i].qs[j] >>  4) | xh_1) - 16;

    y[i*qk + j + 0   ] = x0*d;
    y[i*qk + j + qk/2] = x1*d;
}

__kernel void dequantize_row_q5_1(__global struct block_q5_1* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK5_1 */
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);
    const float m = vload_half(0, (__global half*) &x[i].m);

    uint32_t qh = x[i].qh;

    const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

    const int x0 = (x[i].qs[j] & 0xf) | xh_0;
    const int x1 = (x[i].qs[j] >>  4) | xh_1;

    y[i*qk + j + 0   ] = x0*d + m;
    y[i*qk + j + qk/2] = x1*d + m;
}

__kernel void dequantize_row_q8_0(__global struct block_q8_0* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK8_0 */
    const uint j = get_local_id(0);

    const float d = x[i].d;
    y[i*qk + j] = x[i].qs[j]*d;
}

);

#define CL_CHECK(err, name)                                                                     \
    do {                                                                                        \
        cl_int err_ = (err);                                                                    \
        if (err_ != CL_SUCCESS) {                                                               \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n", name, err_, __FILE__, __LINE__);   \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_bool out_of_order;
static cl_program program;
static cl_kernel kernel_q4_0, kernel_q4_1, kernel_q5_0, kernel_q5_1, kernel_q8_0;
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

    enum { NPLAT = 16, NDEV = 16 };

    char text_buffer[1024] = {0};

    platform = NULL;
    char * GGML_OPENCL_PLATFORM = getenv("GGML_OPENCL_PLATFORM");
    if (GGML_OPENCL_PLATFORM != NULL) {
        cl_platform_id platforms[NPLAT];
        cl_uint num_platforms;
        err = clGetPlatformIDs(NPLAT, platforms, &num_platforms);
        CL_CHECK(err, "clGetPlatformIDs");

        unsigned plat_num;
        if (sscanf(GGML_OPENCL_PLATFORM, " %u", &plat_num) == 1) {
            if (plat_num >= num_platforms) {
                fprintf(stderr, "ggml_opencl: There is no platform %d\n", plat_num);
                exit(1);
            } else {
                platform = platforms[plat_num];
                clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(text_buffer), &text_buffer, NULL);
            }
        } else {
            for (unsigned i = 0; i < num_platforms; i++) {
                clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(text_buffer), &text_buffer, NULL);
                if (strstr(text_buffer, GGML_OPENCL_PLATFORM) != NULL) {
                    platform = platforms[i];
                    break;
                }
            }
        }
        if (platform == NULL) {
            fprintf(stderr, "ggml_opencl: no platform matching '%s' was found.\n", GGML_OPENCL_PLATFORM);
            exit(1);
        } else {
            fprintf(stderr, "ggml_opencl: selecting platform: '%s'\n", text_buffer);
        }
    }

    text_buffer[0] = 0;
    device = NULL;
    char * GGML_OPENCL_DEVICE = getenv("GGML_OPENCL_DEVICE");
    if (GGML_OPENCL_DEVICE != NULL) {
        cl_device_id devices[16];
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, NDEV, devices, &num_devices);

        unsigned dev_num;
        if (sscanf(GGML_OPENCL_DEVICE, " %u", &dev_num) == 1) {
            if (dev_num >= num_devices) {
                fprintf(stderr, "ggml_opencl: There is no device %d\n", dev_num);
                exit(1);
            } else {
                device = devices[dev_num];
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(text_buffer), &text_buffer, NULL);
            }
        } else {
            for (unsigned i = 0; i < num_devices; i++) {
                clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(text_buffer), &text_buffer, NULL);
                if (strstr(text_buffer, GGML_OPENCL_DEVICE) != NULL) {
                    device = devices[i];
                    break;
                }
            }
        }
        if (device == NULL) {
            fprintf(stderr, "ggml_opencl: no device matching '%s' was found.\n", GGML_OPENCL_DEVICE);
            exit(1);
        } else {
            fprintf(stderr, "ggml_opencl: selecting device: '%s'\n", text_buffer);
        }
    }

    cl_context_properties *properties = platform == NULL ? NULL : (cl_context_properties[]){
        (intptr_t)CL_CONTEXT_PLATFORM, (intptr_t)platform, 0
    };

    if (device != NULL) {
        context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
        CL_CHECK(err, "clCreateContext");
    } else {
        context = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
        if (err == CL_DEVICE_NOT_AVAILABLE || err == CL_DEVICE_NOT_FOUND) {
            context = clCreateContextFromType(properties, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, &err);
            if (err == CL_DEVICE_NOT_AVAILABLE || err == CL_DEVICE_NOT_FOUND) {
                context = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, NULL, NULL, &err);
            }
        }
        CL_CHECK(err, "clCreateContextFromType");
    }

    if (device == NULL) {
        err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(&device), &device, NULL);
        CL_CHECK(err, "clGetContextInfo");
        if (platform == NULL) {
            err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(&platform), &platform, NULL);
            CL_CHECK(err, "clGetDeviceInfo");
        }
    }

    if (platform != NULL) {
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(text_buffer), &text_buffer, NULL);
        fprintf(stderr, "ggml_opencl: using platform: '%s'\n", text_buffer);
    }
    if (device != NULL) {
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(text_buffer), &text_buffer, NULL);
        fprintf(stderr, "ggml_opencl: using device: '%s'\n", text_buffer);
    }

    out_of_order = CL_TRUE;
    queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (err == CL_INVALID_PROPERTY || err == CL_INVALID_VALUE) {
        out_of_order = CL_FALSE;
        queue = clCreateCommandQueue(context, device, 0, &err);
    }
    CL_CHECK(err, "clCreateCommandQueue");

    program = build_program_from_source(context, device, program_source);

    // Prepare dequantize kernels
    kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_1 = clCreateKernel(program, "dequantize_row_q4_1", &err);
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
    case GGML_TYPE_Q5_0:
        dequant = true;
        kernel = kernel_q5_0;
        local = 16;
        size_qb = global * (sizeof(ggml_fp16_t) + sizeof(uint32_t) + local) / 32;
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
}
