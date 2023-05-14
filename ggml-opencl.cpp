#include "ggml-opencl.h"

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"

#define MULTILINE_QUOTE(...) #__VA_ARGS__
static const char * program_source = MULTILINE_QUOTE(

typedef char int8_t;
typedef uchar uint8_t;
typedef int int32_t;
typedef uint uint32_t;

struct __attribute__ ((packed)) block_q4_0
{
    half d;
    uint8_t qs[16]; /* QK4_0 / 2 */
};

struct __attribute__ ((packed)) block_q4_1
{
    half d;
    half m;
    uint8_t qs[16]; /* QK4_1 / 2 */
};

struct __attribute__ ((packed)) block_q5_0
{
    half d;
    uint32_t qh;
    uint8_t qs[16]; /* QK5_0 / 2 */
};

struct __attribute__ ((packed)) block_q5_1
{
    half d;
    half m;
    uint32_t qh;
    uint8_t qs[16]; /* QK5_1 / 2 */
};

struct __attribute__ ((packed)) block_q8_0
{
    half d;
    int8_t qs[32]; /* QK8_0 */
};


__kernel void dequantize_row_q4_0(__global struct block_q4_0* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK4_0 */
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);

    const int x0 = (x[i].qs[j] & 0xf) - 8;
    const int x1 = (x[i].qs[j] >>  4) - 8;

    y[i*32 + j + 0 ] = x0*d;
    y[i*32 + j + 16] = x1*d;
}

__kernel void dequantize_row_q4_1(__global struct block_q4_1* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK4_1 */
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);
    const float m = vload_half(0, (__global half*) &x[i].m);

    const int x0 = (x[i].qs[j] & 0xf);
    const int x1 = (x[i].qs[j] >>  4);

    y[i*32 + j + 0 ] = x0*d + m;
    y[i*32 + j + 16] = x1*d + m;
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

    y[i*32 + j + 0 ] = x0*d;
    y[i*32 + j + 16] = x1*d;
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

    y[i*32 + j + 0 ] = x0*d + m;
    y[i*32 + j + 16] = x1*d + m;
}

__kernel void dequantize_row_q8_0(__global struct block_q8_0* x, __global float* y) {
    const uint i = get_global_id(0) / 32; /* QK8_0 */
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);
    y[i*32 + j] = x[i].qs[j]*d;
}

);

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
        }                                                           \
    } while (0)

#define CLBLAST_CHECK(err)                                          \
    do {                                                            \
        CLBlastStatusCode err_ = (err);                             \
        if (err_ != CLBlastSuccess) {                               \
            fprintf(stderr, "ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
        }                                                           \
    } while (0)

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
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

    struct cl_device;
    struct cl_platform {
        cl_platform_id id;
        unsigned number;
        char name[128];
        char vendor[128];
        struct cl_device * devices;
        unsigned n_devices;
        struct cl_device * default_device;
    };

    struct cl_device {
        struct cl_platform * platform;
        cl_device_id id;
        unsigned number;
        cl_device_type type;
        char name[128];
    };

    enum { NPLAT = 16, NDEV = 16 };

    struct cl_platform platforms[NPLAT];
    unsigned n_platforms = 0;
    struct cl_device devices[NDEV];
    unsigned n_devices = 0;
    struct cl_device * default_device = NULL;

    platform = NULL;
    device = NULL;

    cl_platform_id platform_ids[NPLAT];
    CL_CHECK(clGetPlatformIDs(NPLAT, platform_ids, &n_platforms));

    for (unsigned i = 0; i < n_platforms; i++) {
        struct cl_platform * p = &platforms[i];
        p->number = i;
        p->id = platform_ids[i];
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_NAME, sizeof(p->name), &p->name, NULL));
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_VENDOR, sizeof(p->vendor), &p->vendor, NULL));

        cl_device_id device_ids[NDEV];
        cl_int clGetDeviceIDsError = clGetDeviceIDs(p->id, CL_DEVICE_TYPE_ALL, NDEV, device_ids, &p->n_devices);
        if (clGetDeviceIDsError == CL_DEVICE_NOT_FOUND) {
            p->n_devices = 0;
        } else {
            CL_CHECK(clGetDeviceIDsError);
        }
        p->devices = p->n_devices > 0 ? &devices[n_devices] : NULL;
        p->default_device = NULL;

        for (unsigned j = 0; j < p->n_devices; j++) {
            struct cl_device * d = &devices[n_devices];
            d->number = n_devices++;
            d->id = device_ids[j];
            d->platform = p;
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_NAME, sizeof(d->name), &d->name, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_TYPE, sizeof(d->type), &d->type, NULL));

            if (p->default_device == NULL && d->type == CL_DEVICE_TYPE_GPU) {
                p->default_device = d;
            }
        }

        if (default_device == NULL && p->default_device != NULL) {
            default_device = p->default_device;
        }
    }

    if (n_devices == 0) {
        fprintf(stderr, "ggml_opencl: could find any OpenCL devices.\n");
        exit(1);
    }

    char * user_platform_string = getenv("GGML_OPENCL_PLATFORM");
    char * user_device_string = getenv("GGML_OPENCL_DEVICE");
    int user_platform_number = -1;
    int user_device_number = -1;

    unsigned n;
    if (user_platform_string != NULL && sscanf(user_platform_string, " %u", &n) == 1 && n < n_platforms) {
        user_platform_number = (int)n;
    }
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1 && n < n_devices) {
        user_device_number = (int)n;
    }

    struct cl_device * selected_devices = devices;
    unsigned n_selected_devices = n_devices;

    if (user_platform_number == -1 && user_platform_string != NULL && user_platform_string[0] != 0) {
        for (unsigned i = 0; i < n_platforms; i++) {
            struct cl_platform * p = &platforms[i];
            if (strstr(p->name, user_platform_string) != NULL ||
                strstr(p->vendor, user_platform_string) != NULL) {
                user_platform_number = (int)i;
                break;
            }
        }
        if (user_platform_number == -1) {
            fprintf(stderr, "ggml_opencl: no platform matching '%s' was found.\n", user_platform_string);
            exit(1);
        }
    }
    if (user_platform_number != -1) {
        struct cl_platform * p = &platforms[user_platform_number];
        selected_devices = p->devices;
        n_selected_devices = p->n_devices;
        default_device = p->default_device;
        if (n_selected_devices == 0) {
            fprintf(stderr, "ggml_opencl: selected platform '%s' does not have any devices.\n", p->name);
            exit(1);
        }
    }

    if (user_device_number == -1 && user_device_string != NULL && user_device_string[0] != 0) {
        for (unsigned i = 0; i < n_selected_devices; i++) {
            struct cl_device * d = &selected_devices[i];
            if (strstr(d->name, user_device_string) != NULL) {
                user_device_number = d->number;
                break;
            }
        }
        if (user_device_number == -1) {
            fprintf(stderr, "ggml_opencl: no device matching '%s' was found.\n", user_device_string);
            exit(1);
        }
    }
    if (user_device_number != -1) {
        selected_devices = &devices[user_device_number];
        n_selected_devices = 1;
        default_device = &selected_devices[0];
    }

    GGML_ASSERT(n_selected_devices > 0);

    if (default_device == NULL) {
        default_device = &selected_devices[0];
    }

    fprintf(stderr, "ggml_opencl: selecting platform: '%s'\n", default_device->platform->name);
    fprintf(stderr, "ggml_opencl: selecting device: '%s'\n", default_device->name);
    if (default_device->type != CL_DEVICE_TYPE_GPU) {
        fprintf(stderr, "ggml_opencl: warning, not a GPU: '%s'.\n", default_device->name);
    }

    platform = default_device->platform->id;
    device = default_device->id;

    cl_context_properties properties[] = {
        (intptr_t)CL_CONTEXT_PLATFORM, (intptr_t)platform, 0
    };

    CL_CHECK((context = clCreateContext(properties, 1, &device, NULL, NULL, &err), err));

    CL_CHECK((queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err),
        (err != CL_INVALID_PROPERTY && err != CL_INVALID_VALUE ? err :
        (queue = clCreateCommandQueue(context, device, 0, &err), err)
    )));

    program = build_program_from_source(context, device, program_source);

    // Prepare dequantize kernels
    CL_CHECK((kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err), err));
    CL_CHECK((kernel_q4_1 = clCreateKernel(program, "dequantize_row_q4_1", &err), err));
    CL_CHECK((kernel_q5_0 = clCreateKernel(program, "dequantize_row_q5_0", &err), err));
    CL_CHECK((kernel_q5_1 = clCreateKernel(program, "dequantize_row_q5_1", &err), err));
    CL_CHECK((kernel_q8_0 = clCreateKernel(program, "dequantize_row_q8_0", &err), err));
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
    CL_CHECK((*buf = clCreateBuffer(context, flags, req_size, NULL, &err), err));
    *cur_size = req_size;
}

void ggml_cl_sgemm_wrapper(
        const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b,
        const int m, const int n, const int k,
        const float alpha, const void *host_a, const int lda,
        const float *host_b, const int ldb, const float beta,
        float *host_c, const int ldc, const int btype) {

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
        size_qb = global * (sizeof(ggml_fp16_t) + local) / 32;
        break;
    case GGML_TYPE_Q4_1:
        dequant = true;
        kernel = kernel_q4_1;
        local = 16;
        size_qb = global * (sizeof(ggml_fp16_t) * 2 + local) / 32;
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
        size_qb = global * (sizeof(ggml_fp16_t) + local) / 32;
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
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b));
        CL_CHECK(clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_FALSE, 0, size_qb, host_b, 0, NULL, &ev_qb));
    } else {
        CL_CHECK(clEnqueueWriteBuffer(queue, cl_buffer_b, CL_FALSE, 0, size_b, host_b, 0, NULL, &ev_b));
    }

    CL_CHECK(clEnqueueWriteBuffer(queue, cl_buffer_a, CL_FALSE, 0, size_a, host_a, 0, NULL, &ev_a));
    if (dequant) {
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, &ev_qb, &ev_b));
        CL_CHECK(clReleaseEvent(ev_qb));
    }
    CL_CHECK(clWaitForEvents(1, &ev_a));
    CL_CHECK(clWaitForEvents(1, &ev_b));
    CL_CHECK(clReleaseEvent(ev_a));
    CL_CHECK(clReleaseEvent(ev_b));

    cl_event ev_sgemm;
    CLBLAST_CHECK(CLBlastSgemm(
        (CLBlastLayout)order,
        (CLBlastTranspose)trans_a, (CLBlastTranspose)trans_b,
        m, n, k,
        alpha,
        cl_buffer_a, 0, lda,
        cl_buffer_b, 0, ldb,
        beta,
        cl_buffer_c, 0, ldc,
        &queue, &ev_sgemm));

    cl_event ev_c;
    CL_CHECK(clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, host_c, 1, &ev_sgemm, &ev_c));

    // Wait for completion
    CL_CHECK(clWaitForEvents(1, &ev_c));
    CL_CHECK(clReleaseEvent(ev_sgemm));
    CL_CHECK(clReleaseEvent(ev_c));
}
