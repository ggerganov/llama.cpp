#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct ggml_v2_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_v2_context * ctx = ggml_v2_init(params);
//
//       struct ggml_v2_tensor * x = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, 1);
//
//       ggml_v2_set_param(ctx, x); // x is an input variable
//
//       struct ggml_v2_tensor * a  = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, 1);
//       struct ggml_v2_tensor * b  = ggml_v2_new_tensor_1d(ctx, GGML_V2_TYPE_F32, 1);
//       struct ggml_v2_tensor * x2 = ggml_v2_mul(ctx, x, x);
//       struct ggml_v2_tensor * f  = ggml_v2_add(ctx, ggml_v2_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct ggml_v2_cgraph gf = ggml_v2_build_forward(f);
//
//       // set the input variable and parameter values
//       ggml_v2_set_f32(x, 2.0f);
//       ggml_v2_set_f32(a, 3.0f);
//       ggml_v2_set_f32(b, 4.0f);
//
//       ggml_v2_graph_compute(ctx0, &gf);
//
//       printf("f = %f\n", ggml_v2_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_v2_graph_compute() function.
//
// The ggml_v2_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_v2_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_v2_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_v2_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_v2_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - ggml_v2_permute()
//   - ggml_v2_conv_1d_1s()
//   - ggml_v2_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct ggml_v2_tensor)
//
// The tensors are stored in memory via the ggml_v2_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_v2_tensor * c = ggml_v2_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_v2_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       struct ggml_v2_tensor * a = ggml_v2_new_tensor_2d(ctx, GGML_V2_TYPE_F32, 2, 3);
//
//       // a[1, 2] = 1.0f;
//       *(float *) ((char *) a->data + 2*a->nb[1] + 1*a->nb[0]) = 1.0f;
//
//       // a[2, 0] = 2.0f;
//       *(float *) ((char *) a->data + 0*a->nb[1] + 2*a->nb[0]) = 2.0f;
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as ggml_v2_get_f32_1d() and ggml_v2_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_v2_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef GGML_V2_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_V2_BUILD
#            define GGML_V2_API __declspec(dllexport)
#        else
#            define GGML_V2_API __declspec(dllimport)
#        endif
#    else
#        define GGML_V2_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_V2_API
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_V2_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_V2_FILE_VERSION 1

#define GGML_V2_QNT_VERSION        1    // bump this on quantization format changes
#define GGML_V2_QNT_VERSION_FACTOR 1000 // do not change this

#define GGML_V2_MAX_DIMS          4
#define GGML_V2_MAX_NODES         4096
#define GGML_V2_MAX_PARAMS        256
#define GGML_V2_MAX_CONTEXTS      64
#define GGML_V2_MAX_OPT           4
#define GGML_V2_DEFAULT_N_THREADS 4

#define GGML_V2_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_V2_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 ggml_v2_fp16_t;
#else
    typedef uint16_t ggml_v2_fp16_t;
#endif

    // convert FP16 <-> FP32
    GGML_V2_API float       ggml_v2_fp16_to_fp32(ggml_v2_fp16_t x);
    GGML_V2_API ggml_v2_fp16_t ggml_v2_fp32_to_fp16(float x);

    GGML_V2_API void ggml_v2_fp16_to_fp32_row(const ggml_v2_fp16_t * x, float * y, size_t n);
    GGML_V2_API void ggml_v2_fp32_to_fp16_row(const float * x, ggml_v2_fp16_t * y, size_t n);

    struct ggml_v2_object;
    struct ggml_v2_context;

    enum ggml_v2_type {
        GGML_V2_TYPE_F32  = 0,
        GGML_V2_TYPE_F16  = 1,
        GGML_V2_TYPE_Q4_0 = 2,
        GGML_V2_TYPE_Q4_1 = 3,
        GGML_V2_TYPE_Q4_2 = 4, //support has been removed
        GGML_V2_TYPE_Q4_3 = 5, //support has been removed
        GGML_V2_TYPE_Q5_0 = 6,
        GGML_V2_TYPE_Q5_1 = 7,
        GGML_V2_TYPE_Q8_0 = 8,
        GGML_V2_TYPE_Q8_1 = 9,
        GGML_V2_TYPE_I8,
        GGML_V2_TYPE_I16,
        GGML_V2_TYPE_I32,
        GGML_V2_TYPE_Q8_1B = 13, //legacy q8_1
        GGML_V2_TYPE_COUNT,
    };

    enum ggml_v2_backend {
        GGML_V2_BACKEND_CPU = 0,
        GGML_V2_BACKEND_CUDA = 1,
        GGML_V2_BACKEND_CL = 2,
    };

    // model file types
    enum ggml_v2_ftype {
        GGML_V2_FTYPE_UNKNOWN     = -1,
        GGML_V2_FTYPE_ALL_F32     = 0,
        GGML_V2_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_V2_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q4_3 = 6,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        GGML_V2_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    };

    // available tensor operations:
    enum ggml_v2_op {
        GGML_V2_OP_NONE = 0,

        GGML_V2_OP_DUP,
        GGML_V2_OP_ADD,
        GGML_V2_OP_ADD1,
        GGML_V2_OP_ACC,
        GGML_V2_OP_SUB,
        GGML_V2_OP_MUL,
        GGML_V2_OP_DIV,
        GGML_V2_OP_SQR,
        GGML_V2_OP_SQRT,
        GGML_V2_OP_LOG,
        GGML_V2_OP_SUM,
        GGML_V2_OP_SUM_ROWS,
        GGML_V2_OP_MEAN,
        GGML_V2_OP_REPEAT,
        GGML_V2_OP_ABS,
        GGML_V2_OP_SGN,
        GGML_V2_OP_NEG,
        GGML_V2_OP_STEP,
        GGML_V2_OP_RELU,
        GGML_V2_OP_GELU,
        GGML_V2_OP_SILU,
        GGML_V2_OP_SILU_BACK,
        GGML_V2_OP_NORM, // normalize
        GGML_V2_OP_RMS_NORM,
        GGML_V2_OP_RMS_NORM_BACK,

        GGML_V2_OP_MUL_MAT,

        GGML_V2_OP_SCALE,
        GGML_V2_OP_SET,
        GGML_V2_OP_CPY,
        GGML_V2_OP_CONT,
        GGML_V2_OP_RESHAPE,
        GGML_V2_OP_VIEW,
        GGML_V2_OP_PERMUTE,
        GGML_V2_OP_TRANSPOSE,
        GGML_V2_OP_GET_ROWS,
        GGML_V2_OP_GET_ROWS_BACK,
        GGML_V2_OP_DIAG,
        GGML_V2_OP_DIAG_MASK_INF,
        GGML_V2_OP_DIAG_MASK_ZERO,
        GGML_V2_OP_SOFT_MAX,
        GGML_V2_OP_ROPE,
        GGML_V2_OP_ROPE_BACK,
        GGML_V2_OP_ALIBI,
        GGML_V2_OP_CONV_1D_1S,
        GGML_V2_OP_CONV_1D_2S,

        GGML_V2_OP_FLASH_ATTN,
        GGML_V2_OP_FLASH_FF,

        GGML_V2_OP_MAP_UNARY,
        GGML_V2_OP_MAP_BINARY,

        GGML_V2_OP_COUNT,
    };


    // ggml object
    struct ggml_v2_object {
        size_t offs;
        size_t size;

        struct ggml_v2_object * next;

        char padding[8];
    };

    static const size_t GGML_V2_OBJECT_SIZE = sizeof(struct ggml_v2_object);

    // n-dimensional tensor
    struct ggml_v2_tensor {
        enum ggml_v2_type    type;
        enum ggml_v2_backend backend;

        int     n_dims;
        int64_t ne[GGML_V2_MAX_DIMS]; // number of elements
        size_t  nb[GGML_V2_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_v2_op op;

        bool is_param;

        struct ggml_v2_tensor * grad;
        struct ggml_v2_tensor * src0;
        struct ggml_v2_tensor * src1;
        struct ggml_v2_tensor * opt[GGML_V2_MAX_OPT];

        // thread scheduling
        int n_tasks;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        void * data;

        char name[32];

        char padding[16];
    };

    // computation graph
    struct ggml_v2_cgraph {
        int n_nodes;
        int n_leafs;
        int n_threads;

        size_t work_size;
        struct ggml_v2_tensor * work;

        struct ggml_v2_tensor * nodes[GGML_V2_MAX_NODES];
        struct ggml_v2_tensor * grads[GGML_V2_MAX_NODES];
        struct ggml_v2_tensor * leafs[GGML_V2_MAX_NODES];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    // scratch buffer
    struct ggml_v2_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct ggml_v2_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };

    // misc

    GGML_V2_API void    ggml_v2_time_init(void); // call this once at the beginning of the program
    GGML_V2_API int64_t ggml_v2_time_ms(void);
    GGML_V2_API int64_t ggml_v2_time_us(void);
    GGML_V2_API int64_t ggml_v2_cycles(void);
    GGML_V2_API int64_t ggml_v2_cycles_per_ms(void);

    GGML_V2_API void    ggml_v2_print_object (const struct ggml_v2_object * obj);
    GGML_V2_API void    ggml_v2_print_objects(const struct ggml_v2_context * ctx);

    GGML_V2_API int64_t ggml_v2_nelements(const struct ggml_v2_tensor * tensor);
    GGML_V2_API size_t  ggml_v2_nbytes   (const struct ggml_v2_tensor * tensor);

    GGML_V2_API int     ggml_v2_blck_size (enum ggml_v2_type type);
    GGML_V2_API size_t  ggml_v2_type_size (enum ggml_v2_type type); // size in bytes for all elements in a block
    GGML_V2_API float   ggml_v2_type_sizef(enum ggml_v2_type type); // ggml_v2_type_size()/ggml_v2_blck_size() as float

    GGML_V2_API const char * ggml_v2_type_name(enum ggml_v2_type type);

    GGML_V2_API size_t  ggml_v2_element_size(const struct ggml_v2_tensor * tensor);

    GGML_V2_API bool    ggml_v2_is_quantized(enum ggml_v2_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    GGML_V2_API enum ggml_v2_type ggml_v2_ftype_to_ggml_v2_type(enum ggml_v2_ftype ftype);

    // main

    GGML_V2_API struct ggml_v2_context * ggml_v2_init(struct ggml_v2_init_params params);
    GGML_V2_API void    ggml_v2_free(struct ggml_v2_context * ctx);

    GGML_V2_API size_t  ggml_v2_used_mem(const struct ggml_v2_context * ctx);

    GGML_V2_API size_t  ggml_v2_set_scratch(struct ggml_v2_context * ctx, struct ggml_v2_scratch scratch);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_tensor(
            struct ggml_v2_context * ctx,
            enum   ggml_v2_type type,
            int    n_dims,
            const int64_t *ne);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_tensor_1d(
            struct ggml_v2_context * ctx,
            enum   ggml_v2_type type,
            int64_t ne0);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_tensor_2d(
            struct ggml_v2_context * ctx,
            enum   ggml_v2_type type,
            int64_t ne0,
            int64_t ne1);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_tensor_3d(
            struct ggml_v2_context * ctx,
            enum   ggml_v2_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_tensor_4d(
            struct ggml_v2_context * ctx,
            enum   ggml_v2_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_i32(struct ggml_v2_context * ctx, int32_t value);
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_new_f32(struct ggml_v2_context * ctx, float value);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_dup_tensor (struct ggml_v2_context * ctx, const struct ggml_v2_tensor * src);
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_view_tensor(struct ggml_v2_context * ctx, const struct ggml_v2_tensor * src);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_zero(struct ggml_v2_tensor * tensor);
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_i32 (struct ggml_v2_tensor * tensor, int32_t value);
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_f32 (struct ggml_v2_tensor * tensor, float value);

    GGML_V2_API int32_t ggml_v2_get_i32_1d(const struct ggml_v2_tensor * tensor, int i);
    GGML_V2_API void    ggml_v2_set_i32_1d(const struct ggml_v2_tensor * tensor, int i, int32_t value);

    GGML_V2_API float   ggml_v2_get_f32_1d(const struct ggml_v2_tensor * tensor, int i);
    GGML_V2_API void    ggml_v2_set_f32_1d(const struct ggml_v2_tensor * tensor, int i, float value);

    GGML_V2_API void *  ggml_v2_get_data    (const struct ggml_v2_tensor * tensor);
    GGML_V2_API float * ggml_v2_get_data_f32(const struct ggml_v2_tensor * tensor);

    GGML_V2_API const char * ggml_v2_get_name(const struct ggml_v2_tensor * tensor);
    GGML_V2_API void         ggml_v2_set_name(struct ggml_v2_tensor * tensor, const char * name);

    //
    // operations on tensors with backpropagation
    //

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_dup(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_add(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_add_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_add1(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_acc(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_acc_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_sub(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_mul(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_div(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_sqr(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_sqrt(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_log(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_log_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // return scalar
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_sum(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_sum_rows(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // mean along rows
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_mean(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_repeat(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_abs(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_sgn(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_neg(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_step(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_relu(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // TODO: double-check this computation is correct
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_gelu(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_silu(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // a - x
    // b - dy
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_silu_back(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    // normalize along rows
    // TODO: eps is hardcoded to 1e-5 for now
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_norm(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_rms_norm(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // a - x
    // b - dy
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_rms_norm_back(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    // A: m rows, n columns
    // B: p rows, n columns (i.e. we transpose it internally)
    // result is m columns, p rows
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_mul_mat(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_scale(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    // in-place, returns view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_scale_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_1d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_1d_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_2d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_set_2d_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            size_t                nb1,
            size_t                offset);


    // a -> b, return view(b)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_cpy(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    // make contiguous
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_cont(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_reshape(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_reshape_1d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_reshape_2d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_reshape_3d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_reshape_4d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_view_1d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_view_2d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_view_3d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_view_4d(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_permute(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for ggml_v2_permute(ctx, a, 1, 0, 2, 3)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_transpose(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_get_rows(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_get_rows_back(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b,
            struct ggml_v2_tensor  * c);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_diag(
        struct ggml_v2_context     * ctx,
        struct ggml_v2_tensor      * a);

    // set elements above the diagonal to -INF
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_diag_mask_inf(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_diag_mask_inf_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_diag_mask_zero(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_diag_mask_zero_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_soft_max(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // in-place, returns view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_soft_max_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements
    // if mode & 2 == 1, GPT-NeoX style
    // TODO: avoid creating a new tensor every time
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_rope(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // in-place, returns view(a)
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_rope_inplace(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_rope_back(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // alibi position embedding
    // in-place, returns view(a)
    struct ggml_v2_tensor * ggml_v2_alibi(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            int                   n_past,
            int                   n_head);

    // padding = 1
    // TODO: we don't support extra parameters for now
    //       that's why we are hard-coding the stride, padding, and dilation
    //       not great ..
    GGML_V2_API struct ggml_v2_tensor * ggml_v2_conv_1d_1s(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_conv_1d_2s(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_flash_attn(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * q,
            struct ggml_v2_tensor  * k,
            struct ggml_v2_tensor  * v,
            bool                  masked);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_flash_ff(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor  * a,
            struct ggml_v2_tensor  * b0,
            struct ggml_v2_tensor  * b1,
            struct ggml_v2_tensor  * c0,
            struct ggml_v2_tensor  * c1);

    // Mapping operations
    typedef void (*ggml_v2_unary_op_f32_t)(const int, float *, const float *);
    typedef void (*ggml_v2_binary_op_f32_t)(const int, float *, const float *, const float *);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_map_unary_f32(
            struct ggml_v2_context        * ctx,
            struct ggml_v2_tensor         * a,
                   ggml_v2_unary_op_f32_t   fun);

    GGML_V2_API struct ggml_v2_tensor * ggml_v2_map_binary_f32(
            struct ggml_v2_context         * ctx,
            struct ggml_v2_tensor          * a,
            struct ggml_v2_tensor          * b,
                   ggml_v2_binary_op_f32_t   fun);

    //
    // automatic differentiation
    //

    GGML_V2_API void ggml_v2_set_param(
            struct ggml_v2_context * ctx,
            struct ggml_v2_tensor * tensor);

    GGML_V2_API void ggml_v2_build_forward_expand(struct ggml_v2_cgraph * cgraph, struct ggml_v2_tensor * tensor);

    GGML_V2_API struct ggml_v2_cgraph ggml_v2_build_forward (struct ggml_v2_tensor * tensor);
    GGML_V2_API struct ggml_v2_cgraph ggml_v2_build_backward(struct ggml_v2_context * ctx, struct ggml_v2_cgraph * gf, bool keep);

    GGML_V2_API void ggml_v2_graph_compute(struct ggml_v2_context * ctx, struct ggml_v2_cgraph * cgraph);
    GGML_V2_API void ggml_v2_graph_reset  (struct ggml_v2_cgraph * cgraph);

    // print info and performance information for the graph
    GGML_V2_API void ggml_v2_graph_print(const struct ggml_v2_cgraph * cgraph);

    // dump the graph into a file using the dot format
    GGML_V2_API void ggml_v2_graph_dump_dot(const struct ggml_v2_cgraph * gb, const struct ggml_v2_cgraph * gf, const char * filename);

    //
    // optimization
    //

    // optimization methods
    enum ggml_v2_opt_type {
        GGML_V2_OPT_ADAM,
        GGML_V2_OPT_LBFGS,
    };

    // linesearch methods
    enum ggml_v2_linesearch {
        GGML_V2_LINESEARCH_DEFAULT = 1,

        GGML_V2_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        GGML_V2_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        GGML_V2_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum ggml_v2_opt_result {
        GGML_V2_OPT_OK = 0,
        GGML_V2_OPT_DID_NOT_CONVERGE,
        GGML_V2_OPT_NO_CONTEXT,
        GGML_V2_OPT_INVALID_WOLFE,
        GGML_V2_OPT_FAIL,

        GGML_V2_LINESEARCH_FAIL = -128,
        GGML_V2_LINESEARCH_MINIMUM_STEP,
        GGML_V2_LINESEARCH_MAXIMUM_STEP,
        GGML_V2_LINESEARCH_MAXIMUM_ITERATIONS,
        GGML_V2_LINESEARCH_INVALID_PARAMETERS,
    };

    // optimization parameters
    //
    //   see ggml.c (ggml_v2_opt_default_params) for default values
    //
    struct ggml_v2_opt_params {
        enum ggml_v2_opt_type type;

        int n_threads;

        // delta-based convergence test
        //
        //   if past == 0 - disabled
        //   if past > 0:
        //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
        //
        int past;
        float delta;

        // maximum number of iterations without improvement
        //
        //   if 0 - disabled
        //   if > 0:
        //     assume convergence if no cost improvement in this number of iterations
        //
        int max_no_improvement;

        bool print_forward_graph;
        bool print_backward_graph;

        // ADAM parameters
        struct {
            int n_iter;

            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float eps_f; // epsilon for convergence test
            float eps_g; // epsilon for convergence test
        } adam;

        // LBFGS parameters
        struct {
            int m; // number of corrections to approximate the inv. Hessian
            int n_iter;
            int max_linesearch;

            float eps;      // convergence tolerance
            float ftol;     // line search tolerance
            float wolfe;
            float min_step;
            float max_step;

            enum ggml_v2_linesearch linesearch;
        } lbfgs;
    };

    GGML_V2_API struct ggml_v2_opt_params ggml_v2_opt_default_params(enum ggml_v2_opt_type type);

    // optimize the function defined by the tensor f
    GGML_V2_API enum ggml_v2_opt_result ggml_v2_opt(
            struct ggml_v2_context * ctx,
            struct ggml_v2_opt_params params,
            struct ggml_v2_tensor * f);

    //
    // quantization
    //

    GGML_V2_API size_t ggml_v2_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);    
    GGML_V2_API size_t ggml_v2_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    GGML_V2_API size_t ggml_v2_quantize_q4_0_v2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q4_1_v2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q4_2_v2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q4_3_v2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q5_0_v2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q5_1_v2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_q8_0_v2(const float * src, void * dst, int n, int k, int64_t * hist);

    GGML_V2_API size_t ggml_v2_quantize_chunk(enum ggml_v2_type type, const float * src, void * dst, int start, int n, int64_t * hist);
    GGML_V2_API size_t ggml_v2_quantize_chunk_v2(enum ggml_v2_type type, const float * src, void * dst, int start, int n, int64_t * hist);
    //
    // system info
    //

    void SetQuantsUnshuffled(bool unshuffled);
    bool GetQuantsUnshuffled();

    GGML_V2_API int ggml_v2_cpu_has_avx        (void);
    GGML_V2_API int ggml_v2_cpu_has_avx2       (void);
    GGML_V2_API int ggml_v2_cpu_has_avx512     (void);
    GGML_V2_API int ggml_v2_cpu_has_avx512_vbmi(void);
    GGML_V2_API int ggml_v2_cpu_has_avx512_vnni(void);
    GGML_V2_API int ggml_v2_cpu_has_fma        (void);
    GGML_V2_API int ggml_v2_cpu_has_neon       (void);
    GGML_V2_API int ggml_v2_cpu_has_arm_fma    (void);
    GGML_V2_API int ggml_v2_cpu_has_f16c       (void);
    GGML_V2_API int ggml_v2_cpu_has_fp16_va    (void);
    GGML_V2_API int ggml_v2_cpu_has_wasm_simd  (void);
    GGML_V2_API int ggml_v2_cpu_has_blas       (void);
    GGML_V2_API int ggml_v2_cpu_has_cublas     (void);
    GGML_V2_API int ggml_v2_cpu_has_clblast    (void);
    GGML_V2_API int ggml_v2_cpu_has_gpublas    (void);
    GGML_V2_API int ggml_v2_cpu_has_sse3       (void);
    GGML_V2_API int ggml_v2_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
    // restrict not standard in C++
#define GGML_V2_RESTRICT
#else
#define GGML_V2_RESTRICT restrict
#endif
    typedef void (*dequantize_row_q_t)(const void * GGML_V2_RESTRICT x, float * GGML_V2_RESTRICT y, int k);
    typedef void (*quantize_row_q_t)  (const float * GGML_V2_RESTRICT x, void * GGML_V2_RESTRICT y, int k);
    typedef void (*vec_dot_q_t)       (const int n, float * GGML_V2_RESTRICT s, const void * GGML_V2_RESTRICT x, const void * GGML_V2_RESTRICT y);

    typedef struct {
        dequantize_row_q_t dequantize_row_q;
        quantize_row_q_t   quantize_row_q;
        quantize_row_q_t   quantize_row_q_reference;
        quantize_row_q_t   quantize_row_q_dot;
        vec_dot_q_t        vec_dot_q;
        enum ggml_v2_type     vec_dot_type;
    } quantize_fns_t2;

    quantize_fns_t2 ggml_v2_internal_get_quantize_fn(size_t i);

#ifdef  __cplusplus
}
#endif
