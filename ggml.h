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
//       struct ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_context * ctx = ggml_init(params);
//
//       struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//
//       ggml_set_param(ctx, x); // x is an input variable
//
//       struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
//       struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
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
//       struct ggml_cgraph gf = ggml_build_forward(f);
//
//       // set the input variable and parameter values
//       ggml_set_f32(x, 2.0f);
//       ggml_set_f32(a, 3.0f);
//       ggml_set_f32(b, 4.0f);
//
//       ggml_graph_compute(ctx0, &gf);
//
//       printf("f = %f\n", ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_graph_compute() function.
//
// The ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_init() function. This way
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
//   - ggml_permute()
//   - ggml_conv_1d_1s()
//   - ggml_conv_1d_2s()
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
// ## Tensor data (struct ggml_tensor)
//
// The tensors are stored in memory via the ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_tensor * c = ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
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
// Alternatively, there are helper functions, such as ggml_get_f32_1d() and ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_mul_mat)
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

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport)
#        else
#            define GGML_API __declspec(dllimport)
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_API
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_FILE_VERSION 1

#define GGML_MAX_DIMS          4
#define GGML_MAX_NODES         4096
#define GGML_MAX_PARAMS        16
#define GGML_MAX_CONTEXTS      64
#define GGML_MAX_OPT           4
#define GGML_DEFAULT_N_THREADS 4

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 ggml_fp16_t;
#else
    typedef uint16_t ggml_fp16_t;
#endif

    // convert FP16 <-> FP32
    GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t x);
    GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);

    GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, size_t n);
    GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, size_t n);

    struct ggml_object;
    struct ggml_context;

    enum ggml_type {
        GGML_TYPE_F32  = 0,
        GGML_TYPE_F16  = 1,
        GGML_TYPE_Q4_0 = 2,
        GGML_TYPE_Q4_1 = 3,
        GGML_TYPE_Q4_2 = 4,
        // GGML_TYPE_Q4_3 (5) support has been removed
        GGML_TYPE_Q5_0 = 6,
        GGML_TYPE_Q5_1 = 7,
        GGML_TYPE_Q8_0 = 8,
        GGML_TYPE_Q8_1 = 9,
        GGML_TYPE_I8,
        GGML_TYPE_I16,
        GGML_TYPE_I32,
        GGML_TYPE_COUNT,
    };

    // model file types
    enum ggml_ftype {
        GGML_FTYPE_UNKNOWN     = -1,
        GGML_FTYPE_ALL_F32     = 0,
        GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    };

    // available tensor operations:
    enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_SUM,
        GGML_OP_MEAN,
        GGML_OP_REPEAT,
        GGML_OP_ABS,
        GGML_OP_SGN,
        GGML_OP_NEG,
        GGML_OP_STEP,
        GGML_OP_RELU,
        GGML_OP_GELU,
        GGML_OP_SILU,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,

        GGML_OP_MUL_MAT,

        GGML_OP_SCALE,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_SOFT_MAX,
        GGML_OP_ROPE,
        GGML_OP_ALIBI,
        GGML_OP_CONV_1D_1S,
        GGML_OP_CONV_1D_2S,

        GGML_OP_FLASH_ATTN,
        GGML_OP_FLASH_FF,

        GGML_OP_MAP_UNARY,
        GGML_OP_MAP_BINARY,

        GGML_OP_COUNT,
    };


    // ggml object
    struct ggml_object {
        size_t offs;
        size_t size;

        struct ggml_object * next;

        char padding[8];
    };

    static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;

        int     n_dims;
        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = sizeof(type)
                                   // nb[1] = nb[0]   * ne[0] + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        bool is_param;

        struct ggml_tensor * grad;
        struct ggml_tensor * src0;
        struct ggml_tensor * src1;
        struct ggml_tensor * opt[GGML_MAX_OPT];

        // thread scheduling
        int n_tasks;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        void * data;
        char padding[8];
    };

    // computation graph
    struct ggml_cgraph {
        int n_nodes;
        int n_leafs;
        int n_threads;

        size_t work_size;
        struct ggml_tensor * work;

        struct ggml_tensor * nodes[GGML_MAX_NODES];
        struct ggml_tensor * grads[GGML_MAX_NODES];
        struct ggml_tensor * leafs[GGML_MAX_NODES];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    // scratch buffer
    struct ggml_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };

    // misc

    GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
    GGML_API int64_t ggml_time_ms(void);
    GGML_API int64_t ggml_time_us(void);
    GGML_API int64_t ggml_cycles(void);
    GGML_API int64_t ggml_cycles_per_ms(void);

    GGML_API void    ggml_print_object (const struct ggml_object * obj);
    GGML_API void    ggml_print_objects(const struct ggml_context * ctx);

    GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor);
    GGML_API size_t  ggml_nbytes   (const struct ggml_tensor * tensor);

    GGML_API int     ggml_blck_size (enum ggml_type type);
    GGML_API size_t  ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block
    GGML_API float   ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float

    GGML_API const char * ggml_type_name(enum ggml_type type);

    GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);

    GGML_API bool    ggml_is_quantized(enum ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);

    // main

    GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
    GGML_API void    ggml_free(struct ggml_context * ctx);

    GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);

    GGML_API size_t  ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);

    GGML_API struct ggml_tensor * ggml_new_tensor(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int    n_dims,
            const int64_t *ne);

    GGML_API struct ggml_tensor * ggml_new_tensor_1d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0);

    GGML_API struct ggml_tensor * ggml_new_tensor_2d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1);

    GGML_API struct ggml_tensor * ggml_new_tensor_3d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    GGML_API struct ggml_tensor * ggml_new_tensor_4d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
    GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);

    GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
    GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);

    //
    // operations on tensors with backpropagation
    //

    GGML_API struct ggml_tensor * ggml_dup(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_add_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_sub(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_mul(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_div(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_sqr(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sqrt(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // return scalar
    // TODO: compute sum along rows
    GGML_API struct ggml_tensor * ggml_sum(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // mean along rows
    GGML_API struct ggml_tensor * ggml_mean(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_API struct ggml_tensor * ggml_repeat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_abs(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sgn(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_neg(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_step(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_relu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // TODO: double-check this computation is correct
    GGML_API struct ggml_tensor * ggml_gelu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_silu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // normalize along rows
    // TODO: eps is hardcoded to 1e-5 for now
    GGML_API struct ggml_tensor * ggml_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_rms_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // A: m rows, n columns
    // B: p rows, n columns (i.e. we transpose it internally)
    // result is m columns, p rows
    GGML_API struct ggml_tensor * ggml_mul_mat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_scale(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // a -> b, return view(b)
    GGML_API struct ggml_tensor * ggml_cpy(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // make contiguous
    GGML_API struct ggml_tensor * ggml_cont(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    // offset in bytes
    GGML_API struct ggml_tensor * ggml_view_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_view_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_view_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_permute(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
    GGML_API struct ggml_tensor * ggml_transpose(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_get_rows(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // set elements above the diagonal to -INF
    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_diag_mask_inf(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_soft_max(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // rotary position embedding
    // in-place, returns view(a)
    // if mode & 1 == 1, skip n_past elements
    // if mode & 2 == 1, GPT-NeoX style
    // TODO: avoid creating a new tensor every time
    GGML_API struct ggml_tensor * ggml_rope(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past,
            int                   n_dims,
            int                   mode);

    // alibi position embedding
    // in-place, returns view(a)
    struct ggml_tensor * ggml_alibi(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past,
            int                   n_head);

    // padding = 1
    // TODO: we don't support extra parameters for now
    //       that's why we are hard-coding the stride, padding, and dilation
    //       not great ..
    GGML_API struct ggml_tensor * ggml_conv_1d_1s(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_conv_1d_2s(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_flash_attn(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            bool                  masked);

    GGML_API struct ggml_tensor * ggml_flash_ff(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b0,
            struct ggml_tensor  * b1,
            struct ggml_tensor  * c0,
            struct ggml_tensor  * c1);

    // Mapping operations
    typedef void (*ggml_unary_op_f32_t)(const int, float *, const float *);
    typedef void (*ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

    GGML_API struct ggml_tensor * ggml_map_unary_f32(
            struct ggml_context        * ctx,
            struct ggml_tensor         * a,
            const  ggml_unary_op_f32_t fun);

    GGML_API struct ggml_tensor * ggml_map_binary_f32(
            struct ggml_context         * ctx,
            struct ggml_tensor          * a,
            struct ggml_tensor          * b,
            const  ggml_binary_op_f32_t fun);

    //
    // automatic differentiation
    //

    GGML_API void ggml_set_param(
            struct ggml_context * ctx,
            struct ggml_tensor * tensor);

    GGML_API void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);

    GGML_API struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
    GGML_API struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);

    GGML_API void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph);
    GGML_API void ggml_graph_reset  (struct ggml_cgraph * cgraph);

    // print info and performance information for the graph
    GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);

    // dump the graph into a file using the dot format
    GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);

    //
    // optimization
    //

    // optimization methods
    enum ggml_opt_type {
        GGML_OPT_ADAM,
        GGML_OPT_LBFGS,
    };

    // linesearch methods
    enum ggml_linesearch {
        GGML_LINESEARCH_DEFAULT = 1,

        GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum ggml_opt_result {
        GGML_OPT_OK = 0,
        GGML_OPT_DID_NOT_CONVERGE,
        GGML_OPT_NO_CONTEXT,
        GGML_OPT_INVALID_WOLFE,
        GGML_OPT_FAIL,

        GGML_LINESEARCH_FAIL = -128,
        GGML_LINESEARCH_MINIMUM_STEP,
        GGML_LINESEARCH_MAXIMUM_STEP,
        GGML_LINESEARCH_MAXIMUM_ITERATIONS,
        GGML_LINESEARCH_INVALID_PARAMETERS,
    };

    // optimization parameters
    //
    //   see ggml.c (ggml_opt_default_params) for default values
    //
    struct ggml_opt_params {
        enum ggml_opt_type type;

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

            enum ggml_linesearch linesearch;
        } lbfgs;
    };

    GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);

    // optimize the function defined by the tensor f
    GGML_API enum ggml_opt_result ggml_opt(
            struct ggml_context * ctx,
            struct ggml_opt_params params,
            struct ggml_tensor * f);

    //
    // quantization
    //

    GGML_API size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_API size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_API size_t ggml_quantize_q4_2(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_API size_t ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_API size_t ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    GGML_API size_t ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);

    //
    // system info
    //

    GGML_API int ggml_cpu_has_avx        (void);
    GGML_API int ggml_cpu_has_avx2       (void);
    GGML_API int ggml_cpu_has_avx512     (void);
    GGML_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_API int ggml_cpu_has_avx512_vnni(void);
    GGML_API int ggml_cpu_has_fma        (void);
    GGML_API int ggml_cpu_has_neon       (void);
    GGML_API int ggml_cpu_has_arm_fma    (void);
    GGML_API int ggml_cpu_has_f16c       (void);
    GGML_API int ggml_cpu_has_fp16_va    (void);
    GGML_API int ggml_cpu_has_wasm_simd  (void);
    GGML_API int ggml_cpu_has_blas       (void);
    GGML_API int ggml_cpu_has_cublas     (void);
    GGML_API int ggml_cpu_has_clblast    (void);
    GGML_API int ggml_cpu_has_gpublas    (void);
    GGML_API int ggml_cpu_has_sse3       (void);
    GGML_API int ggml_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
    // restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif
    typedef void (*dequantize_row_q_t)(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    typedef void (*quantize_row_q_t)  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
    typedef void (*vec_dot_q_t)       (const int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT x, const void * GGML_RESTRICT y);

    typedef struct {
        dequantize_row_q_t dequantize_row_q;
        quantize_row_q_t   quantize_row_q;
        quantize_row_q_t   quantize_row_q_reference;
        quantize_row_q_t   quantize_row_q_dot;
        vec_dot_q_t        vec_dot_q;
        enum ggml_type     vec_dot_type;
    } quantize_fns_t;

    quantize_fns_t ggml_internal_get_quantize_fn(size_t i);

#ifdef  __cplusplus
}
#endif
