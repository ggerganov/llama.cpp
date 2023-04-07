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
// ## Adding new operators
//
// Suppose you want to add e^x unary operator. Following steps need to be done:
//
// In `ggml.h`:
//
// 1. Add member `GGML_OP_EXP` to `ggml_op` enum.
// 2. Declare the operator function: `struct ggml_tensor * ggml_exp(struct ggml_context * ctx, struct ggml_tensor * x);`.
//
// In `ggml.c`:
//
// 1. Implement `ggml_exp` function: it will create result tensor and set its' operator and arguments.
// 2. Create forward computation function for FP32: `ggml_compute_forward_exp_f32`: it will do the actual computation.
// 3. If needed, create forward computation functions for other types: FP16, INT32, etc.
// 4. Create forward dispatch function `ggml_compute_forward_exp`: it would dispatch the call based on tensor data type.
// 5. Add `case GGML_OP_EXP`:
//   - to `ggml_compute_forward` and call the forward dispatch function here.
//   - to `ggml_compute_backward` and add `GGML_ASSERT(false)` here.
//   - to `ggml_graph_compute` and add `node->n_tasks = 1` here.
// 6. Add operator label to `GGML_OP_LABEL` array and operator symbol to `GGML_OP_SYMBOL` array.
// 7. Fix all assertions that check value of `GGML_OP_COUNT`: you've added 1 operator, so increment asserted value by one.
//
// When in doubt, consult the code of existing operators similar to that you're implementing.
// Resulting operator would work for the forward pass, but will lack backward implementation and multi-threading support.
//
// TODO Implementing backward pass
// TODO Implementing multi-threading
//

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_MAX_DIMS     4
#define GGML_MAX_NODES    4096
#define GGML_MAX_PARAMS   16
#define GGML_MAX_CONTEXTS 64
#define GGML_MAX_OPT      4

#ifdef __ARM_NEON
// we use the built-in 16-bit float type
typedef __fp16 ggml_fp16_t;
#else
typedef uint16_t ggml_fp16_t;
#endif

// convert FP16 <-> FP32
float       ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16(float x);

struct ggml_object;
struct ggml_context;

enum ggml_type {
    GGML_TYPE_Q4_0,
    // Stores min and delta per block, does quantized matmul.
    GGML_TYPE_Q4_1,
    // Same as Q4_1, but stores outliers separately, and matmul is done in FP32.
    // An outlier is the single absmax element in the quantized block.
    GGML_TYPE_Q4_1_O,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_COUNT,
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
    // Element-wise exponential function `e^x`.
    // Same as `torch.exp(x)` from PyTorch.
    GGML_OP_EXP,
    // Element-wise `1 - x`.
    GGML_OP_1_MINUS_X,

    // Element-wise maximum of 2 values. Argument shapes must match.
    // Same as `torch.maximum(x)` from PyTorch.
    GGML_OP_MAX,

    GGML_OP_STEP,
    GGML_OP_RELU,
    GGML_OP_GELU,
    // Element-wise sigmoid activation `1 / (1 + e^-x)`, also called logistic function.
    // Same as `torch.sigmoid(x)` from PyTorch.
    GGML_OP_SIGMOID,
    GGML_OP_SILU,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,

    GGML_OP_MUL_MAT,

    GGML_OP_SCALE,
    GGML_OP_CPY,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_SOFT_MAX,
    GGML_OP_ROPE,
    GGML_OP_CONV_1D_1S,
    GGML_OP_CONV_1D_2S,

    GGML_OP_FLASH_ATTN,
    GGML_OP_FLASH_FF,

    GGML_OP_COUNT,
};

// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type type;

    int    n_dims;
    int    ne[GGML_MAX_DIMS]; // number of elements
    size_t nb[GGML_MAX_DIMS]; // stride in bytes:
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
};

void    ggml_time_init(void); // call this once at the beginning of the program
int64_t ggml_time_ms(void);
int64_t ggml_time_us(void);
int64_t ggml_cycles(void);
int64_t ggml_cycles_per_ms(void);

void ggml_print_object (const struct ggml_object * obj);
void ggml_print_objects(const struct ggml_context * ctx);

int    ggml_nelements(const struct ggml_tensor * tensor);
size_t ggml_nbytes   (const struct ggml_tensor * tensor);

int    ggml_blck_size (enum ggml_type type);
size_t ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block
float  ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float

size_t ggml_element_size(const struct ggml_tensor * tensor);

struct ggml_context * ggml_init(struct ggml_init_params params);
void ggml_free(struct ggml_context * ctx);

size_t ggml_used_mem(const struct ggml_context * ctx);

size_t ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);

bool ggml_mlock_supported(void);
bool ggml_mlock(struct ggml_context * ctx, char ** err_p);

struct ggml_tensor * ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int *ne);

struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0);

struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1);

struct ggml_tensor * ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1,
        int    ne2);

struct ggml_tensor * ggml_new_tensor_4d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1,
        int    ne2,
        int    ne3);

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);

struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
void  ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

 void * ggml_get_data    (const struct ggml_tensor * tensor);
float * ggml_get_data_f32(const struct ggml_tensor * tensor);

//
// operations on tensors with backpropagation
//

struct ggml_tensor * ggml_dup(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_sub(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_div(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_sqr(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_sqrt(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// return scalar
// TODO: compute sum along rows
struct ggml_tensor * ggml_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// mean along rows
struct ggml_tensor * ggml_mean(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
struct ggml_tensor * ggml_repeat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_abs(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_sgn(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_neg(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_exp(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_1_minus_x(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_step(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_relu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// TODO: double-check this computation is correct
struct ggml_tensor * ggml_gelu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_sigmoid(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_silu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// normalize along rows
// TODO: eps is hardcoded to 1e-5 for now
struct ggml_tensor * ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// A: m rows, n columns
// B: p rows, n columns (i.e. we transpose it internally)
// result is m columns, p rows
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

//
// operations on tensors without backpropagation
//

// in-place, returns view(a)
struct ggml_tensor * ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

// a -> b, return view(b)
struct ggml_tensor * ggml_cpy(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
struct ggml_tensor * ggml_reshape(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
struct ggml_tensor * ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2);

// offset in bytes
struct ggml_tensor * ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        size_t                offset);

struct ggml_tensor * ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1, // row stride in bytes
        size_t                offset);

struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3);

// alias for ggml_permute(ctx, a, 1, 0, 2, 3)
struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

struct ggml_tensor * ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

// set elements above the diagonal to -INF
// in-place, returns view(a)
struct ggml_tensor * ggml_diag_mask_inf(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past);

// in-place, returns view(a)
struct ggml_tensor * ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a);

// rotary position embedding
// in-place, returns view(a)
// if mode == 1, skip n_past elements
// TODO: avoid creating a new tensor every time
struct ggml_tensor * ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode);

// padding = 1
// TODO: we don't support extra parameters for now
//       that's why we are hard-coding the stride, padding, and dilation
//       not great ..
struct ggml_tensor * ggml_conv_1d_1s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_conv_1d_2s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b);

struct ggml_tensor * ggml_flash_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        bool                  masked);

struct ggml_tensor * ggml_flash_ff(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b0,
        struct ggml_tensor  * b1,
        struct ggml_tensor  * c0,
        struct ggml_tensor  * c1);

//
// automatic differentiation
//

void ggml_set_param(
        struct ggml_context * ctx,
        struct ggml_tensor * tensor);

void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);

struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);

void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph);
void ggml_graph_reset  (struct ggml_cgraph * cgraph);

// print info and performance information for the graph
void ggml_graph_print(const struct ggml_cgraph * cgraph);

// dump the graph into a file using the dot format
void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);

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

struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);

// optimize the function defined by the tensor f
enum ggml_opt_result ggml_opt(
        struct ggml_context * ctx,
        struct ggml_opt_params params,
        struct ggml_tensor * f);

//
// quantization
//

size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_quantize_q4_1_o(const float * src, void * dst, int n, int k, int64_t * hist);

//
// system info
//

int ggml_cpu_has_avx(void);
int ggml_cpu_has_avx2(void);
int ggml_cpu_has_avx512(void);
int ggml_cpu_has_fma(void);
int ggml_cpu_has_neon(void);
int ggml_cpu_has_arm_fma(void);
int ggml_cpu_has_f16c(void);
int ggml_cpu_has_fp16_va(void);
int ggml_cpu_has_wasm_simd(void);
int ggml_cpu_has_blas(void);
int ggml_cpu_has_sse3(void);
int ggml_cpu_has_vsx(void);

// Run test suite for ggml.
// Exits normally, if all tests pass.
// Aborts the execution if any test did not pass.
void ggml_run_test_suite();

#ifdef  __cplusplus
}
#endif
