#pragma once


#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGML_RWKV_MAX_DIMS     4
#define GGML_RWKV_MAX_NODES    4096
#define GGML_RWKV_MAX_PARAMS   16
#define GGML_RWKV_MAX_CONTEXTS 64
#define GGML_RWKV_MAX_OPT      4

#ifdef __ARM_NEON
// we use the built-in 16-bit float type
typedef __fp16 ggml_rwkv_fp16_t;
#else
typedef uint16_t ggml_rwkv_fp16_t;
#endif

// convert FP16 <-> FP32
float       ggml_rwkv_fp16_to_fp32(ggml_rwkv_fp16_t x);
ggml_rwkv_fp16_t ggml_rwkv_fp32_to_fp16(float x);

struct ggml_rwkv_object;
struct ggml_rwkv_context;

enum ggml_rwkv_type {
    GGML_RWKV_TYPE_Q4_0,
    // Stores min and delta per block, does quantized matmul.
    GGML_RWKV_TYPE_Q4_1,
    // Same as Q4_1, but stores outliers separately, and matmul is done in FP32.
    // An outlier is the single absmax element in the quantized block.
    GGML_RWKV_TYPE_Q4_1_O,
    GGML_RWKV_TYPE_I8,
    GGML_RWKV_TYPE_I16,
    GGML_RWKV_TYPE_I32,
    GGML_RWKV_TYPE_F16,
    GGML_RWKV_TYPE_F32,
    GGML_RWKV_TYPE_COUNT,
};

// available tensor operations:
enum ggml_rwkv_op {
    GGML_RWKV_OP_NONE = 0,

    GGML_RWKV_OP_DUP,
    GGML_RWKV_OP_ADD,
    GGML_RWKV_OP_SUB,
    GGML_RWKV_OP_MUL,
    GGML_RWKV_OP_DIV,
    GGML_RWKV_OP_SQR,
    GGML_RWKV_OP_SQRT,
    GGML_RWKV_OP_SUM,
    GGML_RWKV_OP_MEAN,
    GGML_RWKV_OP_REPEAT,
    GGML_RWKV_OP_ABS,
    GGML_RWKV_OP_SGN,
    GGML_RWKV_OP_NEG,
    // Element-wise exponential function `e^x`.
    // Same as `torch.exp(x)` from PyTorch.
    GGML_RWKV_OP_EXP,
    // Element-wise `1 - x`.
    GGML_RWKV_OP_1_MINUS_X,

    // Element-wise maximum of 2 values. Argument shapes must match.
    // Same as `torch.maximum(x)` from PyTorch.
    GGML_RWKV_OP_MAX,

    GGML_RWKV_OP_STEP,
    GGML_RWKV_OP_RELU,
    GGML_RWKV_OP_GELU,
    // Element-wise sigmoid activation `1 / (1 + e^-x)`, also called logistic function.
    // Same as `torch.sigmoid(x)` from PyTorch.
    GGML_RWKV_OP_SIGMOID,
    GGML_RWKV_OP_SILU,
    GGML_RWKV_OP_NORM, // normalize
    GGML_RWKV_OP_RMS_NORM,

    GGML_RWKV_OP_MUL_MAT,

    GGML_RWKV_OP_SCALE,
    GGML_RWKV_OP_CPY,
    GGML_RWKV_OP_RESHAPE,
    GGML_RWKV_OP_VIEW,
    GGML_RWKV_OP_PERMUTE,
    GGML_RWKV_OP_TRANSPOSE,
    GGML_RWKV_OP_GET_ROWS,
    GGML_RWKV_OP_DIAG_MASK_INF,
    GGML_RWKV_OP_SOFT_MAX,
    GGML_RWKV_OP_ROPE,
    GGML_RWKV_OP_CONV_1D_1S,
    GGML_RWKV_OP_CONV_1D_2S,

    GGML_RWKV_OP_FLASH_ATTN,
    GGML_RWKV_OP_FLASH_FF,

    GGML_RWKV_OP_COUNT,
};

// n-dimensional tensor
struct ggml_rwkv_tensor {
    enum ggml_rwkv_type type;

    int    n_dims;
    int    ne[GGML_RWKV_MAX_DIMS]; // number of elements
    size_t nb[GGML_RWKV_MAX_DIMS]; // stride in bytes:
                              // nb[0] = sizeof(type)
                              // nb[1] = nb[0]   * ne[0] + padding
                              // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum ggml_rwkv_op op;

    bool is_param;

    struct ggml_rwkv_tensor * grad;
    struct ggml_rwkv_tensor * src0;
    struct ggml_rwkv_tensor * src1;
    struct ggml_rwkv_tensor * opt[GGML_RWKV_MAX_OPT];

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
struct ggml_rwkv_cgraph {
    int n_nodes;
    int n_leafs;
    int n_threads;

    size_t work_size;
    struct ggml_rwkv_tensor * work;

    struct ggml_rwkv_tensor * nodes[GGML_RWKV_MAX_NODES];
    struct ggml_rwkv_tensor * grads[GGML_RWKV_MAX_NODES];
    struct ggml_rwkv_tensor * leafs[GGML_RWKV_MAX_NODES];

    // performance
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
};

// scratch buffer
struct ggml_rwkv_scratch {
    size_t offs;
    size_t size;
    void * data;
};

struct ggml_rwkv_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
};

void    ggml_rwkv_time_init(void); // call this once at the beginning of the program
int64_t ggml_rwkv_time_ms(void);
int64_t ggml_rwkv_time_us(void);
int64_t ggml_rwkv_cycles(void);
int64_t ggml_rwkv_cycles_per_ms(void);

void ggml_rwkv_print_object (const struct ggml_rwkv_object * obj);
void ggml_rwkv_print_objects(const struct ggml_rwkv_context * ctx);

int    ggml_rwkv_nelements(const struct ggml_rwkv_tensor * tensor);
size_t ggml_rwkv_nbytes   (const struct ggml_rwkv_tensor * tensor);

int    ggml_rwkv_blck_size (enum ggml_rwkv_type type);
size_t ggml_rwkv_type_size (enum ggml_rwkv_type type); // size in bytes for all elements in a block
float  ggml_rwkv_type_sizef(enum ggml_rwkv_type type); // ggml_rwkv_type_size()/ggml_rwkv_blck_size() as float

size_t ggml_rwkv_element_size(const struct ggml_rwkv_tensor * tensor);

struct ggml_rwkv_context * ggml_rwkv_init(struct ggml_rwkv_init_params params);
void ggml_rwkv_free(struct ggml_rwkv_context * ctx);

size_t ggml_rwkv_used_mem(const struct ggml_rwkv_context * ctx);

size_t ggml_rwkv_set_scratch(struct ggml_rwkv_context * ctx, struct ggml_rwkv_scratch scratch);

bool ggml_rwkv_mlock_supported(void);
bool ggml_rwkv_mlock(struct ggml_rwkv_context * ctx, char ** err_p);

struct ggml_rwkv_tensor * ggml_rwkv_new_tensor(
        struct ggml_rwkv_context * ctx,
        enum   ggml_rwkv_type type,
        int    n_dims,
        const int *ne);

struct ggml_rwkv_tensor * ggml_rwkv_new_tensor_1d(
        struct ggml_rwkv_context * ctx,
        enum   ggml_rwkv_type type,
        int    ne0);

struct ggml_rwkv_tensor * ggml_rwkv_new_tensor_2d(
        struct ggml_rwkv_context * ctx,
        enum   ggml_rwkv_type type,
        int    ne0,
        int    ne1);

struct ggml_rwkv_tensor * ggml_rwkv_new_tensor_3d(
        struct ggml_rwkv_context * ctx,
        enum   ggml_rwkv_type type,
        int    ne0,
        int    ne1,
        int    ne2);

struct ggml_rwkv_tensor * ggml_rwkv_new_tensor_4d(
        struct ggml_rwkv_context * ctx,
        enum   ggml_rwkv_type type,
        int    ne0,
        int    ne1,
        int    ne2,
        int    ne3);

struct ggml_rwkv_tensor * ggml_rwkv_new_i32(struct ggml_rwkv_context * ctx, int32_t value);
struct ggml_rwkv_tensor * ggml_rwkv_new_f32(struct ggml_rwkv_context * ctx, float value);

struct ggml_rwkv_tensor * ggml_rwkv_dup_tensor (struct ggml_rwkv_context * ctx, const struct ggml_rwkv_tensor * src);
struct ggml_rwkv_tensor * ggml_rwkv_view_tensor(struct ggml_rwkv_context * ctx, const struct ggml_rwkv_tensor * src);

struct ggml_rwkv_tensor * ggml_rwkv_set_zero(struct ggml_rwkv_tensor * tensor);
struct ggml_rwkv_tensor * ggml_rwkv_set_i32 (struct ggml_rwkv_tensor * tensor, int32_t value);
struct ggml_rwkv_tensor * ggml_rwkv_set_f32 (struct ggml_rwkv_tensor * tensor, float value);

int32_t ggml_rwkv_get_i32_1d(const struct ggml_rwkv_tensor * tensor, int i);
void    ggml_rwkv_set_i32_1d(const struct ggml_rwkv_tensor * tensor, int i, int32_t value);

float ggml_rwkv_get_f32_1d(const struct ggml_rwkv_tensor * tensor, int i);
void  ggml_rwkv_set_f32_1d(const struct ggml_rwkv_tensor * tensor, int i, float value);

 void * ggml_rwkv_get_data    (const struct ggml_rwkv_tensor * tensor);
float * ggml_rwkv_get_data_f32(const struct ggml_rwkv_tensor * tensor);

//
// operations on tensors with backpropagation
//

struct ggml_rwkv_tensor * ggml_rwkv_dup(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_add(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_sub(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_mul(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_div(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_sqr(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_sqrt(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// return scalar
// TODO: compute sum along rows
struct ggml_rwkv_tensor * ggml_rwkv_sum(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// mean along rows
struct ggml_rwkv_tensor * ggml_rwkv_mean(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
struct ggml_rwkv_tensor * ggml_rwkv_repeat(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_abs(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_sgn(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_neg(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_exp(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_1_minus_x(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_max(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_step(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_relu(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// TODO: double-check this computation is correct
struct ggml_rwkv_tensor * ggml_rwkv_gelu(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_sigmoid(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_silu(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// normalize along rows
// TODO: eps is hardcoded to 1e-5 for now
struct ggml_rwkv_tensor * ggml_rwkv_norm(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_rms_norm(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// A: m rows, n columns
// B: p rows, n columns (i.e. we transpose it internally)
// result is m columns, p rows
struct ggml_rwkv_tensor * ggml_rwkv_mul_mat(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

//
// operations on tensors without backpropagation
//

// in-place, returns view(a)
struct ggml_rwkv_tensor * ggml_rwkv_scale(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

// a -> b, return view(b)
struct ggml_rwkv_tensor * ggml_rwkv_cpy(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
struct ggml_rwkv_tensor * ggml_rwkv_reshape(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
struct ggml_rwkv_tensor * ggml_rwkv_reshape_2d(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   ne0,
        int                   ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
struct ggml_rwkv_tensor * ggml_rwkv_reshape_3d(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2);

// offset in bytes
struct ggml_rwkv_tensor * ggml_rwkv_view_1d(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   ne0,
        size_t                offset);

struct ggml_rwkv_tensor * ggml_rwkv_view_2d(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1, // row stride in bytes
        size_t                offset);

struct ggml_rwkv_tensor * ggml_rwkv_permute(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3);

// alias for ggml_rwkv_permute(ctx, a, 1, 0, 2, 3)
struct ggml_rwkv_tensor * ggml_rwkv_transpose(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

struct ggml_rwkv_tensor * ggml_rwkv_get_rows(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

// set elements above the diagonal to -INF
// in-place, returns view(a)
struct ggml_rwkv_tensor * ggml_rwkv_diag_mask_inf(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   n_past);

// in-place, returns view(a)
struct ggml_rwkv_tensor * ggml_rwkv_soft_max(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a);

// rotary position embedding
// in-place, returns view(a)
// if mode == 1, skip n_past elements
// TODO: avoid creating a new tensor every time
struct ggml_rwkv_tensor * ggml_rwkv_rope(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode);

// padding = 1
// TODO: we don't support extra parameters for now
//       that's why we are hard-coding the stride, padding, and dilation
//       not great ..
struct ggml_rwkv_tensor * ggml_rwkv_conv_1d_1s(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_conv_1d_2s(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b);

struct ggml_rwkv_tensor * ggml_rwkv_flash_attn(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * q,
        struct ggml_rwkv_tensor  * k,
        struct ggml_rwkv_tensor  * v,
        bool                  masked);

struct ggml_rwkv_tensor * ggml_rwkv_flash_ff(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor  * a,
        struct ggml_rwkv_tensor  * b0,
        struct ggml_rwkv_tensor  * b1,
        struct ggml_rwkv_tensor  * c0,
        struct ggml_rwkv_tensor  * c1);

//
// automatic differentiation
//

void ggml_rwkv_set_param(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_tensor * tensor);

void ggml_rwkv_build_forward_expand(struct ggml_rwkv_cgraph * cgraph, struct ggml_rwkv_tensor * tensor);

struct ggml_rwkv_cgraph ggml_rwkv_build_forward (struct ggml_rwkv_tensor * tensor);
struct ggml_rwkv_cgraph ggml_rwkv_build_backward(struct ggml_rwkv_context * ctx, struct ggml_rwkv_cgraph * gf, bool keep);

void ggml_rwkv_graph_compute(struct ggml_rwkv_context * ctx, struct ggml_rwkv_cgraph * cgraph);
void ggml_rwkv_graph_reset  (struct ggml_rwkv_cgraph * cgraph);

// print info and performance information for the graph
void ggml_rwkv_graph_print(const struct ggml_rwkv_cgraph * cgraph);

// dump the graph into a file using the dot format
void ggml_rwkv_graph_dump_dot(const struct ggml_rwkv_cgraph * gb, const struct ggml_rwkv_cgraph * gf, const char * filename);

//
// optimization
//

// optimization methods
enum ggml_rwkv_opt_type {
    GGML_RWKV_OPT_ADAM,
    GGML_RWKV_OPT_LBFGS,
};

// linesearch methods
enum ggml_rwkv_linesearch {
    GGML_RWKV_LINESEARCH_DEFAULT = 1,

    GGML_RWKV_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
    GGML_RWKV_LINESEARCH_BACKTRACKING_WOLFE        = 1,
    GGML_RWKV_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
};

// optimization return values
enum ggml_rwkv_opt_result {
    GGML_RWKV_OPT_OK = 0,
    GGML_RWKV_OPT_DID_NOT_CONVERGE,
    GGML_RWKV_OPT_NO_CONTEXT,
    GGML_RWKV_OPT_INVALID_WOLFE,
    GGML_RWKV_OPT_FAIL,

    GGML_RWKV_LINESEARCH_FAIL = -128,
    GGML_RWKV_LINESEARCH_MINIMUM_STEP,
    GGML_RWKV_LINESEARCH_MAXIMUM_STEP,
    GGML_RWKV_LINESEARCH_MAXIMUM_ITERATIONS,
    GGML_RWKV_LINESEARCH_INVALID_PARAMETERS,
};

// optimization parameters
//
//   see ggml.c (ggml_rwkv_opt_default_params) for default values
//
struct ggml_rwkv_opt_params {
    enum ggml_rwkv_opt_type type;

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

        enum ggml_rwkv_linesearch linesearch;
    } lbfgs;
};

struct ggml_rwkv_opt_params ggml_rwkv_opt_default_params(enum ggml_rwkv_opt_type type);

// optimize the function defined by the tensor f
enum ggml_rwkv_opt_result ggml_rwkv_opt(
        struct ggml_rwkv_context * ctx,
        struct ggml_rwkv_opt_params params,
        struct ggml_rwkv_tensor * f);

//
// quantization
//

size_t ggml_rwkv_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_rwkv_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
size_t ggml_rwkv_quantize_q4_1_o(const float * src, void * dst, int n, int k, int64_t * hist);

//
// system info
//

int ggml_rwkv_cpu_has_avx(void);
int ggml_rwkv_cpu_has_avx2(void);
int ggml_rwkv_cpu_has_avx512(void);
int ggml_rwkv_cpu_has_fma(void);
int ggml_rwkv_cpu_has_neon(void);
int ggml_rwkv_cpu_has_arm_fma(void);
int ggml_rwkv_cpu_has_f16c(void);
int ggml_rwkv_cpu_has_fp16_va(void);
int ggml_rwkv_cpu_has_wasm_simd(void);
int ggml_rwkv_cpu_has_blas(void);
int ggml_rwkv_cpu_has_sse3(void);
int ggml_rwkv_cpu_has_vsx(void);

// Run test suite for ggml.
// Exits normally, if all tests pass.
// Aborts the execution if any test did not pass.
void ggml_rwkv_run_test_suite();

#ifdef  __cplusplus
}
#endif
