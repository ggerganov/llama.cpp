#include <string>

// DEQUANT SHADER
const std::string dequant_head = R"(
#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
)";

const std::string dequant_glsl_fp16_ext = R"(
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
)";

const std::string dequant_output_fp16 = R"(
#define OUT_TYPE float16_t
)";

const std::string dequant_output_fp32 = R"(
#define OUT_TYPE float
)";

const std::string dequant_q4_0_defines = R"(
#define QUANT_K 32
#define QUANT_R 2

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};

#define A_TYPE block_q4_0
)";

const std::string dequant_body = R"(
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { A_TYPE x[]; };
layout (binding = 1) writeonly buffer D { OUT_TYPE y[]; };

layout (push_constant) uniform parameter
{
    int M;
    int K;
    int stride_a;
    int stride_b;
} p;

void main() {
    const int i = int(gl_GlobalInvocationID.x);

    // Transposed
    const int row = i % (p.K / QUANT_K);
    const int col = i / (p.K / QUANT_K);

    if (row * QUANT_K >= p.K || col >= p.M) {
        return;
    }

    const int stride_a = p.stride_a / QUANT_K;

    const A_TYPE blk = x[col * stride_a + row];
    const OUT_TYPE d = blk.d;

    [[unroll]] for (int j = 0; j < QUANT_K/2; ++j) {
        const OUT_TYPE x0 = OUT_TYPE((blk.qs[j] & 0x0F) - 8);
        const OUT_TYPE x1 = OUT_TYPE((blk.qs[j] >>   4) - 8);

        y[col * p.stride_b + row*QUANT_K + j + 0   ] = x0*d;
        y[col * p.stride_b + row*QUANT_K + j + QUANT_K/2] = x1*d;
    }
}
)";

// Mul Mat Vec
const std::string mul_mat_vec_head = R"(
#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
)";

const std::string mul_mat_vec_fp16 = R"(
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
)";

const std::string mul_mat_vec_q4_0_defines = R"(
#define QUANT_K 32
#define QUANT_R 2
#define BLOCK_SIZE 32

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};
#define A_TYPE block_q4_0
#define B_TYPE float16_t
#define OUT_TYPE float

#define DEQUANT_FUNC const float16_t d = x[ib].d; \
const uint8_t vui = x[ib].qs[iqs]; \
const int8_t vi0 = int8_t(vui & 0xF); \
const int8_t vi1 = int8_t(vui >> 4); \
float16_t v0 = float16_t(vi0 - 8)*d; \
float16_t v1 = float16_t(vi1 - 8)*d;
)";

const std::string mul_mat_vec_body = R"(
layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { A_TYPE x[]; };
layout (binding = 1) readonly buffer B { B_TYPE y[]; };
layout (binding = 2) writeonly buffer D { OUT_TYPE dst[]; };

layout (push_constant) uniform parameter
{
    int ncols;
} p;

shared float16_t tmp[BLOCK_SIZE];

void main() {
    const int block_size = int(gl_WorkGroupSize.x);
    const int row = int(gl_WorkGroupID.x);
    const int tid = int(gl_LocalInvocationID.x);

    const int y_offset = QUANT_K/2;

    tmp[tid] = 0.0hf;

    [[unroll]] for (int i = 0; i < p.ncols/block_size; i += 2) {
        const int col = i*block_size + 2*tid;
        const int ib = (row*p.ncols + col)/QUANT_K; // block index
        const int iqs = (col%QUANT_K)/QUANT_R; // quant index
        const int iybs = col - col%QUANT_K; // y block start index

        DEQUANT_FUNC

        // matrix multiplication
        tmp[tid] += v0 * y[iybs + iqs + 0];
        tmp[tid] += v1 * y[iybs + iqs + y_offset];
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s=block_size/2; s>0; s>>=1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }
    if (tid == 0) {
        dst[row] = float(tmp[0]);
    }
}
)";

// F16 to F32
const std::string f32_to_f16_src = R"(
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { float data_a[]; };
layout (binding = 1) writeonly buffer D { float16_t data_b[]; };

layout (push_constant) uniform parameter
{
    int M;
    int K;
    int stride_a;
    int stride_b;
} p;

void main() {
    const int row = int(gl_GlobalInvocationID.x % p.K);
    const int col = int(gl_GlobalInvocationID.x / p.K);

    if (row < p.K && col < p.M) {
        data_b[col * p.stride_b + row] = float16_t(data_a[col * p.stride_a + row]);
    }
}
)";

// MUL F32
const std::string mul_f32_src = R"(
#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) buffer X { float data_x[]; };
layout (binding = 1) buffer Y { float data_y[]; };
layout (binding = 2) buffer D { float data_d[]; };

layout (push_constant) uniform parameter
{
    int M;
    int N;
    int stride_x;
    int stride_y;
    int stride_d;
    int x_offset;
    int y_offset;
    int d_offset;
    float scale;
} p;

void main() {
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);

    if (x >= p.M || y >= p.N) {
        return;
    }

    data_d[p.d_offset + y * p.stride_d + x] = data_x[p.x_offset + y * p.stride_x + x] * data_y[p.y_offset + x];
}
)";
