#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define QUANT_K 32
#define QUANT_R 2
#define BLOCK_SIZE 32

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { float16_t x[]; };
layout (binding = 1) readonly buffer B { float16_t y[]; };
layout (binding = 2) writeonly buffer D { float dst[]; };

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

        // dequantize
        float16_t v0 = x[ib + 0];
        float16_t v1 = x[ib + 1];

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
