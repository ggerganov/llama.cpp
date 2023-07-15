#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#define QUANT_K 32
#define QUANT_R 2

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};

layout (binding = 0) readonly buffer A { block_q4_0 x[]; };
layout (binding = 1) writeonly buffer D { float y[]; };

layout (push_constant) uniform parameter
{
    int N;
} p;

void main() {
    const int i = int(gl_GlobalInvocationID.x);

    if (i >= p.N) {
        return;
    }

    const block_q4_0 blk = x[i];

    const float d = float(blk.d);

    [[unroll]] for (int j = 0; j < QUANT_K/2; ++j) {
        const int x0 = (blk.qs[j] & 0x0F) - 8;
        const int x1 = (blk.qs[j] >>   4) - 8;

        y[i*QUANT_K + j + 0   ] = x0*d;
        y[i*QUANT_K + j + QUANT_K/2] = x1*d;
    }
}
