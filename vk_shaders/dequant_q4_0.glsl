#version 450

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#define QUANT_K 32
#define QUANT_R 2

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

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
    const int idx = int(gl_GlobalInvocationID.x);

    const int i = int(gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x*2);

    if (idx >= p.N) {
        return;
    }

    const int qk = QUANT_K;
    const int qr = QUANT_R;

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    float v0, v1;
    const float d = float(x[ib].d);

    const uint8_t vui = x[ib].qs[iqs];

    const int8_t vi0 = int8_t(vui & 0xF);
    const int8_t vi1 = int8_t(vui >> 4);

    v0 = (vi0 - 8)*d;
    v1 = (vi1 - 8)*d;

    y[iybs + iqs + 0] = v0;
    y[iybs + iqs + y_offset] = v1;
}
