#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { float16_t data_a[]; };
layout (binding = 1) writeonly buffer D { float data_b[]; };

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

    if (row < p.M && col < p.K) {
        data_b[col * p.stride_b + row] = float(data_a[col * p.stride_a + row]);
    }
}
