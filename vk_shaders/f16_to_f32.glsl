#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { float16_t data_a[]; };
layout (binding = 1) writeonly buffer D { float data_b[]; };

layout (push_constant) uniform parameter
{
    int N;
} p;

void main() {
    const int idx = int(gl_GlobalInvocationID.x);

    if (idx < p.N) {
        data_b[idx] = float(data_a[idx]);
    }
}
