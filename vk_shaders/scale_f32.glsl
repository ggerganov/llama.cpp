#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) buffer X { float data_x[]; };
layout (binding = 1) buffer D { float data_d[]; };

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

    data_d[p.d_offset + y * p.stride_d + x] = data_x[p.x_offset + y * p.stride_x + x] * p.scale;
}
