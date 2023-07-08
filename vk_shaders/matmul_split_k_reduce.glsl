#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) buffer A { float data[]; };

layout (push_constant) uniform parameter
{
    int N;
    int k_num;
} p;

void main() {
    const int idx = int(gl_GlobalInvocationID.x);

    if (idx >= p.N) {
        return;
    }

    float result = 0.0f;

    for (int i = 0; i < p.k_num; i++) {
        result += data[i * p.N + idx];
    }

    data[idx] = result;
}
