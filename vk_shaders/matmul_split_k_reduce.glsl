#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) buffer A { float data[]; };

layout (push_constant) uniform parameter
{
    int M;
    int N;
    int k_num;
} p;

void main() {
    const int glr = int(gl_GlobalInvocationID.x);
    const int glc = int(gl_GlobalInvocationID.y);

    if (glr >= p.M || glc >= p.N) {
        return;
    }

    const int idx = glc * p.M + glr;

    float result = 0.0f;

    for (int i = 0; i < p.k_num; i++) {
        result += data[i * p.M * p.N + idx];
    }

    data[idx] = result;
}
