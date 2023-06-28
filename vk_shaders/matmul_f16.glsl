#version 450

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = (BM * BN) / (TM * TN), local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { float16_t data_a[]; };
layout (binding = 1) readonly buffer B { float16_t data_b[]; };
layout (binding = 2) writeonly buffer D { float16_t data_d[]; };

layout (push_constant) uniform parameter
{
    int M;
    int N;
    int K;
    int stride_a;
    int stride_b;
    int stride_d;
} p;

shared float16_t buf_a[BM * (BK+1)];
shared float16_t buf_b[BN * (BK+1)];

void main() {
    const int ir = int(gl_WorkGroupID.x);
    const int ic = int(gl_WorkGroupID.y);

    const int rstride = BM / TM;

    const int lr = int(gl_LocalInvocationID.x % rstride);
    const int lc = int(gl_LocalInvocationID.x / rstride);

    const int loadr = int(gl_LocalInvocationID.x % BK);
    const int loadc = int(gl_LocalInvocationID.x / BK);

    const int loadstride = int(gl_WorkGroupSize.x);

    int pos_a = ir * BM * p.stride_a;
    int pos_b = ic * BN * p.stride_b;

    float16_t sums[TM * TN];
    float16_t cache_a[TM];
    float16_t cache_b[TN];

    [[unroll]] for (int i = 0; i < TM*TN; i++) {
        sums[i] = 0.0hf;
    }

    [[unroll]] for (int block = 0; block < p.K; block += BK) {
        [[unroll]] for (int l = 0; l < BM * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            buf_a[(loadc + lc) * (BK+1) + loadr + lr] = data_a[pos_a + (loadc + lc) * p.stride_a + loadr + lr];
        }
        [[unroll]] for (int l = 0; l < BN * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            buf_b[(loadc + lc) * (BK+1) + loadr + lr] = data_b[pos_b + (loadc + lc) * p.stride_b + loadr + lr];
        }

        barrier();

        pos_a += BK;
        pos_b += BK;

        [[unroll]] for (int i = 0; i < BK; i++) {
            // Load from shared into cache
            [[unroll]] for (int j = 0; j < BM; j++) {
                cache_a[j] = buf_a[(lr + j*rstride) * (BK+1) + i];
            }
            [[unroll]] for (int j = 0; j < TN; j++) {
                cache_b[j] = buf_b[(lc * TN + j) * (BK+1) + i];
            }

            [[unroll]] for (int cc = 0; cc < TN; cc++) {
                [[unroll]] for (int cr = 0; cr < TM; cr++) {
                    sums[cc * TM + cr] += cache_a[cr] * cache_b[cc];
                }
            }
        }

        barrier();
    }

    const int dr = ir * BM + lr;
    const int dc = ic * BN + lc * TN;

    [[unroll]] for (int cc = 0; cc < TN; cc++) {
        [[unroll]] for (int cr = 0; cr < TM; cr++) {
            data_d[(dc + cc) * p.stride_d + dr + cr*rstride] = sums[cc * TM + cr];
        }
    }
}
