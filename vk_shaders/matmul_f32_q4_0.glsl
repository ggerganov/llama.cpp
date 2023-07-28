#version 450

#define WARP 32

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#define QUANT_K 32
#define QUANT_R 2

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { block_q4_0 data_a[]; };
layout (binding = 1) readonly buffer B { vec4 data_b[]; };
layout (binding = 2) writeonly buffer D { float data_d[]; };

layout (push_constant) uniform parameter
{
    int M;
    int N;
    int K;
    int stride_a;
    int stride_b;
    int stride_d;
    int k_split;
} p;

layout (constant_id = 1) const int BM = 64;
layout (constant_id = 2) const int BN = 64;
layout (constant_id = 3) const int BK = 16;
layout (constant_id = 4) const int WM = 32;
layout (constant_id = 5) const int WN = 32;
layout (constant_id = 6) const int WMITER = 2;
layout (constant_id = 7) const int TM = 4;
layout (constant_id = 8) const int TN = 2;

shared float buf_a[BM * (BK+1)];
shared float buf_b[BN * (BK+1)];

void main() {
    const int blocks_x = (p.M + BM - 1) / BM;
    const int ir = int(gl_WorkGroupID.x) % blocks_x;
    const int ik = int(gl_WorkGroupID.x) / blocks_x;
    const int ic = int(gl_WorkGroupID.y);

    const int stride_a = p.stride_a / QUANT_K;

    const int warp_i = int(gl_LocalInvocationID.x / WARP);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    const int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    const int WSUBM = WM / WMITER;
    const int WSUBN = WN / WNITER;

    const int tiw = int(gl_LocalInvocationID.x % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);

    const int loadr = int(gl_LocalInvocationID.x % (BK / 4));
    const int loadc = int(gl_LocalInvocationID.x / (BK / 4));

    const int loadstride = int(gl_WorkGroupSize.x * 4) / BK;

    const int start_k = ik * p.k_split;
    const int end_k = (ik + 1) * p.k_split;

    int pos_b = ic * BN * p.stride_b / 4 + start_k / 4;

    float sums[WMITER * TM * WNITER * TN];
    float cache_a[WMITER * TM];
    float cache_b[WNITER * TN];

    [[unroll]] for (int i = 0; i < WMITER*TM*WNITER*TN; i++) {
        sums[i] = 0.0f;
    }

    [[unroll]] for (int block = start_k; block < end_k; block += BK) {
        [[unroll]] for (int l = 0; l < BM; l += loadstride) {
            const int row = (block + loadr * 4) / QUANT_K;
            const int qi = (block + loadr * 4) % QUANT_K;
            const block_q4_0 blk = data_a[(ir * BM + loadc + l) * stride_a + row];
            const float d = float(blk.d);

            int x0, x1, x2, x3;
            if (qi < 16) {
                x0 = (blk.qs[qi + 0] & 0x0F) - 8;
                x1 = (blk.qs[qi + 1] & 0x0F) - 8;
                x2 = (blk.qs[qi + 2] & 0x0F) - 8;
                x3 = (blk.qs[qi + 3] & 0x0F) - 8;
            } else {
                x0 = (blk.qs[qi + 0] >> 4) - 8;
                x1 = (blk.qs[qi + 1] >> 4) - 8;
                x2 = (blk.qs[qi + 2] >> 4) - 8;
                x3 = (blk.qs[qi + 3] >> 4) - 8;
            }

            buf_a[(loadc + l) * (BK+1) + loadr * 4 + 0] = x0*d;
            buf_a[(loadc + l) * (BK+1) + loadr * 4 + 1] = x1*d;
            buf_a[(loadc + l) * (BK+1) + loadr * 4 + 2] = x2*d;
            buf_a[(loadc + l) * (BK+1) + loadr * 4 + 3] = x3*d;
        }
        [[unroll]] for (int l = 0; l < BN; l += loadstride) {
            vec4 tmp = data_b[pos_b + (loadc + l) * p.stride_b / 4 + loadr];
            buf_b[(loadc + l) * (BK+1) + loadr * 4 + 0] = tmp.x;
            buf_b[(loadc + l) * (BK+1) + loadr * 4 + 1] = tmp.y;
            buf_b[(loadc + l) * (BK+1) + loadr * 4 + 2] = tmp.z;
            buf_b[(loadc + l) * (BK+1) + loadr * 4 + 3] = tmp.w;
        }

        barrier();

        pos_b += BK / 4;

        for (int i = 0; i < min(BK, p.K - block); i++) {
            // Load from shared into cache
            [[unroll]] for (int wsir = 0; wsir < WMITER; wsir++) {
                [[unroll]] for (int j = 0; j < TM; j++) {
                    cache_a[wsir * TM + j] = buf_a[(warp_r * WM + wsir * WSUBM + tiwr * TM + j) * (BK+1) + i];
                }
            }
            [[unroll]] for (int wsic = 0; wsic < WNITER; wsic++) {
                [[unroll]] for (int j = 0; j < TN; j++) {
                    cache_b[wsic * TN + j] = buf_b[(warp_c * WN + wsic * WSUBN + tiwc * TN + j) * (BK+1) + i];
                }
            }

            [[unroll]] for (int wsic = 0; wsic < WNITER; wsic++) {
                [[unroll]] for (int wsir = 0; wsir < WMITER; wsir++) {
                    [[unroll]] for (int cc = 0; cc < TN; cc++) {
                        [[unroll]] for (int cr = 0; cr < TM; cr++) {
                            sums[(wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr] += cache_a[wsir * TM + cr] * cache_b[wsic * TN + cc];
                        }
                    }
                }
            }
        }

        barrier();
    }

    const int dr = ir * BM + warp_r * WM;
    const int dc = ic * BN + warp_c * WN;

    const int k_split_offset = ik * p.M * p.N;

    [[unroll]] for (int wsic = 0; wsic < WNITER; wsic++) {
        [[unroll]] for (int wsir = 0; wsir < WMITER; wsir++) {

            const int dr_warp = dr + wsir * WSUBM + tiwr * TM;
            const int dc_warp = dc + wsic * WSUBN + tiwc * TN;
            [[unroll]] for (int cc = 0; cc < TN; cc++) {
                [[unroll]] for (int cr = 0; cr < TM; cr++) {
                    if (dr_warp + cr < p.M && dc_warp + cc < p.N) {
                        data_d[k_split_offset + (dc_warp + cc) * p.stride_d + dr_warp + cr] = sums[(wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr];
                    }
                }
            }
        }
    }
}
