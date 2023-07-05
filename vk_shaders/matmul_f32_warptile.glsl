#version 450

#define BM 128
#define BN 128
#define BK 16
#define WM 64
#define WN 64
#define WMITER 4
#define TM 4
#define TN 8

#define WARP 32

#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A { float data_a[]; };
layout (binding = 1) readonly buffer B { float data_b[]; };
layout (binding = 2) writeonly buffer D { float data_d[]; };

layout (push_constant) uniform parameter
{
    int M;
    int N;
    int K;
    int stride_a;
    int stride_b;
    int stride_d;
} p;

shared float buf_a[BM * (BK+1)];
shared float buf_b[BN * (BK+1)];

void main() {
    const int ir = int(gl_WorkGroupID.x);
    const int ic = int(gl_WorkGroupID.y);

    const int warp_i = int(gl_LocalInvocationID.x / WARP);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    const int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    const int WSUBM = WM / WMITER;
    const int WSUBN = WN / WNITER;

    const int tiw = int(gl_LocalInvocationID.x % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);

    const int loadr = int(gl_LocalInvocationID.x % BK);
    const int loadc = int(gl_LocalInvocationID.x / BK);

    const int loadstride = int(gl_WorkGroupSize.x);

    int pos_a = ir * BM * p.stride_a;
    int pos_b = ic * BN * p.stride_b;

    float sums[WMITER * TM * WNITER * TN];
    float cache_a[WMITER * TM];
    float cache_b[WNITER * TN];

    [[unroll]] for (int i = 0; i < WMITER*TM*WNITER*TN; i++) {
        sums[i] = 0.0f;
    }

    [[unroll]] for (int block = 0; block < p.K; block += BK) {
        [[unroll]] for (int l = 0; l < BM * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            if (ir * BM + loadc + lc < p.M && block + loadr + lr < p.K) {
                buf_a[(loadc + lc) * (BK+1) + loadr + lr] = data_a[pos_a + (loadc + lc) * p.stride_a + loadr + lr];
            } else {
                buf_a[(loadc + lc) * (BK+1) + loadr + lr] = 0.0f;
            }
        }
        [[unroll]] for (int l = 0; l < BN * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            if (ic * BN + loadc + lc < p.N && block + loadr + lr < p.K) {
                buf_b[(loadc + lc) * (BK+1) + loadr + lr] = data_b[pos_b + (loadc + lc) * p.stride_b + loadr + lr];
            } else {
                buf_b[(loadc + lc) * (BK+1) + loadr + lr] = 0.0f;
            }
        }

        barrier();

        pos_a += BK;
        pos_b += BK;

        [[unroll]] for (int i = 0; i < BK; i++) {
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

    [[unroll]] for (int wsic = 0; wsic < WNITER; wsic++) {
        [[unroll]] for (int wsir = 0; wsir < WMITER; wsir++) {

            const int dr_warp = dr + wsir * WSUBM + tiwr * TM;
            const int dc_warp = dc + wsic * WSUBN + tiwc * TN;
            [[unroll]] for (int cc = 0; cc < TN; cc++) {
                [[unroll]] for (int cr = 0; cr < TM; cr++) {
                    if (dr_warp + cr < p.M && dc_warp + cc < p.N) {
                        data_d[(dc_warp + cc) * p.stride_d + dr_warp + cr] = sums[(wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr];
                    }
                }
            }
        }
    }
}
