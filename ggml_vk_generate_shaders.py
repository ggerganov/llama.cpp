#!/usr/bin/env python

import asyncio
import os
import sys
from tempfile import NamedTemporaryFile

shader_f32 = """
#define FLOAT_TYPE float
"""
shader_f16 = """
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define FLOAT_TYPE float16_t
"""
shader_int8_ext = """
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
"""

# Type-specific defines
shader_f16_defines = """
#define QUANT_K 32
#define QUANT_R 2

#define A_TYPE float16_t
"""
shader_q4_0_defines = """
#define QUANT_K 32
#define QUANT_R 2

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};

#define A_TYPE block_q4_0
"""
shader_q4_1_defines = """
#define QUANT_K 32
#define QUANT_R 2

struct block_q4_1
{
    float16_t d;
    float16_t m;
    uint8_t qs[16];
};

#define A_TYPE block_q4_1
"""
shader_q5_0_defines = """
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#define QUANT_K 32
#define QUANT_R 2

struct block_q5_0
{
    float16_t d;
    uint16_t qh[2];
    uint8_t qs[16];
};

#define A_TYPE block_q5_0
"""
shader_q5_1_defines = """
#define QUANT_K 32
#define QUANT_R 2

struct block_q5_1
{
    float16_t d;
    float16_t m;
    uint qh;
    uint8_t qs[16];
};

#define A_TYPE block_q5_1
"""
shader_q8_0_defines = """
#define QUANT_K 32
#define QUANT_R 1

struct block_q8_0
{
    float16_t d;
    int8_t qs[32];
};

#define A_TYPE block_q8_0
"""

shader_q6_K_defines = """
#define QUANT_K 256

struct block_q6_K
{
    uint8_t ql[QUANT_K/2];
    uint8_t qh[QUANT_K/4];
    int8_t scales[QUANT_K/16];
    float16_t d;
};

#define A_TYPE block_q6_K
"""

# Dequant functions
shader_f16_dequant_func = """
#define DEQUANT_FUNC f16vec2 v = f16vec2(x[ib + 0], x[ib + 1]);
"""
shader_f16_dequant_func_compat = """
#define DEQUANT_FUNC vec2 v = vec2(x[ib + 0], x[ib + 1]);
"""

shader_q4_0_dequant_func = """
#define DEQUANT_FUNC const float16_t d = x[ib].d; \
const uint8_t vui = x[ib].qs[iqs]; \
f16vec2 v = f16vec2(vui & 0xF, vui >> 4); \
v = (v - 8.0hf)*d;
"""
shader_q4_0_dequant_func_compat = """
#define DEQUANT_FUNC const float d = float(x[ib].d); \
const uint vui = uint(x[ib].qs[iqs]); \
vec2 v = vec2(vui & 0xF, vui >> 4); \
v = (v - 8.0f)*d;
"""

shader_q4_1_dequant_func = """
#define DEQUANT_FUNC const float16_t d = x[ib].d; \
const float16_t m = x[ib].m; \
const uint8_t vui = x[ib].qs[iqs]; \
f16vec2 v = f16vec2(vui & 0xF, vui >> 4); \
v = v*d + m;
"""
shader_q4_1_dequant_func_compat = """
#define DEQUANT_FUNC const float d = float(x[ib].d); \
const float m = float(x[ib].m); \
const uint vui = uint(x[ib].qs[iqs]); \
vec2 v = vec2(vui & 0xF, vui >> 4); \
v = v*d + m;
"""

shader_q5_0_dequant_func = """
#define DEQUANT_FUNC const float16_t d = x[ib].d; \
const uint uint_qh = uint(x[ib].qh[1]) << 16 | x[ib].qh[0]; \
const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10); \
const uint8_t vui = x[ib].qs[iqs]; \
f16vec2 v = f16vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y); \
v = (v - 16.0hf) * d;
"""
shader_q5_0_dequant_func_compat = """
#define DEQUANT_FUNC const float d = float(x[ib].d); \
const uint uint_qh = uint(x[ib].qh[1]) << 16 | x[ib].qh[0]; \
const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10); \
const uint vui = uint(x[ib].qs[iqs]); \
vec2 v = vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y); \
v = (v - 16.0f) * d;
"""

shader_q5_1_dequant_func = """
#define DEQUANT_FUNC const float16_t d = x[ib].d; \
const float16_t m = x[ib].m; \
const ivec2 qh = ivec2(((x[ib].qh >> iqs) << 4) & 0x10, (x[ib].qh >> (iqs + 12)) & 0x10); \
const uint8_t vui = x[ib].qs[iqs]; \
f16vec2 v = f16vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y); \
v = v*d + m;
"""
shader_q5_1_dequant_func_compat = """
#define DEQUANT_FUNC const float d = float(x[ib].d); \
const float m = float(x[ib].m); \
const ivec2 qh = ivec2(((x[ib].qh >> iqs) << 4) & 0x10, (x[ib].qh >> (iqs + 12)) & 0x10); \
const uint vui = uint(x[ib].qs[iqs]); \
vec2 v = vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y); \
v = v*d + m;
"""

shader_q8_0_dequant_func = """
#define DEQUANT_FUNC const float16_t d = x[ib].d; \
f16vec2 v = f16vec2(x[ib].qs[iqs], x[ib].qs[iqs + 1]); \
v = v * d;
"""
shader_q8_0_dequant_func_compat = """
#define DEQUANT_FUNC const float d = float(x[ib].d); \
vec2 v = vec2(int(x[ib].qs[iqs]), int(x[ib].qs[iqs + 1])); \
v = v * d;
"""

# MULMAT

mulmat_head = """#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require

#define WARP 32

#ifndef LOAD_VEC
#define LOAD_VEC 1
#endif
"""

mulmat_body = """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

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

shared FLOAT_TYPE buf_a[BM * (BK+1)];
shared FLOAT_TYPE buf_b[BN * (BK+1)];

void main() {
    const int blocks_x = (p.M + BM - 1) / BM;
    const int ir = int(gl_WorkGroupID.x) % blocks_x;
    const int ik = int(gl_WorkGroupID.x) / blocks_x;
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

    const int loadr = int(gl_LocalInvocationID.x % (BK / LOAD_VEC));
    const int loadc = int(gl_LocalInvocationID.x / (BK / LOAD_VEC));

    const int loadstride = int(gl_WorkGroupSize.x * LOAD_VEC) / BK;

    const int start_k = ik * p.k_split;
    const int end_k = (ik + 1) * p.k_split;

    int pos_a = ir * BM * p.stride_a / LOAD_VEC + start_k / LOAD_VEC;
    int pos_b = ic * BN * p.stride_b / LOAD_VEC + start_k / LOAD_VEC;

    D_TYPE sums[WMITER * TM * WNITER * TN];
    FLOAT_TYPE cache_a[WMITER * TM];
    FLOAT_TYPE cache_b[WNITER * TN];

    [[unroll]] for (int i = 0; i < WMITER*TM*WNITER*TN; i++) {
        sums[i] = 0.0f;
   }

    [[unroll]] for (int block = start_k; block < end_k; block += BK) {
        [[unroll]] for (int l = 0; l < BM; l += loadstride) {
#if LOAD_VEC == 8
            const int idx = pos_a + (loadc + l) * p.stride_a / LOAD_VEC + loadr;
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 0] = FLOAT_TYPE(data_a[idx][0].x);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 1] = FLOAT_TYPE(data_a[idx][0].y);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 2] = FLOAT_TYPE(data_a[idx][0].z);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 3] = FLOAT_TYPE(data_a[idx][0].w);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 4] = FLOAT_TYPE(data_a[idx][1].x);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 5] = FLOAT_TYPE(data_a[idx][1].y);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 6] = FLOAT_TYPE(data_a[idx][1].z);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 7] = FLOAT_TYPE(data_a[idx][1].w);
#elif LOAD_VEC == 4
            const int idx = pos_a + (loadc + l) * p.stride_a / LOAD_VEC + loadr;
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 0] = FLOAT_TYPE(data_a[idx].x);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 1] = FLOAT_TYPE(data_a[idx].y);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 2] = FLOAT_TYPE(data_a[idx].z);
            buf_a[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 3] = FLOAT_TYPE(data_a[idx].w);
#else
            if (ir * BM + loadc + l < p.M && block + loadr < p.K) {
                buf_a[(loadc + l) * (BK+1) + loadr] = FLOAT_TYPE(data_a[pos_a + (loadc + l) * p.stride_a + loadr]);
           } else {
                buf_a[(loadc + l) * (BK+1) + loadr] = FLOAT_TYPE(0.0f);
           }
#endif
       }
        [[unroll]] for (int l = 0; l < BN; l += loadstride) {
#if LOAD_VEC == 8
            const int idx = pos_b + (loadc + l) * p.stride_b / LOAD_VEC + loadr;
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 0] = FLOAT_TYPE(data_b[idx][0].x);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 1] = FLOAT_TYPE(data_b[idx][0].y);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 2] = FLOAT_TYPE(data_b[idx][0].z);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 3] = FLOAT_TYPE(data_b[idx][0].w);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 4] = FLOAT_TYPE(data_b[idx][1].x);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 5] = FLOAT_TYPE(data_b[idx][1].y);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 6] = FLOAT_TYPE(data_b[idx][1].z);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 7] = FLOAT_TYPE(data_b[idx][1].w);
#elif LOAD_VEC == 4
            const int idx = pos_b + (loadc + l) * p.stride_b / LOAD_VEC + loadr;
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 0] = FLOAT_TYPE(data_b[idx].x);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 1] = FLOAT_TYPE(data_b[idx].y);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 2] = FLOAT_TYPE(data_b[idx].z);
            buf_b[(loadc + l) * (BK+1) + loadr * LOAD_VEC + 3] = FLOAT_TYPE(data_b[idx].w);
#else
            if (ic * BN + loadc + l < p.N && block + loadr < p.K) {
                buf_b[(loadc + l) * (BK+1) + loadr] = FLOAT_TYPE(data_b[pos_b + (loadc + l) * p.stride_b + loadr]);
           } else {
                buf_b[(loadc + l) * (BK+1) + loadr] = FLOAT_TYPE(0.0f);
           }
#endif
       }

        barrier();

        pos_a += BK / LOAD_VEC;
        pos_b += BK / LOAD_VEC;

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
                            sums[(wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr] += D_TYPE(cache_a[wsir * TM + cr]) * D_TYPE(cache_b[wsic * TN + cc]);
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
"""

mulmat_split_k_reduce_src = """#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) buffer A {float data[];};

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
"""

# DEQUANT SHADER
dequant_head = """#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
"""

dequant_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE x[];};
layout (binding = 1) writeonly buffer D {D_TYPE y[];};

layout (push_constant) uniform parameter
{
    int M;
    int K;
    int stride_a;
    int stride_b;
} p;

void main() {
    const int i = int(gl_GlobalInvocationID.x);

    // Transposed
    const int row = i % (p.K / QUANT_K);
    const int col = i / (p.K / QUANT_K);

    if (row * QUANT_K >= p.K || col >= p.M) {
        return;
   }

    const int stride_a = p.stride_a / QUANT_K;

    const int ib = col * stride_a + row;

    const int y_offset = QUANT_R == 1 ? 1 : QUANT_K/2;
    const int step = QUANT_R == 1 ? 2 : 1;

    [[unroll]] for (int iqs = 0; iqs < QUANT_K/QUANT_R; iqs += step) {
        DEQUANT_FUNC

        y[col * p.stride_b + row*QUANT_K + iqs + 0       ] = D_TYPE(v.x);
        y[col * p.stride_b + row*QUANT_K + iqs + y_offset] = D_TYPE(v.y);
   }
}
"""

# K-quants
dequant_q6_K_body = """
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE x[];};
layout (binding = 1) writeonly buffer D {D_TYPE y[];};

layout (push_constant) uniform parameter
{
    int M;
    int K;
    int stride_a;
    int stride_b;
} p;

void main() {
    for (int wgy = 0; wgy < 256; wgy++) {
        const int i = int(gl_WorkGroupID.x * 256 + wgy);
        if (i >= p.M * p.K / QUANT_K) {
            return;
       }
        const int tid = int(gl_LocalInvocationID.x);
        const int ip = tid / 32;
        const int il = tid - 32 * ip;
        const int is = 8 * ip + il / 16;

        const int y_idx = i * QUANT_K + 128 * ip + il;

        const int ql_idx = 64 * ip + il;
        const uint8_t qh = x[i].qh[32 * ip + il];

        const FLOAT_TYPE d = FLOAT_TYPE(x[i].d);

        y[y_idx +  0] = D_TYPE(d * FLOAT_TYPE(x[i].scales[is + 0] * (int8_t((x[i].ql[ql_idx +  0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32)));
        y[y_idx + 32] = D_TYPE(d * FLOAT_TYPE(x[i].scales[is + 2] * (int8_t((x[i].ql[ql_idx + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32)));
        y[y_idx + 64] = D_TYPE(d * FLOAT_TYPE(x[i].scales[is + 4] * (int8_t((x[i].ql[ql_idx +  0] >>  4) | (((qh >> 4) & 3) << 4)) - 32)));
        y[y_idx + 96] = D_TYPE(d * FLOAT_TYPE(x[i].scales[is + 6] * (int8_t((x[i].ql[ql_idx + 32] >>  4) | (((qh >> 6) & 3) << 4)) - 32)));
   }
}
"""

# Mul Mat Vec
mul_mat_vec_head = """#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
"""

mul_mat_vec_body = """
layout(local_size_x = QUANT_K, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE x[];};
layout (binding = 1) readonly buffer B {B_TYPE y[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

layout (push_constant) uniform parameter
{
    int ncols;
} p;

shared FLOAT_TYPE tmp[QUANT_K];

void main() {
    const int block_size = int(gl_WorkGroupSize.x);
    const int row = int(gl_WorkGroupID.x);
    const int tid = int(gl_LocalInvocationID.x);

    const int y_offset = QUANT_R == 1 ? 1 : QUANT_K/2;

    tmp[tid] = FLOAT_TYPE(0.0f);

    [[unroll]] for (int i = 0; i < p.ncols/block_size; i += 2) {
        const int col = i*block_size + 2*tid;
        const int ib = (row*p.ncols + col)/QUANT_K; // block index
        const int iqs = (col%QUANT_K)/QUANT_R; // quant index
        const int iybs = col - col%QUANT_K; // y block start index

        DEQUANT_FUNC

        // matrix multiplication
        tmp[tid] += FLOAT_TYPE(v.x) * FLOAT_TYPE(y[iybs + iqs + 0]);
        tmp[tid] += FLOAT_TYPE(v.y) * FLOAT_TYPE(y[iybs + iqs + y_offset]);
   }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s = block_size/2; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
       }
        barrier();
   }
    if (tid == 0) {
        dst[row] = D_TYPE(tmp[0]);
   }
}
"""

mul_mat_vec_q6_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE x[];};
layout (binding = 1) readonly buffer B {B_TYPE y[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

layout (push_constant) uniform parameter
{
    int ncols;
} p;

shared FLOAT_TYPE tmp[32];

void main() {
    const int row = int(gl_WorkGroupID.x);

    const int num_blocks_per_row = p.ncols / QUANT_K;
    const int ib0 = row*num_blocks_per_row;

    const int tid = int(gl_LocalInvocationID.x)/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = int(gl_LocalInvocationID.x)%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;            // 16 or 8

    const int v_im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int v_in = tid - step*v_im;                      // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*v_in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * v_in;                               // 0, 4, 8, ..., 28
    const int is = v_in / 4;
#endif

    const int ql_offset = 64*v_im + l0;
    const int qh_offset = 32*v_im + l0;
    const int s_offset  =  8*v_im + is;
    const int y_offset = 128*v_im + l0;

    tmp[16 * ix + tid] = FLOAT_TYPE(0.0); // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const int y_idx    = i * QUANT_K + y_offset;

        const FLOAT_TYPE d = FLOAT_TYPE(x[ib0 + i].d);

#if K_QUANTS_PER_ITERATION == 1
        FLOAT_TYPE sum = FLOAT_TYPE(y[y_idx +  0]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 0]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset +  0] & 0xF) | ((x[ib0 + i].qh[qh_offset +  0] & 0x03) << 4)) - 32)
                       + FLOAT_TYPE(y[y_idx + 16]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 1]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + 16] & 0xF) | ((x[ib0 + i].qh[qh_offset + 16] & 0x03) << 4)) - 32)
                       + FLOAT_TYPE(y[y_idx + 32]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 2]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + 32] & 0xF) | ((x[ib0 + i].qh[qh_offset +  0] & 0x0c) << 2)) - 32)
                       + FLOAT_TYPE(y[y_idx + 48]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 3]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + 48] & 0xF) | ((x[ib0 + i].qh[qh_offset + 16] & 0x0c) << 2)) - 32)
                       + FLOAT_TYPE(y[y_idx + 64]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 4]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset +  0]  >> 4) | ((x[ib0 + i].qh[qh_offset +  0] & 0x30) >> 0)) - 32)
                       + FLOAT_TYPE(y[y_idx + 80]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 5]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + 16]  >> 4) | ((x[ib0 + i].qh[qh_offset + 16] & 0x30) >> 0)) - 32)
                       + FLOAT_TYPE(y[y_idx + 96]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 6]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + 32]  >> 4) | ((x[ib0 + i].qh[qh_offset +  0] & 0xc0) >> 2)) - 32)
                       + FLOAT_TYPE(y[y_idx +112]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 7]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + 48]  >> 4) | ((x[ib0 + i].qh[qh_offset + 16] & 0xc0) >> 2)) - 32);
        tmp[16 * ix + tid] += sum;
#else
        FLOAT_TYPE sum = FLOAT_TYPE(0.0);
        for (int l = 0; l < 4; ++l) {
            sum += FLOAT_TYPE(y[y_idx + l+ 0]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 0]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + l+ 0] & 0xF) | (((x[ib0 + i].qh[qh_offset + l] >> 0) & 3) << 4)) - 32)
                 + FLOAT_TYPE(y[y_idx + l+32]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 2]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + l+32] & 0xF) | (((x[ib0 + i].qh[qh_offset + l] >> 2) & 3) << 4)) - 32)
                 + FLOAT_TYPE(y[y_idx + l+64]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 4]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + l+ 0]  >> 4) | (((x[ib0 + i].qh[qh_offset + l] >> 4) & 3) << 4)) - 32)
                 + FLOAT_TYPE(y[y_idx + l+96]) * FLOAT_TYPE(x[ib0 + i].scales[s_offset + 6]) * d * FLOAT_TYPE(int8_t((x[ib0 + i].ql[ql_offset + l+32]  >> 4) | (((x[ib0 + i].qh[qh_offset + l] >> 6) & 3) << 4)) - 32);
       }
        tmp[16 * ix + tid] += sum;
#endif
   }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
       }
        barrier();
   }
    if (tid == 0) {
        dst[row] = D_TYPE(tmp[0]);
   }
}
"""

# F16 to F32
f32_to_f16_src = """#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {float data_a[];};
layout (binding = 1) writeonly buffer D {float16_t data_b[];};

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

    if (row < p.K && col < p.M) {
        data_b[col * p.stride_b + row] = float16_t(data_a[col * p.stride_a + row]);
   }
}
"""

# MUL F32
mul_f32_src = """#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) buffer X {X_TYPE data_x[];};
layout (binding = 1) buffer Y {Y_TYPE data_y[];};
layout (binding = 2) buffer D {D_TYPE data_d[];};

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

    data_d[p.d_offset + y * p.stride_d + x] = D_TYPE(data_x[p.x_offset + y * p.stride_x + x]) * D_TYPE(data_y[p.y_offset + x]);
}
"""

# ADD
add_head = """
#version 450

#extension GL_EXT_shader_16bit_storage : require
"""

add_body = """
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
layout (binding = 0) buffer X {X_TYPE data_x[];};
layout (binding = 1) buffer Y {Y_TYPE data_y[];};
layout (binding = 2) buffer D {D_TYPE data_d[];};

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

    data_d[p.d_offset + y * p.stride_d + x] = D_TYPE(FLOAT_TYPE(data_x[p.x_offset + y * p.stride_x + x]) + FLOAT_TYPE(data_y[p.y_offset + x]));
}
"""

# SCALE
scale_src = """#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) buffer X {X_TYPE data_x[];};
layout (binding = 1) buffer D {D_TYPE data_d[];};

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

    data_d[p.d_offset + y * p.stride_d + x] = D_TYPE(data_x[p.x_offset + y * p.stride_x + x]) * D_TYPE(p.scale);
}
"""


VK_NUM_TYPES = 16

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15


type_names = {
    GGML_TYPE_F32: "f32",
    GGML_TYPE_F16: "f16",
    GGML_TYPE_Q4_0: "q4_0",
    GGML_TYPE_Q4_1: "q4_1",
    GGML_TYPE_Q5_0: "q5_0",
    GGML_TYPE_Q5_1: "q5_1",
    GGML_TYPE_Q8_0: "q8_0",
    GGML_TYPE_Q8_1: "q8_1",
    GGML_TYPE_Q2_K: "q2_K",
    GGML_TYPE_Q3_K: "q3_K",
    GGML_TYPE_Q4_K: "q4_K",
    GGML_TYPE_Q5_K: "q5_K",
    GGML_TYPE_Q6_K: "q6_K",
    GGML_TYPE_Q8_K: "q8_K",
}

K_QUANTS_PER_ITERATION = 1


async def string_to_spv_file(name, code, defines, fp16):
    with NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()

        cmd = ["glslc", "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", f.name, "-o", os.path.join("vk_shaders", f"{name}{'_fp32' if not fp16 else ''}.comp")]

        cmd.extend([f"-D{key}={value}" for key, value in defines.items()])

        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()

        stdout = stdout.decode()
        error = stderr.decode()

        if proc.returncode:
            # Generate preprocessed code
            cmd = ["glslc", "-E", f.name]
            cmd.extend([f"-D{key}={value}" for key, value in defines.items()])

            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            stdout, stderr = await proc.communicate()

            print(" ".join(cmd))

            if proc.returncode:
                raise RuntimeError(f"{name=} {f.name=} {stdout=} {stderr=}")

            preprocessed_code = stdout.decode()

            cmd.extend([f"-D{key}={value}" for key, value in defines.items()])
            code_with_lines = "\n".join([f"{i}: {line}" for i, line in enumerate(preprocessed_code.splitlines())])
            print(f"ERROR compiling {name}\n\n{code_with_lines}\n\n{error=}")
            sys.exit(proc.returncode)


async def main():
    print("ggml_vulkan: Generating and compiling shaders to SPIR-V")

    os.makedirs("vk_shaders", exist_ok=True)

    for fp16 in (False, True):
        # mulmat
        if fp16:
            shader_float_type = shader_f16
            load_vec = "8"
            vec_type_f16 = "f16mat2x4"
            vec_type = "mat2x4"
        else:
            shader_float_type = shader_f32
            load_vec = "4"
            vec_type_f16 = "f16vec4"
            vec_type = "vec4"

        tasks = []

        stream = []
        stream.extend((mulmat_head, shader_float_type, mulmat_body));
        tasks.append(string_to_spv_file("matmul_f32_l", "".join(stream), {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f32_m", "".join(stream), {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f32_s", "".join(stream), {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f32_aligned_l", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f32_aligned_m", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f32_aligned_s", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear();
        stream.extend((mulmat_head, shader_float_type, mulmat_body));
        tasks.append(string_to_spv_file("matmul_f16_l", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float16_t", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_m", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float16_t", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_s", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float16_t", "D_TYPE": "float"}, fp16))

        tasks.append(string_to_spv_file("matmul_f16_aligned_l", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type_f16, "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_aligned_m", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type_f16, "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_aligned_s", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type_f16, "D_TYPE": "float"}, fp16))

        tasks.append(string_to_spv_file("matmul_f16_f32_l", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_f32_m", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_f32_s", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_f32_aligned_l", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_f32_aligned_m", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv_file("matmul_f16_f32_aligned_s", "".join(stream), {"LOAD_VEC": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        # Build dequant shaders
        tasks.append(string_to_spv_file("f32_to_f16", f32_to_f16_src, {}, fp16))

        for i in range(0, VK_NUM_TYPES):
            stream.clear();

            stream.extend((dequant_head, shader_int8_ext, shader_float_type));

            if i == GGML_TYPE_F16:
                stream.extend((shader_f16_defines, shader_f16_dequant_func_compat if not fp16 else shader_f16_dequant_func, dequant_body))
            elif i == GGML_TYPE_Q4_0:
                stream.extend((shader_q4_0_defines, shader_q4_0_dequant_func_compat if not fp16 else shader_q4_0_dequant_func, dequant_body))
            elif i == GGML_TYPE_Q4_1:
                stream.extend((shader_q4_1_defines, shader_q4_1_dequant_func_compat if not fp16 else shader_q4_1_dequant_func, dequant_body))
            elif i == GGML_TYPE_Q5_0:
                stream.extend((shader_q5_0_defines, shader_q5_0_dequant_func_compat if not fp16 else shader_q5_0_dequant_func, dequant_body))
            elif i == GGML_TYPE_Q5_1:
                stream.extend((shader_q5_1_defines, shader_q5_1_dequant_func_compat if not fp16 else shader_q5_1_dequant_func, dequant_body))
            elif i == GGML_TYPE_Q8_0:
                stream.extend((shader_q8_0_defines, shader_q8_0_dequant_func_compat if not fp16 else shader_q8_0_dequant_func, dequant_body))
            elif i == GGML_TYPE_Q6_K:
                stream.extend((shader_q6_K_defines, dequant_q6_K_body))
            else:
                continue

            tasks.append(string_to_spv_file(f"dequant_{type_names[i]}", "".join(stream), {"D_TYPE": "float16_t"}, fp16))

        # mul mat vec
        for i in range(0, VK_NUM_TYPES):
            stream.clear();
            stream.extend((mul_mat_vec_head, shader_int8_ext, shader_float_type))

            if i == GGML_TYPE_F16:
                stream.extend((shader_f16_defines, shader_f16_dequant_func_compat if not fp16 else shader_f16_dequant_func, mul_mat_vec_body))
            elif i == GGML_TYPE_Q4_0:
                stream.extend((shader_q4_0_defines, shader_q4_0_dequant_func_compat if not fp16 else shader_q4_0_dequant_func, mul_mat_vec_body))
            elif i == GGML_TYPE_Q4_1:
                stream.extend((shader_q4_1_defines, shader_q4_1_dequant_func_compat if not fp16 else shader_q4_1_dequant_func, mul_mat_vec_body))
            elif i == GGML_TYPE_Q5_0:
                stream.extend((shader_q5_0_defines, shader_q5_0_dequant_func_compat if not fp16 else shader_q5_0_dequant_func, mul_mat_vec_body))
            elif i == GGML_TYPE_Q5_1:
                stream.extend((shader_q5_1_defines, shader_q5_1_dequant_func_compat if not fp16 else shader_q5_1_dequant_func, mul_mat_vec_body))
            elif i == GGML_TYPE_Q8_0:
                stream.extend((shader_q8_0_defines, shader_q8_0_dequant_func_compat if not fp16 else shader_q8_0_dequant_func, mul_mat_vec_body))
            elif i == GGML_TYPE_Q6_K:
                stream.extend((shader_q6_K_defines, mul_mat_vec_q6_K_body))
            else:
                continue

            tasks.append(string_to_spv_file(f"mul_mat_vec_{type_names[i]}", "".join(stream), {"B_TYPE": "float", "D_TYPE": "float16_t", "K_QUANTS_PER_ITERATION": K_QUANTS_PER_ITERATION}, fp16))
            tasks.append(string_to_spv_file(f"mul_mat_vec_{type_names[i]}_f32", "".join(stream), {"B_TYPE": "float", "D_TYPE": "float", "K_QUANTS_PER_ITERATION": K_QUANTS_PER_ITERATION}, fp16))

        # add
        stream.clear();

        stream.extend((add_head, shader_float_type, add_body))
        tasks.append(string_to_spv_file("add_f32", "".join(stream), {"X_TYPE": "float", "Y_TYPE": "float", "D_TYPE": "float"}, fp16))

        stream.clear();
        stream.extend((add_head, shader_float_type, add_body))
        tasks.append(string_to_spv_file("add_f16_f32_f16", "".join(stream), {"X_TYPE": "float16_t", "Y_TYPE": "float", "D_TYPE": "float16_t"}, fp16))

        # Static shaders
        tasks.append(string_to_spv_file("split_k_reduce", mulmat_split_k_reduce_src, {}, fp16))
        tasks.append(string_to_spv_file("mul_f32", mul_f32_src, {"X_TYPE": "float", "Y_TYPE": "float", "D_TYPE": "float"}, fp16))

        tasks.append(string_to_spv_file("scale_f32", scale_src, {"X_TYPE": "float", "D_TYPE": "float"}, fp16))

        await asyncio.gather(*tasks)


asyncio.run(main())
